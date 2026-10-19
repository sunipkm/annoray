//! SZ 2 (2.1, as the HDF5 SZ filter bundled with hdf5plugin writes it) decoding of `float`
//! and `double` data, bit for bit as the reference decodes.
//!
//! A buffer (usually Zstandard- or zlib-compressed as a whole) starts with the version, a
//! flags byte, the compressor's parameters and the value count. Then one of:
//!
//! - the values themselves (stored losslessly when that was smaller), or one value for all;
//! - blocked prediction (2-D to 4-D data): per block, a first-order Lorenzo predictor or a
//!   linear regression whose coefficients are themselves predicted and quantized, then
//!   Huffman-coded quantization indexes and the values that could not be predicted;
//! - 1-D Lorenzo prediction, the unpredictable values packed by their leading bytes shared
//!   with the previous one, significant bytes and residual bits;
//! - point-wise relative bounds (the "accelerated" logarithmic scheme of the MSST'19 paper):
//!   multiplicative prediction, quantization by a table of powers, signs restored at the end.
//!
//! Not supported: the non-regression 2-D/3-D Lorenzo path, point-wise relative bounds without
//! acceleration, and time-based compression. Multiply-adds are fused where the plugin's build
//! fuses them (see [`crate::sz3`]); the power table uses the platform's `pow`.

use crate::Error;
use crate::bits::{BitReader, corrupt};
use crate::sz3::{Huffman, Real};

const CODEC: &str = "sz";

fn err() -> Error {
    corrupt(CODEC)
}

/// The value types, as SZ 2 serializes them.
pub trait SzReal: Real {
    /// Bytes of the compressor's parameters in the header (28 for `float`, 36 for `double`).
    const META: usize;
    /// From big-endian bytes.
    fn from_be(bytes: &[u8]) -> Option<Self>;
    /// The value with its sign bit set.
    fn negative(self) -> Self;
}

impl SzReal for f32 {
    const META: usize = 28;
    fn from_be(bytes: &[u8]) -> Option<Self> {
        Some(Self::from_be_bytes(bytes.try_into().ok()?))
    }
    fn negative(self) -> Self {
        Self::from_bits(self.to_bits() | 0x8000_0000)
    }
}

impl SzReal for f64 {
    const META: usize = 36;
    fn from_be(bytes: &[u8]) -> Option<Self> {
        Some(Self::from_be_bytes(bytes.try_into().ok()?))
    }
    fn negative(self) -> Self {
        Self::from_bits(self.to_bits() | 0x8000_0000_0000_0000)
    }
}

/// Fields of a buffer, read in order.
struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Rd<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).ok_or_else(err)?;
        let s = self.b.get(self.at..end).ok_or_else(err)?;
        self.at = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        self.bytes(1)?.first().copied().ok_or_else(err)
    }

    fn be_u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_be_bytes(
            self.bytes(4)?.try_into().map_err(|_| err())?,
        ))
    }

    fn be_size(&mut self, wide: bool) -> Result<usize, Error> {
        let v = if wide {
            u64::from_be_bytes(self.bytes(8)?.try_into().map_err(|_| err())?)
        } else {
            u64::from(self.be_u32()?)
        };
        usize::try_from(v).map_err(|_| err())
    }

    fn ne_size(&mut self) -> Result<usize, Error> {
        let v = u64::from_ne_bytes(self.bytes(8)?.try_into().map_err(|_| err())?);
        usize::try_from(v).map_err(|_| err())
    }

    fn be_real<T: SzReal>(&mut self) -> Result<T, Error> {
        T::from_be(self.bytes(T::SIZE)?).ok_or_else(err)
    }

    fn be_f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_be_bytes(
            self.bytes(4)?.try_into().map_err(|_| err())?,
        ))
    }

    fn be_f64(&mut self) -> Result<f64, Error> {
        Ok(f64::from_be_bytes(
            self.bytes(8)?.try_into().map_err(|_| err())?,
        ))
    }

    /// `n` native-endian values (after checking the buffer holds them).
    fn ne_reals<T: SzReal>(&mut self, n: usize) -> Result<Vec<T>, Error> {
        let b = self.bytes(n.checked_mul(T::SIZE).ok_or_else(err)?)?;
        b.chunks_exact(T::SIZE)
            .map(|c| T::from_le(c).ok_or_else(err))
            .collect()
    }

    /// A Huffman tree as SZ 2 stores it: the node count, another word, the tree.
    fn tree(&mut self) -> Result<Huffman, Error> {
        let count = usize::try_from(self.be_u32()?).map_err(|_| err())?;
        self.bytes(4)?;
        let rest = self.b.get(self.at..).ok_or_else(err)?;
        let swap = rest.first().copied() == Some(1);
        let (tree, used) = Huffman::tree(rest, count, swap)?;
        self.bytes(used)?;
        Ok(tree)
    }
}

/// What the decoders need of the header and the flat bytes after it.
struct Tdps<'a> {
    pw_rel: bool,
    accelerate: bool,
    intervals: usize,
    median: f64,
    req_length: u8,
    plus_bits: u8,
    real_precision: f64,
    type_array: &'a [u8],
    exact_count: usize,
    pwr_bytes: &'a [u8],
    lead: &'a [u8],
    exact_mid: &'a [u8],
    residual: &'a [u8],
    min_log: f64,
}

/// Unpredictable values: leading bytes shared with the previous one, the significant bytes,
/// the residual bits of a last byte.
struct Exact<'a> {
    lead: Vec<u8>,
    next_lead: usize,
    mid: &'a [u8],
    mid_at: usize,
    residual: BitReader<'a>,
    req_bytes: usize,
    resi_bits: u32,
    prev: [u8; 8],
}

impl<'a> Exact<'a> {
    fn new(t: &Tdps<'a>) -> Result<Self, Error> {
        // Two bits per value, most significant first.
        let mut lead = Vec::with_capacity(t.exact_count.min(t.lead.len().saturating_mul(4)));
        'outer: for b in t.lead {
            for shift in [6u32, 4, 2, 0] {
                if lead.len() == t.exact_count {
                    break 'outer;
                }
                lead.push((b >> shift) & 3);
            }
        }
        if lead.len() < t.exact_count {
            return Err(err());
        }
        Ok(Self {
            lead,
            next_lead: 0,
            mid: t.exact_mid,
            mid_at: 0,
            residual: BitReader::new(t.residual, CODEC),
            req_bytes: usize::from(t.req_length / 8),
            resi_bits: u32::from(t.req_length % 8),
            prev: [0; 8],
        })
    }

    fn next<T: SzReal>(&mut self) -> Result<T, Error> {
        let mut cur = [0u8; 8];
        let lead = usize::from(*self.lead.get(self.next_lead).ok_or_else(err)?);
        self.next_lead = self.next_lead.saturating_add(1);
        cur.get_mut(..lead)
            .ok_or_else(err)?
            .copy_from_slice(self.prev.get(..lead).ok_or_else(err)?);
        for j in lead..self.req_bytes {
            *cur.get_mut(j).ok_or_else(err)? =
                self.mid.get(self.mid_at).copied().ok_or_else(err)?;
            self.mid_at = self.mid_at.saturating_add(1);
        }
        if self.resi_bits != 0 {
            let bits = self.residual.bits(self.resi_bits)?;
            let byte = u8::try_from(bits << (8u32.saturating_sub(self.resi_bits))).unwrap_or(0);
            *cur.get_mut(self.req_bytes).ok_or_else(err)? = byte;
        }
        // Bytes past the type's size stay zero.
        self.prev = cur;
        T::from_be(cur.get(..T::SIZE).ok_or_else(err)?).ok_or_else(err)
    }
}

/// The quantization indexes of the values (the type array: its tree, then the code).
fn types(type_array: &[u8], n: usize) -> Result<Vec<i32>, Error> {
    let mut rd = Rd {
        b: type_array,
        at: 0,
    };
    let tree = rd.tree()?;
    tree.decode_bits(type_array.get(rd.at..).ok_or_else(err)?, n)
}

fn idx(t: i32) -> Result<usize, Error> {
    usize::try_from(t).map_err(|_| err())
}

fn get<T: Copy>(v: &[T], i: usize) -> Result<T, Error> {
    v.get(i).copied().ok_or_else(err)
}

fn set<T>(v: &mut [T], i: usize, x: T) -> Result<(), Error> {
    *v.get_mut(i).ok_or_else(err)? = x;
    Ok(())
}

/// 1-D data, absolute bounds: each value its predecessor plus a multiple of twice the bound.
fn one_d<T: SzReal>(t: &Tdps<'_>, n: usize) -> Result<Vec<T>, Error> {
    let radius = i32::try_from(t.intervals.checked_div(2).unwrap_or(0)).map_err(|_| err())?;
    let interval = T::from_f64(t.real_precision * 2.0);
    let median = T::from_f64(t.median);
    let ty = types(t.type_array, n)?;
    let mut exact = Exact::new(t)?;
    let mut out: Vec<T> = Vec::with_capacity(n);
    for (i, q) in ty.iter().enumerate() {
        let v = if *q == 0 {
            let x = exact.next::<T>()?;
            #[allow(clippy::arithmetic_side_effects)] // floating point
            let v = x + median;
            v
        } else {
            let prev = get(&out, i.checked_sub(1).ok_or_else(err)?)?;
            T::from_f64(f64::from(q.wrapping_sub(radius))).fma(interval, prev)
        };
        out.push(v);
    }
    Ok(out)
}

/// The accelerated point-wise relative scheme (MSST'19), 1 to 3 dimensions (`dims`, slowest
/// first): multiplicative Lorenzo predictions, scaled by a table of powers of the bound.
fn msst19<T: SzReal>(t: &Tdps<'_>, dims: &[usize]) -> Result<Vec<T>, Error> {
    let n = dims
        .iter()
        .try_fold(1usize, |a, d| a.checked_mul(*d))
        .ok_or_else(err)?;
    let capacity = t.intervals;
    let radius = i32::try_from(capacity.checked_div(2).unwrap_or(0)).map_err(|_| err())?;
    let inv = 2.0 - 2f64.powi(0i32.saturating_sub(i32::from(t.plus_bits)));
    let table: Vec<f64> = (0..capacity)
        .map(|i| {
            let k = i32::try_from(i).unwrap_or(i32::MAX).wrapping_sub(radius);
            (1.0 + t.real_precision).powf(inv * f64::from(k))
        })
        .collect();
    let ty = types(t.type_array, n)?;
    let mut exact = Exact::new(t)?;
    let mut d: Vec<T> = vec![T::default(); n];
    let f = |x: T| x.to_f64();
    // `pred` as the reference computes it for each position (`None`: no prediction).
    let mut put = |d: &mut Vec<T>, i: usize, pred: Option<T>| -> Result<(), Error> {
        let q = get(&ty, i)?;
        let v = match pred {
            Some(p) if q != 0 => T::from_f64(f(p).abs() * get(&table, idx(q)?)?),
            _ => exact.next::<T>()?,
        };
        set(d, i, v)
    };
    // The 2-D float decoder predicts in `float` arithmetic; the 3-D one through `double`.
    let wide = dims.len() == 3 || T::SIZE == 8;
    #[allow(clippy::arithmetic_side_effects)] // floating point
    let ratio = |a: T, b: T, c: T| -> T {
        if wide {
            T::from_f64(f(a) * f(b) / f(c))
        } else {
            a * b / c
        }
    };
    // The value `back` places before `k`.
    let at = |d: &[T], k: usize, back: usize| -> Result<T, Error> {
        get(d, k.checked_sub(back).ok_or_else(err)?)
    };
    let sum = |parts: &[usize]| parts.iter().fold(0usize, |a, b| a.saturating_add(*b));
    match dims {
        [len] => {
            let mut pred = T::default();
            for i in 0..*len {
                let q = get(&ty, i)?;
                let v = if q == 0 {
                    exact.next::<T>()?
                } else {
                    T::from_f64(f(pred).abs() * get(&table, idx(q)?)?)
                };
                set(&mut d, i, v)?;
                pred = v;
            }
        }
        [r1, r2] => {
            let (r1, r2) = (*r1, *r2);
            put(&mut d, 0, None)?;
            if r2 > 1 {
                let p = get(&d, 0)?;
                put(&mut d, 1, Some(p))?;
            }
            for j in 2..r2 {
                let (a, b) = (at(&d, j, 1)?, at(&d, j, 2)?);
                put(&mut d, j, Some(ratio(a, a, b)))?;
            }
            for i in 1..r1 {
                let row = i.saturating_mul(r2);
                let p = at(&d, row, r2)?;
                put(&mut d, row, Some(p))?;
                for j in 1..r2 {
                    let k = row.saturating_add(j);
                    let p = ratio(at(&d, k, 1)?, at(&d, k, r2)?, at(&d, k, sum(&[r2, 1]))?);
                    put(&mut d, k, Some(p))?;
                }
            }
        }
        [r1, r2, r3] => {
            let (r1, r2, r3) = (*r1, *r2, *r3);
            let r23 = r2.checked_mul(r3).ok_or_else(err)?;
            // The first plane: as 2-D (through `double`).
            put(&mut d, 0, None)?;
            if r3 > 1 {
                let p = get(&d, 0)?;
                put(&mut d, 1, Some(p))?;
            }
            for j in 2..r3 {
                let (a, b) = (at(&d, j, 1)?, at(&d, j, 2)?);
                put(&mut d, j, Some(ratio(a, a, b)))?;
            }
            for i in 1..r2 {
                let row = i.saturating_mul(r3);
                let p = at(&d, row, r3)?;
                put(&mut d, row, Some(p))?;
                for j in 1..r3 {
                    let k = row.saturating_add(j);
                    let p = ratio(at(&d, k, 1)?, at(&d, k, r3)?, at(&d, k, sum(&[r3, 1]))?);
                    put(&mut d, k, Some(p))?;
                }
            }
            for kk in 1..r1 {
                let plane = kk.saturating_mul(r23);
                let p = at(&d, plane, r23)?;
                put(&mut d, plane, Some(p))?;
                for j in 1..r3 {
                    let k = plane.saturating_add(j);
                    let p = ratio(at(&d, k, 1)?, at(&d, k, r23)?, at(&d, k, sum(&[r23, 1]))?);
                    put(&mut d, k, Some(p))?;
                }
                for i in 1..r2 {
                    let row = plane.saturating_add(i.saturating_mul(r3));
                    let p = ratio(
                        at(&d, row, r3)?,
                        at(&d, row, r23)?,
                        at(&d, row, sum(&[r23, r3]))?,
                    );
                    put(&mut d, row, Some(p))?;
                    for j in 1..r3 {
                        let k = row.saturating_add(j);
                        let num = f(at(&d, k, 1)?)
                            * f(at(&d, k, r3)?)
                            * f(at(&d, k, r23)?)
                            * f(at(&d, k, sum(&[r23, r3, 1]))?);
                        let den = f(at(&d, k, sum(&[r3, 1]))?)
                            * f(at(&d, k, sum(&[r23, r3]))?)
                            * f(at(&d, k, sum(&[r23, 1]))?);
                        put(&mut d, k, Some(T::from_f64(num / den)))?;
                    }
                }
            }
        }
        _ => return Err(err()),
    }
    // Values below the smallest logarithm were zeros; signs come from their own stream.
    let threshold = T::from_f64(t.min_log);
    let zero = T::default();
    let below = |x: T| x.to_f64() < threshold.to_f64();
    if t.pwr_bytes.is_empty() {
        for v in &mut d {
            if below(*v) {
                *v = zero;
            }
        }
    } else {
        let signs = crate::zstd::decompress(t.pwr_bytes, n)?;
        for (i, v) in d.iter_mut().enumerate() {
            if below(*v) && v.to_f64() >= 0.0 {
                *v = zero;
                continue;
            }
            if signs.get(i).copied().unwrap_or(0) != 0 {
                *v = v.negative();
            }
        }
    }
    Ok(d)
}

/// A Huffman tree stored with its size: the size, the node count, the tree.
fn sized_tree(rd: &mut Rd<'_>) -> Result<Huffman, Error> {
    let size = usize::try_from(rd.be_u32()?).map_err(|_| err())?;
    let count = usize::try_from(rd.be_u32()?).map_err(|_| err())?;
    let rest = rd.b.get(rd.at..).ok_or_else(err)?;
    let swap = rest.first().copied() == Some(1);
    let (tree, _) = Huffman::tree(rest, count, swap)?;
    rd.bytes(size)?;
    Ok(tree)
}

/// SZ 2's block partition of `len`: as many blocks as fit whole (at least one), the first
/// `len % blocks` of them one longer.
fn blocks(len: usize, size: usize) -> Vec<(usize, usize)> {
    let num = if len <= size {
        1
    } else {
        len.checked_div(size).unwrap_or(1)
    };
    let late = len.checked_div(num).unwrap_or(0);
    let split = len.checked_rem(num).unwrap_or(0);
    let early = if split != 0 {
        late.saturating_add(1)
    } else {
        late
    };
    (0..num)
        .map(|i| {
            if i < split {
                (i.saturating_mul(early), early)
            } else {
                (i.saturating_mul(late).saturating_add(split), late)
            }
        })
        .collect()
}

/// Blocked Lorenzo/regression prediction (2-D, or 3-D with 4-D data's leading dimensions
/// merged): `dims` slowest first.
fn blocked<T: SzReal>(ra: &[u8], dims: &[usize], wide: bool) -> Result<Vec<T>, Error> {
    let nd = dims.len();
    if nd != 2 && nd != 3 {
        return Err(err());
    }
    let mut rd = Rd { b: ra, at: 0 };
    let block = usize::try_from(rd.be_u32()?).map_err(|_| err())?;
    if block == 0 {
        return Err(err());
    }
    let parts: Vec<Vec<(usize, usize)>> = dims.iter().map(|d| blocks(*d, block)).collect();
    let nblocks = parts
        .iter()
        .try_fold(1usize, |a, p| a.checked_mul(p.len()))
        .ok_or_else(err)?;
    let precision: T = if T::SIZE == 4 {
        T::from_f64(f64::from(rd.be_f32()?))
    } else {
        T::from_f64(rd.be_f64()?)
    };
    let intervals = rd.be_u32()?;
    let radius = i32::try_from(intervals / 2).map_err(|_| err())?;
    let tree = sized_tree(&mut rd)?;
    let use_mean = rd.u8()? != 0;
    let mean: T = T::from_le(rd.bytes(T::SIZE)?).ok_or_else(err)?;
    // One bit per block: 1 Lorenzo, 0 regression.
    let indicator_len = nblocks
        .saturating_sub(1)
        .checked_div(8)
        .unwrap_or(0)
        .saturating_add(1);
    let ind_bytes = rd.bytes(indicator_len)?;
    let indicator: Vec<bool> = (0..nblocks)
        .map(|i| {
            ind_bytes.get(i / 8).is_some_and(|b| {
                let shift = 7u32.saturating_sub(u32::try_from(i % 8).unwrap_or(0));
                b.checked_shr(shift).unwrap_or(0) & 1 == 1
            })
        })
        .collect();
    let reg_count = indicator.iter().filter(|x| !**x).count();
    // Regression coefficients: per coefficient, its precision, radius, codes, unpredictables.
    let ncoef = nd.saturating_add(1);
    let mut coef_types: Vec<Vec<i32>> = Vec::new();
    let mut coef_prec: Vec<T> = Vec::new();
    let mut coef_radius: Vec<i32> = Vec::new();
    let mut coef_unpred: Vec<Vec<T>> = Vec::new();
    if reg_count > 0 {
        for _ in 0..ncoef {
            coef_prec.push(if T::SIZE == 4 {
                T::from_f64(f64::from(rd.be_f32()?))
            } else {
                T::from_f64(rd.be_f64()?)
            });
            coef_radius.push(i32::try_from(rd.be_u32()?).map_err(|_| err())?);
            let tree = sized_tree(&mut rd)?;
            // The code's length: SZ's size type, in a `size_t`'s 8 bytes.
            let slot = rd.bytes(8)?;
            let len = if wide {
                u64::from_be_bytes(slot.try_into().map_err(|_| err())?)
            } else {
                u64::from(u32::from_be_bytes(
                    slot.get(..4)
                        .ok_or_else(err)?
                        .try_into()
                        .map_err(|_| err())?,
                ))
            };
            let code = rd.bytes(usize::try_from(len).map_err(|_| err())?)?;
            coef_types.push(tree.decode_bits(code, reg_count)?);
            let n = usize::try_from(rd.be_u32()?).map_err(|_| err())?;
            coef_unpred.push(rd.ne_reals(n)?);
        }
    }
    let total_unpred = rd.ne_size()?;
    let unpred: Vec<T> = rd.ne_reals(total_unpred)?;
    let n = dims
        .iter()
        .try_fold(1usize, |a, d| a.checked_mul(*d))
        .ok_or_else(err)?;
    let ty = tree.decode_bits(ra.get(rd.at..).ok_or_else(err)?, n)?;
    let mut out = vec![T::default(); n];
    let strides: Vec<usize> = match dims {
        [_, b] => vec![*b, 1],
        [_, b, c] => vec![b.saturating_mul(*c), *c, 1],
        _ => return Err(err()),
    };
    let mut q = ty.iter();
    let mut unpred = unpred.iter();
    let mut coef = vec![T::default(); ncoef];
    let mut coef_at = 0usize;
    let mut coef_unpred_at = vec![0usize; ncoef];
    let step = |q: i32, radius: i32| T::from_f64(f64::from(q.wrapping_sub(radius).wrapping_mul(2)));
    let two = |v: &[usize], i: usize| v.get(i).copied().unwrap_or(0);
    for b in 0..nblocks {
        // The block's position along each dimension (the last fastest).
        let mut rem = b;
        let mut place = vec![(0usize, 0usize); nd];
        for k in (0..nd).rev() {
            let p = parts.get(k).ok_or_else(err)?;
            let i = rem.checked_rem(p.len()).unwrap_or(0);
            rem = rem.checked_div(p.len()).unwrap_or(0);
            *place.get_mut(k).ok_or_else(err)? = *p.get(i).ok_or_else(err)?;
        }
        let extent: Vec<usize> = place.iter().map(|p| p.1).collect();
        let lorenzo = *indicator.get(b).ok_or_else(err)?;
        if !lorenzo {
            for (e, c) in coef.iter_mut().enumerate() {
                let tq = get(coef_types.get(e).ok_or_else(err)?, coef_at)?;
                *c = if tq != 0 {
                    let r = get(&coef_radius, e)?;
                    step(tq, r).fma(get(&coef_prec, e)?, *c)
                } else {
                    let at = get(&coef_unpred_at, e)?;
                    let v = get(coef_unpred.get(e).ok_or_else(err)?, at)?;
                    set(&mut coef_unpred_at, e, at.saturating_add(1))?;
                    v
                };
            }
            coef_at = coef_at.saturating_add(1);
        }
        let count: usize = place.iter().map(|p| p.1).product();
        let mut local = vec![0usize; nd];
        for _ in 0..count {
            let pos: Vec<usize> = place
                .iter()
                .zip(&local)
                .map(|(p, l)| p.0.saturating_add(*l))
                .collect();
            let flat = pos
                .iter()
                .zip(&strides)
                .fold(0usize, |a, (p, s)| a.saturating_add(p.saturating_mul(*s)));
            let tq = *q.next().ok_or_else(err)?;
            let v = if lorenzo {
                if use_mean && tq == radius {
                    mean
                } else if tq == 0 {
                    *unpred.next().ok_or_else(err)?
                } else {
                    let tq = if use_mean && tq < radius {
                        tq.saturating_add(1)
                    } else {
                        tq
                    };
                    // Zero outside the array, terms in the reference's order.
                    let at = |back: &[usize]| -> T {
                        let mut i = flat;
                        for (k, o) in back.iter().enumerate() {
                            if *o == 0 {
                                continue;
                            }
                            if two(&pos, k) == 0 {
                                return T::default();
                            }
                            i = i.saturating_sub(two(&strides, k));
                        }
                        out.get(i).copied().unwrap_or_default()
                    };
                    #[allow(clippy::arithmetic_side_effects)] // floating point
                    let pred = if nd == 2 {
                        at(&[0, 1]) + at(&[1, 0]) - at(&[1, 1])
                    } else {
                        at(&[0, 0, 1]) + at(&[0, 1, 0]) + at(&[1, 0, 0])
                            - at(&[0, 1, 1])
                            - at(&[1, 0, 1])
                            - at(&[1, 1, 0])
                            + at(&[1, 1, 1])
                    };
                    step(tq, radius).fma(precision, pred)
                }
            } else if tq != 0 {
                #[allow(clippy::cast_precision_loss)]
                let l = |k: usize| T::from_f64(two(&local, k) as f64);
                let (c0, c1, c2) = (get(&coef, 0)?, get(&coef, 1)?, get(&coef, 2)?);
                #[allow(clippy::arithmetic_side_effects)] // floating point
                let pred = if nd == 2 {
                    c0.fma(l(0), c1 * l(1)) + c2
                } else {
                    c2.fma(l(2), c0.fma(l(0), c1 * l(1))) + coef.get(3).copied().unwrap_or_default()
                };
                step(tq, radius).fma(precision, pred)
            } else {
                *unpred.next().ok_or_else(err)?
            };
            set(&mut out, flat, v)?;
            for k in (0..nd).rev() {
                let e = two(&extent, k);
                if let Some(l) = local.get_mut(k) {
                    *l = l.saturating_add(1);
                    if *l < e {
                        break;
                    }
                    *l = 0;
                }
            }
        }
    }
    Ok(out)
}

/// Decode an SZ 2 buffer of `T`s with extents `dims` (slowest first) into native-endian
/// bytes, at most `limit` bytes.
///
/// # Errors
/// [`Error::Corrupt`] for malformed buffers; [`Error::Unsupported`] for the features listed
/// in the module documentation.
pub fn decompress<T: SzReal>(data: &[u8], dims: &[usize], limit: usize) -> Result<Vec<u8>, Error> {
    let n = dims
        .iter()
        .try_fold(1usize, |a, d| a.checked_mul(*d))
        .ok_or_else(err)?;
    if n.checked_mul(T::SIZE).is_none_or(|b| b > limit) {
        return Err(Error::TooLarge(CODEC.into()));
    }
    // The whole buffer is compressed losslessly, unless it was smaller as it is.
    let room = limit.saturating_mul(2).saturating_add(1 << 16);
    let flat: Vec<u8> = if data.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        crate::zstd::decompress(data, room)?
    } else if matches!(
        data,
        [104, 5 | 129 | 222, ..] | [120, 1 | 94 | 156 | 218, ..]
    ) {
        crate::zlib::zlib_decompress(data, room)?
    } else {
        data.to_vec()
    };
    let mut rd = Rd { b: &flat, at: 0 };
    rd.bytes(3)?; // version
    let flags = rd.u8()?;
    let meta = rd.bytes(T::META)?;
    if meta.get(14).copied() != Some(101) {
        return Err(Error::Unsupported(
            "SZ data transposed (SZ_Transpose)".into(),
        ));
    }
    let wide = (flags >> 6) & 1 == 1;
    let count = rd.be_size(wide)?;
    if count != n {
        return Err(err());
    }
    let values: Vec<T> = if (flags >> 4) & 1 == 1 {
        // Stored as they are (big-endian).
        (0..n)
            .map(|_| rd.be_real::<T>())
            .collect::<Result<_, _>>()?
    } else if flags & 1 == 1 {
        vec![rd.be_real::<T>()?; n]
    } else if flags >> 7 == 1 {
        let ra = flat.get(rd.at..).ok_or_else(err)?;
        match dims {
            [a, b] => blocked(ra, &[*a, *b], wide)?,
            [a, b, c] => blocked(ra, &[*a, *b, *c], wide)?,
            [a, b, c, d] => blocked(ra, &[a.checked_mul(*b).ok_or_else(err)?, *c, *d], wide)?,
            _ => return Err(err()),
        }
    } else {
        let pw_rel = (flags >> 5) & 1 == 1;
        let accelerate = (flags >> 3) & 1 == 1;
        rd.be_u32()?; // quantization intervals (as the maximum)
        let mut pwr_size = 0usize;
        if pw_rel {
            rd.u8()?; // radius exponent
            rd.be_size(wide)?; // segment size
            pwr_size = usize::try_from(rd.be_u32()?).map_err(|_| err())?;
        }
        let intervals = usize::try_from(rd.be_u32()?).map_err(|_| err())?;
        let median = if T::SIZE == 4 {
            f64::from(rd.be_f32()?)
        } else {
            rd.be_f64()?
        };
        let req_length = rd.u8()?;
        let (plus_bits, _max_bits) = if pw_rel && accelerate {
            (rd.u8()?, rd.u8()?)
        } else {
            (0, 0)
        };
        let real_precision = rd.be_f64()?;
        let type_size = rd.be_size(wide)?;
        let exact_count = rd.be_size(wide)?;
        let mid_size = rd.be_size(wide)?;
        let min_log = if pw_rel {
            if T::SIZE == 4 {
                f64::from(rd.be_f32()?)
            } else {
                rd.be_f64()?
            }
        } else {
            0.0
        };
        let type_array = rd.bytes(type_size)?;
        let pwr_bytes = rd.bytes(pwr_size)?;
        let lead = rd.bytes(exact_count.saturating_mul(2).div_ceil(8))?;
        let exact_mid = rd.bytes(mid_size)?;
        let residual = flat.get(rd.at..).ok_or_else(err)?;
        let t = Tdps {
            pw_rel,
            accelerate,
            intervals,
            median,
            req_length,
            plus_bits,
            real_precision,
            type_array,
            exact_count,
            pwr_bytes,
            lead,
            exact_mid,
            residual,
            min_log,
        };
        if t.pw_rel {
            if !t.accelerate {
                return Err(Error::Unsupported(
                    "SZ point-wise relative bounds without acceleration".into(),
                ));
            }
            match dims {
                [_] | [_, _] | [_, _, _] => msst19(&t, dims)?,
                [a, b, c, d] => msst19(&t, &[a.checked_mul(*b).ok_or_else(err)?, *c, *d])?,
                _ => return Err(err()),
            }
        } else if dims.len() == 1 {
            one_d(&t, n)?
        } else {
            return Err(Error::Unsupported(
                "SZ blocked Lorenzo prediction without regression".into(),
            ));
        }
    };
    Ok(values.into_iter().flat_map(Real::to_ne).collect())
}
