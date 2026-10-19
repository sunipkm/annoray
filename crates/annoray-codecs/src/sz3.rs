//! SZ3 (version 3.1, as the HDF5 SZ3 filter bundled with hdf5plugin writes it) decoding of
//! `float` and `double` data, bit for bit as the reference decodes.
//!
//! A compressed buffer is a Zstandard frame (preceded by its decoded length) holding the
//! predictor's state, a quantizer (error bound, radius and the values it could not predict)
//! and Huffman-coded quantization indexes, followed by the configuration it was made with and
//! that configuration's length. Two algorithms decode: interpolation (linear or cubic, level by
//! level from coarse to fine) and Lorenzo/regression prediction block by block (first- and
//! second-order Lorenzo, linear regression, or a per-block choice among them; 3-D data through
//! SZ3's "fast" frontend). With an error bound of 0 the data are stored losslessly.
//!
//! Not supported: polynomial (second-order) regression, integer data, and the arithmetic coder.
//! Arithmetic follows the reference's types (`float` stays `float`, constants promote to
//! `double` where the C++ does), so the values are those HDF5 reads back.

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::Error;
use crate::bits::corrupt;

const CODEC: &str = "sz3";

fn err() -> Error {
    corrupt(CODEC)
}

/// The floating-point types SZ3 data can be.
pub trait Real:
    Copy
    + Default
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Bytes per value.
    const SIZE: usize;
    /// `value as Self` (rounded to nearest).
    fn from_f64(value: f64) -> Self;
    /// The value as `f64` (exact).
    fn to_f64(self) -> f64;
    /// A small integer constant (exact).
    fn small(value: i32) -> Self;
    /// From little-endian bytes (`SIZE` of them).
    fn from_le(bytes: &[u8]) -> Option<Self>;
    /// Native-endian bytes.
    fn to_ne(self) -> Vec<u8>;
    /// `self * a + b`, rounded once.
    fn fma(self, a: Self, b: Self) -> Self;
}

impl Real for f32 {
    const SIZE: usize = 4;
    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[allow(clippy::cast_precision_loss)]
    fn small(value: i32) -> Self {
        value as f32
    }
    fn from_le(bytes: &[u8]) -> Option<Self> {
        Some(Self::from_le_bytes(bytes.try_into().ok()?))
    }
    fn to_ne(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }
    fn fma(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
}

impl Real for f64 {
    const SIZE: usize = 8;
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn small(value: i32) -> Self {
        f64::from(value)
    }
    fn from_le(bytes: &[u8]) -> Option<Self> {
        Some(Self::from_le_bytes(bytes.try_into().ok()?))
    }
    fn to_ne(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }
    fn fma(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
}

/// Little-endian fields from a buffer.
struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Rd<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, at: 0 }
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).ok_or_else(err)?;
        let s = self.b.get(self.at..end).ok_or_else(err)?;
        self.at = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        self.bytes(1)?.first().copied().ok_or_else(err)
    }

    fn i32(&mut self) -> Result<i32, Error> {
        Ok(i32::from_le_bytes(
            self.bytes(4)?.try_into().map_err(|_| err())?,
        ))
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().map_err(|_| err())?,
        ))
    }

    fn usize(&mut self) -> Result<usize, Error> {
        let v = u64::from_le_bytes(self.bytes(8)?.try_into().map_err(|_| err())?);
        usize::try_from(v).map_err(|_| err())
    }

    fn f64(&mut self) -> Result<f64, Error> {
        Ok(f64::from_le_bytes(
            self.bytes(8)?.try_into().map_err(|_| err())?,
        ))
    }

    fn f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_le_bytes(
            self.bytes(4)?.try_into().map_err(|_| err())?,
        ))
    }

    fn real<T: Real>(&mut self) -> Result<T, Error> {
        T::from_le(self.bytes(T::SIZE)?).ok_or_else(err)
    }

    /// `n` values, after checking the buffer holds them (so a corrupt count cannot allocate).
    fn reals<T: Real>(&mut self, n: usize) -> Result<Vec<T>, Error> {
        let bytes = self.bytes(n.checked_mul(T::SIZE).ok_or_else(err)?)?;
        bytes
            .chunks_exact(T::SIZE)
            .map(|c| T::from_le(c).ok_or_else(err))
            .collect()
    }
}

// ----- Huffman -----

#[derive(Debug, Clone, Copy)]
struct Node {
    left: Option<usize>,
    right: Option<usize>,
    c: i32,
    leaf: bool,
}

/// SZ's Huffman coder of `int`s (SZ 2 and SZ3 share it): a tree stored as child-index
/// arrays, codes most significant bit first.
pub(crate) struct Huffman {
    nodes: Vec<Node>,
    offset: i32,
}

impl Huffman {
    /// A tree of `count` nodes from `bytes` (its endianness byte, then the child, symbol and
    /// leaf arrays); the bytes it took. `swap`: the writer's integers are big-endian (SZ 2
    /// records the writer's byte order; SZ3 assumes the reader's).
    pub(crate) fn tree(bytes: &[u8], count: usize, swap: bool) -> Result<(Self, usize), Error> {
        if count == 0 {
            return Err(err());
        }
        // Child indexes are 1, 2 or 4 bytes wide, by the tree's size.
        let width: usize = if count <= 256 {
            1
        } else if count <= 65536 {
            2
        } else {
            4
        };
        let per_node = width
            .checked_mul(2)
            .and_then(|w| w.checked_add(4 + 1))
            .ok_or_else(err)?;
        let len = count
            .checked_mul(per_node)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(err)?;
        let arrays = bytes.get(1..len).ok_or_else(err)?;
        let int = |s: &[u8]| -> Result<u32, Error> {
            let mut v = [0u8; 4];
            if swap {
                let at = 4usize.saturating_sub(s.len());
                v.get_mut(at..).ok_or_else(err)?.copy_from_slice(s);
                Ok(u32::from_be_bytes(v))
            } else {
                v.get_mut(..s.len()).ok_or_else(err)?.copy_from_slice(s);
                Ok(u32::from_le_bytes(v))
            }
        };
        let index = |array: usize, i: usize| -> Result<usize, Error> {
            let at = array
                .checked_mul(count)
                .and_then(|a| a.checked_add(i))
                .and_then(|a| a.checked_mul(width))
                .ok_or_else(err)?;
            let s = arrays.get(at..at.saturating_add(width)).ok_or_else(err)?;
            usize::try_from(int(s)?).map_err(|_| err())
        };
        let lr_end = count.checked_mul(width.saturating_mul(2)).ok_or_else(err)?;
        let c_of = |i: usize| -> Result<i32, Error> {
            let at = lr_end
                .checked_add(i.checked_mul(4).ok_or_else(err)?)
                .ok_or_else(err)?;
            let s = arrays.get(at..at.saturating_add(4)).ok_or_else(err)?;
            Ok(i32::from_ne_bytes(int(s)?.to_ne_bytes()))
        };
        let t_of = |i: usize| -> Result<bool, Error> {
            let at = lr_end
                .checked_add(count.checked_mul(4).ok_or_else(err)?)
                .and_then(|a| a.checked_add(i))
                .ok_or_else(err)?;
            Ok(arrays.get(at).copied().ok_or_else(err)? != 0)
        };
        // The root: its symbol and leaf flag only for small trees (as the reference).
        let root = if count <= 256 {
            Node {
                left: None,
                right: None,
                c: c_of(0)?,
                leaf: t_of(0)?,
            }
        } else {
            Node {
                left: None,
                right: None,
                c: 0,
                leaf: false,
            }
        };
        let mut nodes = vec![root];
        // Depth-first, as `unpad_tree`: (tree node, array index).
        let mut stack = vec![(0usize, 0usize)];
        while let Some((n, i)) = stack.pop() {
            if nodes.get(n).is_none_or(|x| x.leaf) {
                continue;
            }
            if nodes.len() > count.saturating_mul(2) {
                return Err(err());
            }
            for side in [1usize, 0] {
                let child = index(side, i)?;
                if child == 0 {
                    continue;
                }
                nodes.push(Node {
                    left: None,
                    right: None,
                    c: c_of(child)?,
                    leaf: t_of(child)?,
                });
                let k = nodes.len().saturating_sub(1);
                if let Some(parent) = nodes.get_mut(n) {
                    if side == 0 {
                        parent.left = Some(k);
                    } else {
                        parent.right = Some(k);
                    }
                }
                stack.push((k, child));
            }
        }
        Ok((Self { nodes, offset: 0 }, len))
    }

    /// `n` symbols from the code bits (most significant first).
    pub(crate) fn decode_bits(&self, bits: &[u8], n: usize) -> Result<Vec<i32>, Error> {
        let root = self.nodes.first().ok_or_else(err)?;
        if root.leaf {
            return Ok(vec![root.c.wrapping_add(self.offset); n]);
        }
        let mut out = Vec::with_capacity(n.min(bits.len().saturating_mul(8)));
        let mut node = 0usize;
        let mut i = 0usize;
        while out.len() < n {
            let byte = bits.get(i / 8).copied().ok_or_else(err)?;
            let bit = byte
                .checked_shr(7u32.saturating_sub(u32::try_from(i % 8).unwrap_or(0)))
                .unwrap_or(0)
                & 1;
            i = i.saturating_add(1);
            let cur = self.nodes.get(node).ok_or_else(err)?;
            node = if bit == 0 { cur.left } else { cur.right }.ok_or_else(err)?;
            let next = self.nodes.get(node).ok_or_else(err)?;
            if next.leaf {
                out.push(next.c.wrapping_add(self.offset));
                node = 0;
            }
        }
        Ok(out)
    }

    /// SZ3's form: the symbol offset, the node count (big-endian) and another word, the tree.
    fn load(rd: &mut Rd<'_>) -> Result<Self, Error> {
        let offset = rd.i32()?;
        let head = rd.bytes(8)?;
        let count = usize::try_from(u32::from_be_bytes(
            head.get(..4)
                .ok_or_else(err)?
                .try_into()
                .map_err(|_| err())?,
        ))
        .map_err(|_| err())?;
        let rest = rd.b.get(rd.at..).ok_or_else(err)?;
        let (mut tree, used) = Self::tree(rest, count, false)?;
        rd.bytes(used)?;
        tree.offset = offset;
        Ok(tree)
    }

    /// SZ3's form: the code's length in bytes, then the code.
    fn decode(&self, rd: &mut Rd<'_>, n: usize) -> Result<Vec<i32>, Error> {
        let encoded = rd.usize()?;
        let root = self.nodes.first().ok_or_else(err)?;
        if root.leaf {
            return Ok(vec![root.c.wrapping_add(self.offset); n]);
        }
        let bits = rd.bytes(encoded)?;
        self.decode_bits(bits, n)
    }
}

// ----- quantizer -----

/// SZ3's linear quantizer: indexes around `radius` in steps of twice the error bound, 0 for
/// values stored as they are.
struct Quant<T> {
    eb: f64,
    radius: i32,
    unpred: Vec<T>,
    next: usize,
}

impl<T: Real> Quant<T> {
    fn load(rd: &mut Rd<'_>) -> Result<Self, Error> {
        rd.u8()?;
        let eb = rd.f64()?;
        let radius = rd.i32()?;
        let n = rd.usize()?;
        Ok(Self {
            eb,
            radius,
            unpred: rd.reals(n)?,
            next: 0,
        })
    }

    fn recover_pred(&self, pred: T, q: i32) -> T {
        let steps = q.wrapping_sub(self.radius).wrapping_mul(2);
        // `pred + 2 * (q - radius) * eb`, with the multiply-add fused as the plugin's build
        // compiles it.
        T::from_f64(f64::from(steps).mul_add(self.eb, pred.to_f64()))
    }

    fn recover_unpred(&mut self) -> Result<T, Error> {
        let v = self.unpred.get(self.next).copied().ok_or_else(err)?;
        self.next = self.next.saturating_add(1);
        Ok(v)
    }

    fn recover(&mut self, pred: T, q: i32) -> Result<T, Error> {
        if q == 0 {
            self.recover_unpred()
        } else {
            Ok(self.recover_pred(pred, q))
        }
    }
}

// ----- configuration -----

struct Config {
    dims: Vec<usize>,
    num: usize,
    algo: u8,
    abs: f64,
    lorenzo: bool,
    lorenzo2: bool,
    regression: bool,
    regression2: bool,
}

fn config(data: &[u8]) -> Result<(Config, usize), Error> {
    let n = data.len();
    let size_at = n.checked_sub(4).ok_or_else(err)?;
    let size = usize::try_from(i32::from_le_bytes(
        data.get(size_at..)
            .ok_or_else(err)?
            .try_into()
            .map_err(|_| err())?,
    ))
    .map_err(|_| err())?;
    let start = size_at.checked_sub(size).ok_or_else(err)?;
    let mut rd = Rd::new(data.get(start..size_at).ok_or_else(err)?);
    let ndims = usize::from(rd.u8()?);
    if ndims == 0 || ndims > 4 {
        return Err(Error::Unsupported(format!(
            "SZ3 data of {ndims} dimensions"
        )));
    }
    let dims = (0..ndims)
        .map(|_| rd.usize())
        .collect::<Result<Vec<_>, _>>()?;
    let num = rd.usize()?;
    let algo = rd.u8()?;
    rd.u8()?; // error bound mode
    let abs = rd.f64()?;
    rd.f64()?; // relative error bound
    let lorenzo = rd.u8()? != 0;
    let lorenzo2 = rd.u8()? != 0;
    let regression = rd.u8()? != 0;
    let regression2 = rd.u8()? != 0;
    Ok((
        Config {
            dims,
            num,
            algo,
            abs,
            lorenzo,
            lorenzo2,
            regression,
            regression2,
        },
        start,
    ))
}

/// SZ3's lossless stage: the decoded length, then a Zstandard frame.
fn unzstd(data: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    let mut rd = Rd::new(data);
    let len = rd.usize()?;
    if len > limit {
        return Err(Error::TooLarge(CODEC.into()));
    }
    let out = crate::zstd::decompress(data.get(8..).ok_or_else(err)?, len)?;
    if out.len() != len {
        return Err(err());
    }
    Ok(out)
}

// ----- helpers over C-order arrays -----

fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        let next = s.get(k.saturating_add(1)).copied().unwrap_or(1);
        let d = dims.get(k.saturating_add(1)).copied().unwrap_or(1);
        if let Some(slot) = s.get_mut(k) {
            *slot = next.saturating_mul(d);
        }
    }
    s
}

fn at<T: Real>(data: &[T], i: usize) -> Result<T, Error> {
    data.get(i).copied().ok_or_else(err)
}

fn put<T: Real>(data: &mut [T], i: usize, v: T) -> Result<(), Error> {
    *data.get_mut(i).ok_or_else(err)? = v;
    Ok(())
}

/// Evaluate a Lorenzo formula: terms `(coefficient, offsets back per dimension)`, summed left
/// to right as the reference's expression (a coefficient of ±1 is no multiplication).
#[allow(clippy::arithmetic_side_effects)] // floating point (f32/f64) only
fn lorenzo_eval<T: Real>(terms: &[(i32, &[usize])], value: impl Fn(&[usize]) -> T) -> T {
    let mut acc = T::default();
    for (k, (c, off)) in terms.iter().enumerate() {
        let v = value(off);
        let mag = c.unsigned_abs();
        let term = if mag == 1 {
            v
        } else {
            T::small(i32::try_from(mag).unwrap_or(1)) * v
        };
        acc = if k == 0 {
            if *c < 0 { -term } else { term }
        } else if *c < 0 {
            acc - term
        } else {
            acc + term
        };
    }
    acc
}

const L1_1: &[(i32, &[usize])] = &[(1, &[1])];
const L1_2: &[(i32, &[usize])] = &[(1, &[0, 1]), (1, &[1, 0]), (-1, &[1, 1])];
const L1_3: &[(i32, &[usize])] = &[
    (1, &[0, 0, 1]),
    (1, &[0, 1, 0]),
    (1, &[1, 0, 0]),
    (-1, &[0, 1, 1]),
    (-1, &[1, 0, 1]),
    (-1, &[1, 1, 0]),
    (1, &[1, 1, 1]),
];
const L_4: &[(i32, &[usize])] = &[
    (1, &[0, 0, 0, 1]),
    (1, &[0, 0, 1, 0]),
    (-1, &[0, 0, 1, 1]),
    (1, &[0, 1, 0, 0]),
    (-1, &[0, 1, 0, 1]),
    (-1, &[0, 1, 1, 0]),
    (1, &[0, 1, 1, 1]),
    (1, &[1, 0, 0, 0]),
    (-1, &[1, 0, 0, 1]),
    (-1, &[1, 0, 1, 0]),
    (1, &[1, 0, 1, 1]),
    (-1, &[1, 1, 0, 0]),
    (1, &[1, 1, 0, 1]),
    (1, &[1, 1, 1, 0]),
    (-1, &[1, 1, 1, 1]),
];
const L2_1: &[(i32, &[usize])] = &[(2, &[1]), (-1, &[2])];
const L2_2: &[(i32, &[usize])] = &[
    (2, &[0, 1]),
    (-1, &[0, 2]),
    (2, &[1, 0]),
    (-4, &[1, 1]),
    (2, &[1, 2]),
    (-1, &[2, 0]),
    (2, &[2, 1]),
    (-1, &[2, 2]),
];
const L2_3: &[(i32, &[usize])] = &[
    (2, &[0, 0, 1]),
    (-1, &[0, 0, 2]),
    (2, &[0, 1, 0]),
    (-4, &[0, 1, 1]),
    (2, &[0, 1, 2]),
    (-1, &[0, 2, 0]),
    (2, &[0, 2, 1]),
    (-1, &[0, 2, 2]),
    (2, &[1, 0, 0]),
    (-4, &[1, 0, 1]),
    (2, &[1, 0, 2]),
    (-4, &[1, 1, 0]),
    (8, &[1, 1, 1]),
    (-4, &[1, 1, 2]),
    (2, &[1, 2, 0]),
    (-4, &[1, 2, 1]),
    (2, &[1, 2, 2]),
    (-1, &[2, 0, 0]),
    (2, &[2, 0, 1]),
    (-1, &[2, 0, 2]),
    (2, &[2, 1, 0]),
    (-4, &[2, 1, 1]),
    (2, &[2, 1, 2]),
    (-1, &[2, 2, 0]),
    (2, &[2, 2, 1]),
    (-1, &[2, 2, 2]),
];

fn lorenzo_terms(nd: usize, layers: u8) -> &'static [(i32, &'static [usize])] {
    match (nd, layers) {
        (1, 1) => L1_1,
        (2, 1) => L1_2,
        (3, 1) => L1_3,
        (1, _) => L2_1,
        (2, _) => L2_2,
        (3, _) => L2_3,
        _ => L_4,
    }
}

/// A Lorenzo prediction at global position `pos` (values before the array's start are 0).
fn lorenzo_at<T: Real>(data: &[T], pos: &[usize], st: &[usize], nd: usize, layers: u8) -> T {
    lorenzo_eval(lorenzo_terms(nd, layers), |off| {
        let mut flat = 0usize;
        for ((p, o), s) in pos.iter().zip(off).zip(st) {
            let Some(g) = p.checked_sub(*o) else {
                return T::default();
            };
            flat = flat.saturating_add(g.saturating_mul(*s));
        }
        data.get(flat).copied().unwrap_or_default()
    })
}

// ----- Lorenzo/regression: general frontend -----

/// The predictors of a block (the composed predictor's members, in their order).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pred {
    Lorenzo1,
    Lorenzo2,
    Regression,
}

/// The linear regression predictor's coefficients, recovered block by block.
struct Regression<T> {
    linear: Quant<T>,
    independent: Quant<T>,
    inds: Vec<i32>,
    next: usize,
    coeffs: Vec<T>,
}

impl<T: Real> Regression<T> {
    fn load(rd: &mut Rd<'_>, nd: usize) -> Result<Self, Error> {
        rd.u8()?;
        let count = rd.usize()?;
        let (independent, linear, inds) = if count != 0 {
            let independent = Quant::load(rd)?;
            let linear = Quant::load(rd)?;
            let h = Huffman::load(rd)?;
            let inds = h.decode(rd, count)?;
            (independent, linear, inds)
        } else {
            let none = || Quant {
                eb: 1.0,
                radius: 32768,
                unpred: Vec::new(),
                next: 0,
            };
            (none(), none(), Vec::new())
        };
        Ok(Self {
            linear,
            independent,
            inds,
            next: 0,
            coeffs: vec![T::default(); nd.saturating_add(1)],
        })
    }

    fn take(&mut self) -> Result<i32, Error> {
        let q = self.inds.get(self.next).copied().ok_or_else(err)?;
        self.next = self.next.saturating_add(1);
        Ok(q)
    }

    /// Recover the next block's coefficients (predicted by the previous block's).
    fn next_block(&mut self, nd: usize) -> Result<(), Error> {
        for i in 0..nd {
            let q = self.take()?;
            let prev = at(&self.coeffs, i)?;
            let v = self.linear.recover(prev, q)?;
            put(&mut self.coeffs, i, v)?;
        }
        let q = self.take()?;
        let prev = at(&self.coeffs, nd)?;
        let v = self.independent.recover(prev, q)?;
        put(&mut self.coeffs, nd, v)
    }

    #[allow(clippy::arithmetic_side_effects)] // floating point (f32/f64) only
    fn predict(&self, local: &[usize]) -> T {
        let mut pred = T::default();
        for (i, l) in local.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let x = T::from_f64(*l as f64);
            pred = x.fma(self.coeffs.get(i).copied().unwrap_or_default(), pred);
        }
        pred + self.coeffs.get(local.len()).copied().unwrap_or_default()
    }
}

fn general_lorenzo_reg<T: Real>(conf: &Config, data: &[u8], limit: usize) -> Result<Vec<T>, Error> {
    let buffer = unzstd(data, limit.saturating_mul(4).saturating_add(1 << 20))?;
    let mut rd = Rd::new(&buffer);
    let nd = conf.dims.len();
    let dims = (0..nd).map(|_| rd.usize()).collect::<Result<Vec<_>, _>>()?;
    let block = usize::try_from(rd.u32()?).map_err(|_| err())?;
    if block == 0 {
        return Err(err());
    }
    let mut members = Vec::new();
    if conf.lorenzo {
        members.push(Pred::Lorenzo1);
    }
    if conf.lorenzo2 {
        members.push(Pred::Lorenzo2);
    }
    if conf.regression {
        members.push(Pred::Regression);
    }
    if conf.regression2 {
        return Err(Error::Unsupported("SZ3 polynomial regression".into()));
    }
    if members.is_empty() {
        return Err(err());
    }
    // Every predictor loads its state, in order; then the composed predictor's choices.
    let mut regression = None;
    for m in &members {
        match m {
            Pred::Regression => regression = Some(Regression::<T>::load(&mut rd, nd)?),
            _ => {
                rd.u8()?;
            }
        }
    }
    let selection = if members.len() > 1 {
        let n = rd.usize()?;
        if n > 0 {
            Huffman::load(&mut rd)?.decode(&mut rd, n)?
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    let mut quant = Quant::<T>::load(&mut rd)?;
    let num = dims
        .iter()
        .try_fold(1usize, |a, d| a.checked_mul(*d))
        .ok_or_else(err)?;
    if num != conf.num || num.checked_mul(T::SIZE).is_none_or(|b| b > limit) {
        return Err(err());
    }
    let inds = Huffman::load(&mut rd)?.decode(&mut rd, num)?;
    let mut out = vec![T::default(); num];
    let st = strides(&dims);
    let grid: Vec<usize> = dims.iter().map(|d| d.div_ceil(block)).collect();
    let nblocks = grid
        .iter()
        .try_fold(1usize, |a, g| a.checked_mul(*g))
        .ok_or_else(err)?;
    let mut q = inds.iter();
    let mut chosen = selection.iter();
    for b in 0..nblocks {
        // The block's position and extent (the last ones along each axis may be short).
        let mut rem = b;
        let mut origin = vec![0usize; nd];
        for k in (0..nd).rev() {
            let g = grid.get(k).copied().unwrap_or(1);
            if let Some(o) = origin.get_mut(k) {
                *o = rem.checked_rem(g).unwrap_or(0).saturating_mul(block);
            }
            rem = rem.checked_div(g).unwrap_or(0);
        }
        let extent: Vec<usize> = origin
            .iter()
            .zip(&dims)
            .map(|(o, d)| d.saturating_sub(*o).min(block))
            .collect();
        let mut pred = if members.len() > 1 {
            let sid = usize::try_from(*chosen.next().ok_or_else(err)?).map_err(|_| err())?;
            *members.get(sid).ok_or_else(err)?
        } else {
            *members.first().ok_or_else(err)?
        };
        if pred == Pred::Regression {
            if extent.iter().any(|e| *e <= 1) {
                pred = Pred::Lorenzo1;
            } else if let Some(r) = regression.as_mut() {
                r.next_block(nd)?;
            }
        }
        let count = extent.iter().product::<usize>();
        let mut local = vec![0usize; nd];
        for _ in 0..count {
            let pos: Vec<usize> = origin
                .iter()
                .zip(&local)
                .map(|(o, l)| o.saturating_add(*l))
                .collect();
            let p = match pred {
                Pred::Lorenzo1 => lorenzo_at(&out, &pos, &st, nd, 1),
                Pred::Lorenzo2 => lorenzo_at(&out, &pos, &st, nd, 2),
                Pred::Regression => regression.as_ref().ok_or_else(err)?.predict(&local),
            };
            let qi = *q.next().ok_or_else(err)?;
            let v = quant.recover(p, qi)?;
            let flat = pos
                .iter()
                .zip(&st)
                .fold(0usize, |a, (p, s)| a.saturating_add(p.saturating_mul(*s)));
            put(&mut out, flat, v)?;
            // Next position in the block, last axis fastest.
            for k in (0..nd).rev() {
                let e = extent.get(k).copied().unwrap_or(1);
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

// ----- Lorenzo/regression: the fast 3-D frontend -----

fn fast_3d<T: Real>(conf: &Config, data: &[u8], limit: usize) -> Result<Vec<T>, Error> {
    let buffer = unzstd(data, limit.saturating_mul(4).saturating_add(1 << 20))?;
    let mut rd = Rd::new(&buffer);
    // `meta_params`, as the C struct is laid out (44 bytes).
    let params = rd.bytes(44)?;
    let int_at = |o: usize| -> Result<i32, Error> {
        Ok(i32::from_le_bytes(
            params
                .get(o..o.saturating_add(4))
                .ok_or_else(err)?
                .try_into()
                .map_err(|_| err())?,
        ))
    };
    let float_at = |o: usize| -> Result<f32, Error> {
        Ok(f32::from_le_bytes(
            params
                .get(o..o.saturating_add(4))
                .ok_or_else(err)?
                .try_into()
                .map_err(|_| err())?,
        ))
    };
    let block = usize::try_from(int_at(0)?).map_err(|_| err())?;
    let pred_dim = int_at(4)?;
    let pad = usize::try_from(int_at(12)?).map_err(|_| err())?;
    let eb_independent = float_at(20)?;
    let eb_linear = float_at(24)?;
    rd.f64()?; // precision
    let use_mean = rd.u8()? != 0;
    let mean: T = rd.real()?;
    let reg_count = rd.usize()?;
    let (d1, d2, d3) = match conf.dims.as_slice() {
        [a, b, c] => (*a, *b, *c),
        _ => return Err(err()),
    };
    if block == 0 || pad < 2 {
        return Err(err());
    }
    let num = d1
        .checked_mul(d2)
        .and_then(|x| x.checked_mul(d3))
        .ok_or_else(err)?;
    if num != conf.num || num.checked_mul(T::SIZE).is_none_or(|b| b > limit) {
        return Err(err());
    }
    let (nx, ny, nz) = (d1.div_ceil(block), d2.div_ceil(block), d3.div_ceil(block));
    let nblocks = nx
        .checked_mul(ny)
        .and_then(|x| x.checked_mul(nz))
        .ok_or_else(err)?;
    let indicator = Huffman::load(&mut rd)?.decode(&mut rd, nblocks)?;
    // Regression coefficients: 4 per regression block, each predicted from the last.
    let mut reg: Vec<f32> = Vec::new();
    if reg_count > 0 {
        let unpred_count = rd.usize()?;
        let unpred: Vec<f32> = (0..unpred_count)
            .map(|_| rd.f32())
            .collect::<Result<_, _>>()?;
        let n = reg_count.checked_mul(4).ok_or_else(err)?;
        let types = Huffman::load(&mut rd)?.decode(&mut rd, n)?;
        let precisions = [
            T::from_f64(f64::from(eb_linear)),
            T::from_f64(f64::from(eb_linear)),
            T::from_f64(f64::from(eb_linear)),
            T::from_f64(f64::from(eb_independent)),
        ];
        reg = vec![0.0f32; n.checked_add(4).ok_or_else(err)?];
        let mut unpred = unpred.iter();
        for (k, t) in types.iter().enumerate() {
            let prev = reg.get(k).copied().ok_or_else(err)?;
            let v = if *t == 0 {
                *unpred.next().ok_or_else(err)?
            } else {
                let p = precisions.get(k % 4).copied().ok_or_else(err)?.to_f64();
                let steps = t.wrapping_sub(32768).wrapping_mul(2);
                #[allow(clippy::cast_possible_truncation)]
                let v = f64::from(steps).mul_add(p, f64::from(prev)) as f32;
                v
            };
            *reg.get_mut(k.saturating_add(4)).ok_or_else(err)? = v;
        }
    }
    let mut quant = Quant::<T>::load(&mut rd)?;
    let inds = Huffman::load(&mut rd)?.decode(&mut rd, num)?;
    let radius = quant.radius;
    let mut out = vec![T::default(); num];
    // The prediction buffer: one block of x-planes (plus padding) over all y and z.
    let (bd1, bd0) = (
        d3.checked_add(pad).ok_or_else(err)?,
        d2.checked_add(pad)
            .and_then(|y| y.checked_mul(d3.checked_add(pad)?))
            .ok_or_else(err)?,
    );
    let blen = block
        .checked_add(pad)
        .and_then(|x| x.checked_mul(bd0))
        .ok_or_else(err)?;
    let mut buf = vec![T::default(); blen];
    let mut types = inds.iter();
    let mut ind = indicator.iter();
    let mut reg_at = 4usize;
    let lead = pad.saturating_mul(bd0.saturating_add(bd1).saturating_add(1));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let sx = d1.saturating_sub(i.saturating_mul(block)).min(block);
                let sy = d2.saturating_sub(j.saturating_mul(block)).min(block);
                let sz = d3.saturating_sub(k.saturating_mul(block)).min(block);
                let selector = *ind.next().ok_or_else(err)?;
                // This block's corner in the buffer and in the data.
                let b0 = lead
                    .saturating_add(j.saturating_mul(block).saturating_mul(bd1))
                    .saturating_add(k.saturating_mul(block));
                let o0 = i
                    .saturating_mul(block)
                    .saturating_mul(d2.saturating_mul(d3))
                    .saturating_add(j.saturating_mul(block).saturating_mul(d3))
                    .saturating_add(k.saturating_mul(block));
                let coeffs = if selector == 1 {
                    let c = [
                        reg.get(reg_at).copied().ok_or_else(err)?,
                        reg.get(reg_at.saturating_add(1)).copied().ok_or_else(err)?,
                        reg.get(reg_at.saturating_add(2)).copied().ok_or_else(err)?,
                        reg.get(reg_at.saturating_add(3)).copied().ok_or_else(err)?,
                    ];
                    reg_at = reg_at.saturating_add(4);
                    Some(c)
                } else {
                    None
                };
                for x in 0..sx {
                    for y in 0..sy {
                        for z in 0..sz {
                            let bpos = b0
                                .saturating_add(x.saturating_mul(bd0))
                                .saturating_add(y.saturating_mul(bd1))
                                .saturating_add(z);
                            let opos = o0
                                .saturating_add(x.saturating_mul(d2.saturating_mul(d3)))
                                .saturating_add(y.saturating_mul(d3))
                                .saturating_add(z);
                            let t = *types.next().ok_or_else(err)?;
                            let v = if t == 0 {
                                quant.recover_unpred()?
                            } else if let Some(c) = coeffs {
                                #[allow(clippy::cast_precision_loss)]
                                let p = c[2]
                                    .mul_add(z as f32, c[0].mul_add(x as f32, c[1] * y as f32))
                                    + c[3];
                                quant.recover_pred(T::from_f64(f64::from(p)), t)
                            } else if use_mean && t == radius {
                                mean
                            } else {
                                let b = |off: usize| -> T {
                                    bpos.checked_sub(off)
                                        .and_then(|p| buf.get(p).copied())
                                        .unwrap_or_default()
                                };
                                let p = meta_lorenzo(&b, bd0, bd1, pred_dim, selector == 2);
                                let t = if use_mean && t > radius {
                                    t.saturating_sub(1)
                                } else {
                                    t
                                };
                                quant.recover_pred(p, t)
                            };
                            put(&mut buf, bpos, v)?;
                            put(&mut out, opos, v)?;
                        }
                    }
                }
            }
        }
        // The last planes become the next slab's padding.
        let from = block.saturating_mul(bd0);
        let n = pad.saturating_mul(bd0);
        let tail: Vec<T> = buf
            .get(from..from.saturating_add(n))
            .ok_or_else(err)?
            .to_vec();
        buf.get_mut(..n).ok_or_else(err)?.copy_from_slice(&tail);
    }
    Ok(out)
}

/// SZ's buffered Lorenzo predictions (`b(o)`: the value `o` places back; `d0`, `d1`: the
/// buffer's plane and row lengths), as `lorenzo_predict_*` in the reference.
#[allow(clippy::arithmetic_side_effects)] // floating point (f32/f64) only
fn meta_lorenzo<T: Real>(b: &impl Fn(usize) -> T, d0: usize, d1: usize, dim: i32, two: bool) -> T {
    let t = T::small;
    let o = |x: usize, y: usize, z: usize| {
        x.saturating_mul(d0)
            .saturating_add(y.saturating_mul(d1))
            .saturating_add(z)
    };
    match (dim, two) {
        (3, false) => {
            b(o(0, 0, 1)) + b(o(0, 1, 0)) + b(o(1, 0, 0))
                - b(o(0, 1, 1))
                - b(o(1, 0, 1))
                - b(o(1, 1, 0))
                + b(o(1, 1, 1))
        }
        (3, true) => lorenzo_eval(L2_3, |off| match off {
            [x, y, z] => b(o(*x, *y, *z)),
            _ => T::default(),
        }),
        // Two dimensions: the row (z) and the plane (x) neighbours, as the reference.
        (2, false) => b(o(0, 0, 1)) + b(o(1, 0, 0)) - b(o(1, 0, 1)),
        (2, true) => {
            t(2) * b(o(1, 0, 0)) - b(o(2, 0, 0)) + t(2) * b(o(0, 0, 1)) - t(4) * b(o(1, 0, 1))
                + t(2) * b(o(2, 0, 1))
                - b(o(0, 0, 2))
                + t(2) * b(o(1, 0, 2))
                - b(o(2, 0, 2))
        }
        (_, false) => b(1),
        (_, true) => t(2) * b(1) - b(2),
    }
}

// ----- interpolation -----

/// Every permutation of `0..n`, in lexicographic order (`std::next_permutation`).
fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn go(prefix: &mut Vec<usize>, left: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if left.is_empty() {
            out.push(prefix.clone());
            return;
        }
        for i in 0..left.len() {
            let v = left.remove(i);
            prefix.push(v);
            go(prefix, left, out);
            prefix.pop();
            left.insert(i, v);
        }
    }
    let mut out = Vec::new();
    go(&mut Vec::new(), &mut (0..n).collect(), &mut out);
    out
}

struct Interp<'a, T> {
    data: Vec<T>,
    quant: Quant<T>,
    inds: core::slice::Iter<'a, i32>,
    cubic: bool,
}

impl<T: Real> Interp<'_, T> {
    fn v(&self, i: usize) -> Result<T, Error> {
        at(&self.data, i)
    }

    fn recover(&mut self, i: usize, pred: T) -> Result<(), Error> {
        let q = *self.inds.next().ok_or_else(err)?;
        let v = self.quant.recover(pred, q)?;
        put(&mut self.data, i, v)
    }

    /// The value `k` strides before (negative: after) `d`.
    fn near(&self, d: usize, k: isize, s: usize) -> Result<T, Error> {
        let delta = k.unsigned_abs().checked_mul(s).ok_or_else(err)?;
        let i = if k < 0 {
            d.checked_add(delta)
        } else {
            d.checked_sub(delta)
        };
        self.v(i.ok_or_else(err)?)
    }

    /// One line from `begin` to `end` (inclusive) in steps of `stride`.
    #[allow(clippy::arithmetic_side_effects)] // floating point (f32/f64) only
    fn line(&mut self, begin: usize, end: usize, stride: usize) -> Result<(), Error> {
        let n = end
            .saturating_sub(begin)
            .checked_div(stride)
            .ok_or_else(err)?
            .saturating_add(1);
        if n <= 1 {
            return Ok(());
        }
        let s = stride;
        let pos = |i: usize| begin.saturating_add(i.saturating_mul(s));
        let c = |k: i32| T::small(k);
        if !self.cubic || n < 5 {
            let mut i = 1usize;
            while i.saturating_add(1) < n {
                let d = pos(i);
                let p = (self.near(d, 1, s)? + self.near(d, -1, s)?) / c(2);
                self.recover(d, p)?;
                i = i.saturating_add(2);
            }
            if n % 2 == 0 {
                let d = pos(n.saturating_sub(1));
                let p = if n < 4 {
                    self.near(d, 1, s)?
                } else {
                    // `-0.5 * a + 1.5 * b`: double constants, so double arithmetic.
                    let (a, b) = (self.near(d, 3, s)?.to_f64(), self.near(d, 1, s)?.to_f64());
                    T::from_f64((-0.5f64).mul_add(a, 1.5 * b))
                };
                self.recover(d, p)?;
            }
            return Ok(());
        }
        let mut i = 3usize;
        while i.saturating_add(3) < n {
            let d = pos(i);
            let (a, b) = (self.near(d, 3, s)?, self.near(d, 1, s)?);
            let (cc, dd) = (self.near(d, -1, s)?, self.near(d, -3, s)?);
            // The multiplications fused into the additions, as the plugin's build does.
            self.recover(d, (c(9).fma(cc, c(9).fma(b, -a)) - dd) / c(16))?;
            i = i.saturating_add(2);
        }
        let d = pos(1);
        let (a, b, cc) = (
            self.near(d, 1, s)?,
            self.near(d, -1, s)?,
            self.near(d, -3, s)?,
        );
        self.recover(d, (c(3).fma(a, c(6) * b) - cc) / c(8))?;
        let d = pos(i);
        let (a, b, cc) = (
            self.near(d, 3, s)?,
            self.near(d, 1, s)?,
            self.near(d, -1, s)?,
        );
        self.recover(d, c(3).fma(cc, c(6).fma(b, -a)) / c(8))?;
        if n % 2 == 0 {
            let d = pos(n.saturating_sub(1));
            let (a, b, cc) = (
                self.near(d, 5, s)?,
                self.near(d, 3, s)?,
                self.near(d, 1, s)?,
            );
            self.recover(d, c(15).fma(cc, c(3).fma(a, -(c(10) * b))) / c(8))?;
        }
        Ok(())
    }
}

/// The lines of one block at one level, dimension after dimension in the order `dims`:
/// along `dims[m]`, at the positions of the other dimensions already known (every
/// `2 * stride` in those not yet interpolated, every `stride` in those done).
fn interp_block<T: Real>(
    it: &mut Interp<'_, T>,
    begin: &[usize],
    end: &[usize],
    order: &[usize],
    offsets: &[usize],
    stride: usize,
) -> Result<(), Error> {
    let nd = order.len();
    for m in 0..nd {
        let along = *order.get(m).ok_or_else(err)?;
        // Ranges of the other dimensions, in `order` order.
        let mut ranges: Vec<(usize, usize, usize)> = Vec::new();
        for (r, dim) in order.iter().enumerate() {
            if r == m {
                continue;
            }
            let b = *begin.get(*dim).ok_or_else(err)?;
            let e = *end.get(*dim).ok_or_else(err)?;
            let step = if r < m {
                stride
            } else {
                stride.saturating_mul(2)
            };
            let first = if b == 0 { 0 } else { b.saturating_add(step) };
            ranges.push((first, e, step));
        }
        let mut idx: Vec<usize> = ranges.iter().map(|r| r.0).collect();
        if ranges.iter().any(|r| r.0 > r.1) {
            continue;
        }
        let ab = *begin.get(along).ok_or_else(err)?;
        let ae = *end.get(along).ok_or_else(err)?;
        let off_along = *offsets.get(along).ok_or_else(err)?;
        loop {
            let mut base = ab.saturating_mul(off_along);
            let mut it_dims = order.iter().enumerate().filter(|(r, _)| *r != m);
            for v in &idx {
                let (_, dim) = it_dims.next().ok_or_else(err)?;
                base = base.saturating_add(v.saturating_mul(*offsets.get(*dim).ok_or_else(err)?));
            }
            it.line(
                base,
                base.saturating_add(ae.saturating_sub(ab).saturating_mul(off_along)),
                stride.saturating_mul(off_along),
            )?;
            // Next index, the last of the other dimensions fastest.
            let mut k = idx.len();
            let mut done = true;
            while k > 0 {
                k = k.saturating_sub(1);
                let (first, last, step) = *ranges.get(k).ok_or_else(err)?;
                let slot = idx.get_mut(k).ok_or_else(err)?;
                *slot = slot.saturating_add(step);
                if *slot <= last {
                    done = false;
                    break;
                }
                *slot = first;
            }
            if done {
                break;
            }
        }
    }
    Ok(())
}

fn interpolation<T: Real>(conf: &Config, data: &[u8], limit: usize) -> Result<Vec<T>, Error> {
    let buffer = unzstd(data, limit.saturating_mul(4).saturating_add(1 << 20))?;
    let mut rd = Rd::new(&buffer);
    let nd = conf.dims.len();
    let dims = (0..nd).map(|_| rd.usize()).collect::<Result<Vec<_>, _>>()?;
    let block = usize::try_from(rd.u32()?).map_err(|_| err())?;
    let interpolator = rd.i32()?;
    let direction = usize::try_from(rd.i32()?).map_err(|_| err())?;
    let num = dims
        .iter()
        .try_fold(1usize, |a, d| a.checked_mul(*d))
        .ok_or_else(err)?;
    if num != conf.num
        || num == 0
        || block == 0
        || num.checked_mul(T::SIZE).is_none_or(|b| b > limit)
    {
        return Err(err());
    }
    let quant = Quant::<T>::load(&mut rd)?;
    let inds = Huffman::load(&mut rd)?.decode(&mut rd, num)?;
    let offsets = strides(&dims);
    let orders = permutations(nd);
    let order = orders.get(direction).ok_or_else(err)?.clone();
    // Levels: as many as the longest dimension needs to halve to 1.
    let levels = dims
        .iter()
        .map(|d| {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let l = (*d as f64).log2().ceil() as u32;
            l
        })
        .max()
        .unwrap_or(0);
    let eb = quant.eb;
    let mut it = Interp {
        data: vec![T::default(); num],
        quant,
        inds: inds.iter(),
        cubic: interpolator == 1,
    };
    let first = *it.inds.next().ok_or_else(err)?;
    let v = it.quant.recover(T::default(), first)?;
    put(&mut it.data, 0, v)?;
    for level in (1..=levels).rev() {
        it.quant.eb = if level >= 3 { eb * 0.5 } else { eb };
        let stride = 1usize
            .checked_shl(level.saturating_sub(1))
            .ok_or_else(err)?;
        let step = stride.saturating_mul(block);
        let grid: Vec<usize> = dims
            .iter()
            .map(|d| {
                d.saturating_sub(1)
                    .checked_div(step)
                    .unwrap_or(0)
                    .saturating_add(1)
            })
            .collect();
        let nblocks = grid
            .iter()
            .try_fold(1usize, |a, g| a.checked_mul(*g))
            .ok_or_else(err)?;
        for b in 0..nblocks {
            let mut rem = b;
            let mut begin = vec![0usize; nd];
            for k in (0..nd).rev() {
                let g = grid.get(k).copied().unwrap_or(1);
                if let Some(slot) = begin.get_mut(k) {
                    *slot = rem.checked_rem(g).unwrap_or(0).saturating_mul(step);
                }
                rem = rem.checked_div(g).unwrap_or(0);
            }
            let end: Vec<usize> = begin
                .iter()
                .zip(&dims)
                .map(|(b, d)| b.saturating_add(step).min(d.saturating_sub(1)))
                .collect();
            interp_block(&mut it, &begin, &end, &order, &offsets, stride)?;
        }
    }
    Ok(it.data)
}

/// Decode an SZ3 buffer of `T`s into native-endian bytes, at most `limit` bytes.
///
/// # Errors
/// [`Error::Corrupt`] for malformed buffers; [`Error::Unsupported`] for the features listed
/// in the module documentation.
pub fn decompress<T: Real>(data: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    let (conf, end) = config(data)?;
    let num = conf
        .dims
        .iter()
        .try_fold(1usize, |a, d| a.checked_mul(*d))
        .ok_or_else(err)?;
    if num != conf.num || num.checked_mul(T::SIZE).is_none_or(|b| b > limit) {
        return Err(Error::TooLarge(CODEC.into()));
    }
    let values: Vec<T> = if conf.abs == 0.0 {
        // Stored losslessly (the whole buffer is the lossless stage's input).
        let raw = unzstd(data, limit)?;
        let mut rd = Rd::new(&raw);
        rd.reals(num)?
    } else {
        let body = data.get(..end).ok_or_else(err)?;
        match conf.algo {
            0 if conf.dims.len() == 3 && !conf.regression2 => fast_3d(&conf, body, limit)?,
            0 => general_lorenzo_reg(&conf, body, limit)?,
            2 => interpolation(&conf, body, limit)?,
            a => return Err(Error::Unsupported(format!("SZ3 algorithm {a}"))),
        }
    };
    Ok(values.into_iter().flat_map(Real::to_ne).collect())
}
