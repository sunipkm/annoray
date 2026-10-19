//! ZFP (LLNL's compressed floating-point and integer arrays) decoding, bit for bit as the
//! reference library decodes (codec version 5, `ZFP_ROUND_NEVER`): fixed-rate,
//! fixed-precision, fixed-accuracy and reversible modes; 32- and 64-bit floats and integers;
//! 1 to 4 dimensions.
//!
//! A field is cut into blocks of 4 values per dimension (the first dimension, `x`, varying
//! fastest), each decoded from a bit stream read least significant bit first in 64-bit
//! little-endian words: a common exponent (floats), bit planes of the coefficients in
//! negabinary with group tests, then the inverse decorrelating transform. The integer
//! arithmetic wraps where the reference's does (two's complement).

use crate::Error;
use crate::bits::corrupt;

const CODEC: &str = "zfp";

fn err() -> Error {
    corrupt(CODEC)
}

/// The value type of a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZfpType {
    /// 32-bit signed integers.
    Int32,
    /// 64-bit signed integers.
    Int64,
    /// 32-bit floats.
    Float,
    /// 64-bit floats.
    Double,
}

impl ZfpType {
    /// Bytes per value.
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Self::Int32 | Self::Float => 4,
            Self::Int64 | Self::Double => 8,
        }
    }
}

/// What a ZFP header describes: the field and the compression parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ZfpHeader {
    /// The value type.
    pub ztype: ZfpType,
    /// Extents, `x` (fastest varying) first.
    pub dims: Vec<usize>,
    /// Least bits per block.
    pub minbits: u32,
    /// Most bits per block.
    pub maxbits: u32,
    /// Most bit planes.
    pub maxprec: u32,
    /// Least exponent (below the library's minimum: reversible mode).
    pub minexp: i32,
}

const MIN_BITS: u32 = 1;
const MAX_BITS: u32 = 16658;
const MAX_PREC: u32 = 64;
const MIN_EXP: i32 = -1074;
const MODE_SHORT_MAX: u64 = (1 << 12) - 2;

/// Bits least significant first, from 64-bit little-endian words.
struct Bits<'a> {
    data: &'a [u8],
    pos: u64,
}

impl Bits<'_> {
    fn bit(&mut self) -> Result<bool, Error> {
        let byte = usize::try_from(self.pos / 8).map_err(|_| err())?;
        let b = self.data.get(byte).copied().ok_or_else(err)?;
        let v = (b >> (self.pos % 8)) & 1 == 1;
        self.pos = self.pos.checked_add(1).ok_or_else(err)?;
        Ok(v)
    }

    /// `n` (at most 64) bits, the first read the least significant.
    fn bits(&mut self, n: u32) -> Result<u64, Error> {
        let mut v = 0u64;
        let mut i = 0u32;
        // Whole bytes when aligned, else bit by bit.
        while i < n {
            if self.pos.is_multiple_of(8) && n.saturating_sub(i) >= 8 {
                let byte = usize::try_from(self.pos / 8).map_err(|_| err())?;
                let b = self.data.get(byte).copied().ok_or_else(err)?;
                v |= u64::from(b).checked_shl(i).unwrap_or(0);
                self.pos = self.pos.checked_add(8).ok_or_else(err)?;
                i = i.saturating_add(8);
            } else {
                v |= u64::from(self.bit()?).checked_shl(i).unwrap_or(0);
                i = i.saturating_add(1);
            }
        }
        Ok(v)
    }

    fn skip(&mut self, n: u64) -> Result<(), Error> {
        self.pos = self.pos.checked_add(n).ok_or_else(err)?;
        Ok(())
    }
}

/// Parse a full ZFP header (magic, field metadata, mode) from the start of `bytes`.
///
/// # Errors
/// [`Error::Corrupt`] for a missing magic or invalid values; [`Error::Unsupported`] for
/// codec versions other than 5.
pub fn read_header(bytes: &[u8]) -> Result<ZfpHeader, Error> {
    let mut s = Bits {
        data: bytes,
        pos: 0,
    };
    let magic = [s.bits(8)?, s.bits(8)?, s.bits(8)?];
    if magic != [u64::from(b'z'), u64::from(b'f'), u64::from(b'p')] {
        return Err(err());
    }
    let codec = s.bits(8)?;
    if codec != 5 {
        return Err(Error::Unsupported(format!("ZFP codec version {codec}")));
    }
    let mut meta = s.bits(52)?;
    let ztype = match meta & 3 {
        0 => ZfpType::Int32,
        1 => ZfpType::Int64,
        2 => ZfpType::Float,
        _ => ZfpType::Double,
    };
    meta >>= 2;
    let ndims = meta & 3;
    meta >>= 2;
    // Extents per dimension: 32 bits (of 48) for one, else 24, 16 or 12 bits each.
    let (width, count, mask): (u32, usize, u64) = match ndims {
        0 => (48, 1, 0xffff_ffff),
        1 => (24, 2, 0x00ff_ffff),
        2 => (16, 3, 0xffff),
        _ => (12, 4, 0x0fff),
    };
    let mut dims = Vec::with_capacity(count);
    for _ in 0..count {
        let d = usize::try_from(meta & mask).map_err(|_| err())?;
        dims.push(d.checked_add(1).ok_or_else(err)?);
        meta = meta.checked_shr(width).unwrap_or(0);
    }
    let mut mode = s.bits(12)?;
    if mode > MODE_SHORT_MAX {
        mode |= s.bits(52)?.checked_shl(12).unwrap_or(0);
    }
    let field = |v: u64, mask: u64, shift: u32| -> Result<u32, Error> {
        u32::try_from(v.checked_shr(shift).unwrap_or(0) & mask).map_err(|_| err())
    };
    let (minbits, maxbits, maxprec, minexp) = if mode <= MODE_SHORT_MAX {
        let m = u32::try_from(mode).map_err(|_| err())?;
        if m < 2048 {
            (m.saturating_add(1), m.saturating_add(1), MAX_PREC, MIN_EXP)
        } else if m < 2048 + 128 {
            (MIN_BITS, MAX_BITS, m.saturating_sub(2047), MIN_EXP)
        } else if m == 2048 + 128 {
            (MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP.saturating_sub(1))
        } else {
            let e = i32::try_from(m).map_err(|_| err())?;
            let minexp = e.saturating_add(MIN_EXP).saturating_sub(2048 + 128 + 1);
            (MIN_BITS, MAX_BITS, MAX_PREC, minexp)
        }
    } else {
        let minbits = field(mode, 0x7fff, 12)?.saturating_add(1);
        let maxbits = field(mode, 0x7fff, 27)?.saturating_add(1);
        let maxprec = field(mode, 0x7f, 42)?.saturating_add(1);
        let minexp = i32::try_from(field(mode, 0x7fff, 49)?)
            .map_err(|_| err())?
            .saturating_sub(16495);
        (minbits, maxbits, maxprec, minexp)
    };
    if minbits > maxbits || maxprec == 0 || maxprec > 64 {
        return Err(err());
    }
    Ok(ZfpHeader {
        ztype,
        dims,
        minbits,
        maxbits,
        maxprec,
        minexp,
    })
}

const PERM_1: [u8; 4] = [0, 1, 2, 3];
const PERM_2: [u8; 16] = [0, 1, 4, 5, 2, 8, 6, 9, 3, 12, 10, 7, 13, 11, 14, 15];
const PERM_3: [u8; 64] = [
    0, 1, 4, 16, 20, 17, 5, 2, 8, 32, 21, 6, 18, 24, 9, 33, 36, 3, 12, 48, 22, 25, 37, 40, 34, 10,
    7, 19, 28, 13, 49, 52, 41, 38, 26, 23, 29, 53, 11, 35, 44, 14, 50, 56, 42, 27, 39, 45, 30, 54,
    57, 60, 51, 15, 43, 46, 58, 61, 55, 31, 62, 59, 47, 63,
];
const PERM_4: [u8; 256] = [
    0, 1, 4, 16, 64, 5, 80, 17, 68, 65, 20, 2, 8, 32, 128, 84, 81, 69, 21, 6, 18, 66, 24, 72, 9,
    96, 33, 36, 129, 132, 144, 3, 12, 48, 192, 85, 82, 70, 22, 73, 25, 88, 37, 100, 97, 148, 145,
    133, 10, 160, 34, 136, 130, 40, 7, 19, 67, 28, 76, 13, 112, 49, 52, 193, 196, 208, 86, 89, 101,
    149, 161, 137, 41, 134, 38, 164, 26, 152, 146, 104, 98, 74, 83, 71, 23, 77, 29, 92, 53, 116,
    113, 212, 209, 197, 11, 35, 131, 44, 140, 14, 176, 50, 56, 194, 200, 224, 90, 165, 102, 153,
    150, 105, 168, 162, 138, 42, 87, 93, 117, 213, 27, 75, 99, 39, 135, 147, 108, 45, 141, 156, 30,
    78, 177, 180, 54, 114, 120, 57, 198, 210, 216, 201, 225, 228, 15, 240, 51, 204, 195, 60, 169,
    166, 154, 106, 91, 103, 151, 109, 157, 94, 181, 118, 121, 214, 217, 229, 163, 139, 43, 142, 46,
    172, 58, 184, 178, 232, 226, 202, 241, 205, 61, 199, 55, 244, 31, 220, 211, 124, 115, 79, 170,
    167, 155, 107, 158, 110, 173, 122, 185, 182, 233, 230, 218, 95, 245, 119, 221, 215, 125, 242,
    206, 62, 203, 59, 248, 47, 236, 227, 188, 179, 143, 171, 174, 186, 234, 246, 222, 126, 219,
    123, 249, 111, 237, 231, 189, 183, 159, 252, 243, 207, 63, 175, 250, 187, 238, 235, 190, 253,
    247, 223, 127, 254, 251, 239, 191, 255,
];

fn perm(dims: usize) -> &'static [u8] {
    match dims {
        1 => &PERM_1,
        2 => &PERM_2,
        3 => &PERM_3,
        _ => &PERM_4,
    }
}

/// Values per block: 4 per dimension.
fn block_size(dims: usize) -> usize {
    match dims {
        1 => 4,
        2 => 16,
        3 => 64,
        _ => 256,
    }
}

/// Decode `size` unsigned coefficients of `intprec` bits, at most `maxprec` bit planes and
/// (when that can bind) `maxbits` bits; the bits read.
fn decode_ints(
    s: &mut Bits<'_>,
    maxbits: u32,
    maxprec: u32,
    intprec: u32,
    size: usize,
) -> Result<(Vec<u64>, u32), Error> {
    let mut data = vec![0u64; size];
    let kmin = intprec.saturating_sub(maxprec);
    // `with_maxbits`: whether the rate can bind at all.
    let size32 = u32::try_from(size).map_err(|_| err())?;
    let constrained = maxprec
        .saturating_add(1)
        .saturating_mul(size32)
        .saturating_sub(1)
        > maxbits;
    let budget: u64 = if constrained {
        u64::from(maxbits)
    } else {
        u64::MAX
    };
    let mut bits = budget;
    let mut n = 0usize;
    let mut k = intprec;
    while bits > 0 && k > kmin {
        k = k.saturating_sub(1);
        let plane = 1u64.checked_shl(k).unwrap_or(0);
        // Step 1: the first n bits of plane k, as they are.
        let m = u64::try_from(n).map_err(|_| err())?.min(bits);
        bits = bits.saturating_sub(m);
        let m = usize::try_from(m).map_err(|_| err())?;
        for d in data.iter_mut().take(m) {
            if s.bit()? {
                *d = d.wrapping_add(plane);
            }
        }
        // Step 2: unary run-length decode the rest of the plane.
        while bits > 0 && n < size {
            bits = bits.saturating_sub(1);
            if !s.bit()? {
                break;
            }
            // Positive group test: scan for the next one-bit.
            while bits > 0 && n.saturating_add(1) < size {
                bits = bits.saturating_sub(1);
                if s.bit()? {
                    break;
                }
                n = n.saturating_add(1);
            }
            if let Some(d) = data.get_mut(n) {
                *d = d.wrapping_add(plane);
            }
            n = n.saturating_add(1);
        }
    }
    let used = budget.saturating_sub(bits);
    Ok((data, u32::try_from(used).unwrap_or(u32::MAX)))
}

/// Apply `f` to the four values `p[at + i * s]`, `i` in `0..4` (unchanged out of range).
fn lift4<T: Copy>(p: &mut [T], at: usize, s: usize, f: impl Fn([T; 4]) -> [T; 4]) {
    let idx = [
        at,
        at.wrapping_add(s),
        at.wrapping_add(s.wrapping_mul(2)),
        at.wrapping_add(s.wrapping_mul(3)),
    ];
    let (Some(&a), Some(&b), Some(&c), Some(&d)) =
        (p.get(idx[0]), p.get(idx[1]), p.get(idx[2]), p.get(idx[3]))
    else {
        return;
    };
    for (i, v) in idx.iter().zip(f([a, b, c, d])) {
        if let Some(slot) = p.get_mut(*i) {
            *slot = v;
        }
    }
}

/// The inverse decorrelating transform of a block, one dimension after another (the
/// slowest first) as the reference orders it.
fn inv_xform<T: Copy>(p: &mut [T], dims: usize, lift: &impl Fn([T; 4]) -> [T; 4]) {
    let r = 0..4usize;
    match dims {
        1 => lift4(p, 0, 1, lift),
        2 => {
            for x in r.clone() {
                lift4(p, x, 4, lift);
            }
            for y in r {
                lift4(p, y.wrapping_mul(4), 1, lift);
            }
        }
        3 => {
            for y in r.clone() {
                for x in r.clone() {
                    lift4(p, x.wrapping_add(y.wrapping_mul(4)), 16, lift);
                }
            }
            for x in r.clone() {
                for z in r.clone() {
                    lift4(p, z.wrapping_mul(16).wrapping_add(x), 4, lift);
                }
            }
            for z in r.clone() {
                for y in r.clone() {
                    lift4(
                        p,
                        y.wrapping_mul(4).wrapping_add(z.wrapping_mul(16)),
                        1,
                        lift,
                    );
                }
            }
        }
        _ => {
            let at = |x: usize, y: usize, z: usize, w: usize| {
                x.wrapping_add(y.wrapping_mul(4))
                    .wrapping_add(z.wrapping_mul(16))
                    .wrapping_add(w.wrapping_mul(64))
            };
            for z in r.clone() {
                for y in r.clone() {
                    for x in r.clone() {
                        lift4(p, at(x, y, z, 0), 64, lift);
                    }
                }
            }
            for y in r.clone() {
                for x in r.clone() {
                    for w in r.clone() {
                        lift4(p, at(x, y, 0, w), 16, lift);
                    }
                }
            }
            for x in r.clone() {
                for w in r.clone() {
                    for z in r.clone() {
                        lift4(p, at(x, 0, z, w), 4, lift);
                    }
                }
            }
            for w in r.clone() {
                for z in r.clone() {
                    for y in r.clone() {
                        lift4(p, at(0, y, z, w), 1, lift);
                    }
                }
            }
        }
    }
}

macro_rules! int_codec {
    ($m:ident, $int:ty, $uint:ty, $bits:expr, $pbits:expr, $nbmask:expr) => {
        mod $m {
            use super::{Bits, Error, block_size, decode_ints, err, inv_xform, perm};

            const INTPREC: u32 = $bits;

            /// The non-orthogonal inverse lifting step (wrapping, as the reference's).
            fn inv_lift(v: [$int; 4]) -> [$int; 4] {
                let [mut x, mut y, mut z, mut w] = v;
                y = y.wrapping_add(w >> 1);
                w = w.wrapping_sub(y >> 1);
                y = y.wrapping_add(w);
                w = w.wrapping_sub(y.wrapping_sub(w));
                z = z.wrapping_add(x);
                x = x.wrapping_sub(z.wrapping_sub(x));
                y = y.wrapping_add(z);
                z = z.wrapping_sub(y.wrapping_sub(z));
                w = w.wrapping_add(x);
                x = x.wrapping_sub(w.wrapping_sub(x));
                [x, y, z, w]
            }

            /// The reversible (Pascal-matrix) inverse lifting step, in unsigned arithmetic.
            fn rev_inv_lift(v: [$int; 4]) -> [$int; 4] {
                let [x, y, z, w] = v.map(<$uint>::from_ne_bytes_of);
                let w = w.wrapping_add(z);
                let z = z.wrapping_add(y);
                let w = w.wrapping_add(z);
                let y = y.wrapping_add(x);
                let z = z.wrapping_add(y);
                let w = w.wrapping_add(z);
                [x, y, z, w].map(<$int>::from_ne_bytes_of)
            }

            trait Reinterpret<T> {
                fn from_ne_bytes_of(v: T) -> Self;
            }
            impl Reinterpret<$int> for $uint {
                fn from_ne_bytes_of(v: $int) -> Self {
                    <$uint>::from_ne_bytes(v.to_ne_bytes())
                }
            }
            impl Reinterpret<$uint> for $int {
                fn from_ne_bytes_of(v: $uint) -> Self {
                    <$int>::from_ne_bytes(v.to_ne_bytes())
                }
            }

            /// Negabinary coefficients, reordered, as two's complement integers.
            fn reorder(ublock: &[u64], dims: usize) -> Result<Vec<$int>, Error> {
                let mut iblock = vec![<$int>::default(); ublock.len()];
                let mask: $uint = $nbmask;
                for (u, &p) in ublock.iter().zip(perm(dims)) {
                    let x = <$uint>::try_from(*u).map_err(|_| err())?;
                    let v = <$int>::from_ne_bytes_of((x ^ mask).wrapping_sub(mask));
                    *iblock.get_mut(usize::from(p)).ok_or_else(err)? = v;
                }
                Ok(iblock)
            }

            /// A block of integers (lossy coding); the bits read.
            pub(super) fn decode_block(
                s: &mut Bits<'_>,
                minbits: u32,
                maxbits: u32,
                maxprec: u32,
                dims: usize,
            ) -> Result<(Vec<$int>, u32), Error> {
                let (ublock, mut bits) =
                    decode_ints(s, maxbits, maxprec, INTPREC, block_size(dims))?;
                if bits < minbits {
                    s.skip(u64::from(minbits.saturating_sub(bits)))?;
                    bits = minbits;
                }
                let mut iblock = reorder(&ublock, dims)?;
                inv_xform(&mut iblock, dims, &inv_lift);
                Ok((iblock, bits))
            }

            /// A block of integers (reversible coding); the bits read.
            pub(super) fn rev_decode_block(
                s: &mut Bits<'_>,
                minbits: u32,
                maxbits: u32,
                dims: usize,
            ) -> Result<(Vec<$int>, u32), Error> {
                let pbits: u32 = $pbits;
                let prec = u32::try_from(s.bits(pbits)?)
                    .map_err(|_| err())?
                    .saturating_add(1);
                let (ublock, used) = decode_ints(
                    s,
                    maxbits.saturating_sub(pbits),
                    prec,
                    INTPREC,
                    block_size(dims),
                )?;
                let mut bits = pbits.saturating_add(used);
                if bits < minbits {
                    s.skip(u64::from(minbits.saturating_sub(bits)))?;
                    bits = minbits;
                }
                let mut iblock = reorder(&ublock, dims)?;
                inv_xform(&mut iblock, dims, &rev_inv_lift);
                Ok((iblock, bits))
            }
        }
    };
}

int_codec!(i32c, i32, u32, 32, 5, 0xaaaa_aaaa_u32);
int_codec!(i64c, i64, u64, 64, 6, 0xaaaa_aaaa_aaaa_aaaa_u64);

/// `2^e` as an `f32` (0 below the smallest subnormal).
fn pow2_f32(e: i32) -> f32 {
    if e > 127 {
        f32::INFINITY
    } else if e >= -126 {
        u32::try_from(e.saturating_add(127)).map_or(0.0, |b| f32::from_bits(b << 23))
    } else if e >= -149 {
        u32::try_from(e.saturating_add(149))
            .ok()
            .and_then(|k| 1u32.checked_shl(k))
            .map_or(0.0, f32::from_bits)
    } else {
        0.0
    }
}

fn pow2_f64(e: i32) -> f64 {
    if e > 1023 {
        f64::INFINITY
    } else if e >= -1022 {
        u64::try_from(e.saturating_add(1023)).map_or(0.0, |b| f64::from_bits(b << 52))
    } else if e >= -1074 {
        u32::try_from(e.saturating_add(1074))
            .ok()
            .and_then(|k| 1u64.checked_shl(k))
            .map_or(0.0, f64::from_bits)
    } else {
        0.0
    }
}

/// The bit planes to decode for a block with largest exponent `emax`.
fn precision(emax: i32, maxprec: u32, minexp: i32, dims: usize) -> u32 {
    let d = i32::try_from(dims).unwrap_or(4);
    let p = emax
        .saturating_sub(minexp)
        .saturating_add(d.saturating_mul(2))
        .saturating_add(2)
        .max(0);
    maxprec.min(u32::try_from(p).unwrap_or(0))
}

/// One block's values.
enum Block {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl Block {
    fn put(&self, i: usize, dst: &mut [u8]) -> Result<(), Error> {
        let src: Vec<u8> = match self {
            Self::F32(v) => v.get(i).ok_or_else(err)?.to_ne_bytes().to_vec(),
            Self::F64(v) => v.get(i).ok_or_else(err)?.to_ne_bytes().to_vec(),
            Self::I32(v) => v.get(i).ok_or_else(err)?.to_ne_bytes().to_vec(),
            Self::I64(v) => v.get(i).ok_or_else(err)?.to_ne_bytes().to_vec(),
        };
        if src.len() != dst.len() {
            return Err(err());
        }
        dst.copy_from_slice(&src);
        Ok(())
    }
}

/// The bits a float block's leading tests used, for the integer block's limits.
struct Limits {
    minbits: u32,
    maxbits: u32,
}

fn limits(h: &ZfpHeader, bits: u32) -> Limits {
    Limits {
        minbits: h.minbits.saturating_sub(bits.min(h.minbits)),
        maxbits: h.maxbits.saturating_sub(bits),
    }
}

/// The leading tests of a float block. `None`: all zero; `Some(None)`: reversible block of
/// reinterpreted integers; `Some(Some(emax))`: the common exponent follows.
fn float_head(
    s: &mut Bits<'_>,
    h: &ZfpHeader,
    ebits: u32,
    ebias: i32,
) -> Result<(Option<Option<i32>>, u32), Error> {
    let reversible = h.minexp < MIN_EXP;
    let mut bits: u32 = 1;
    if !s.bit()? {
        if h.minbits > bits {
            s.skip(u64::from(h.minbits.saturating_sub(bits)))?;
        }
        return Ok((None, bits));
    }
    if reversible {
        bits = bits.saturating_add(1);
        if s.bit()? {
            return Ok((Some(None), bits));
        }
    }
    bits = bits.saturating_add(ebits);
    let e = i32::try_from(s.bits(ebits)?).map_err(|_| err())?;
    Ok((Some(Some(e.saturating_sub(ebias))), bits))
}

fn decode_block(s: &mut Bits<'_>, h: &ZfpHeader, dims: usize) -> Result<Block, Error> {
    let size = block_size(dims);
    let reversible = h.minexp < MIN_EXP;
    Ok(match h.ztype {
        ZfpType::Int32 => Block::I32(if reversible {
            i32c::rev_decode_block(s, h.minbits, h.maxbits, dims)?.0
        } else {
            i32c::decode_block(s, h.minbits, h.maxbits, h.maxprec, dims)?.0
        }),
        ZfpType::Int64 => Block::I64(if reversible {
            i64c::rev_decode_block(s, h.minbits, h.maxbits, dims)?.0
        } else {
            i64c::decode_block(s, h.minbits, h.maxbits, h.maxprec, dims)?.0
        }),
        ZfpType::Float => {
            let (head, bits) = float_head(s, h, 8, 127)?;
            let l = limits(h, bits);
            Block::F32(match head {
                None => vec![0.0; size],
                Some(None) => {
                    // Two's complement to sign-magnitude, reinterpreted as floats.
                    let (iblock, _) = i32c::rev_decode_block(s, l.minbits, l.maxbits, dims)?;
                    iblock
                        .iter()
                        .map(|x| {
                            let u = u32::from_ne_bytes(x.to_ne_bytes());
                            f32::from_bits(if *x < 0 { u ^ 0x7fff_ffff } else { u })
                        })
                        .collect()
                }
                Some(Some(emax)) => {
                    let iblock = if reversible {
                        i32c::rev_decode_block(s, l.minbits, l.maxbits, dims)?.0
                    } else {
                        let maxprec = precision(emax, h.maxprec, h.minexp, dims);
                        i32c::decode_block(s, l.minbits, l.maxbits, maxprec, dims)?.0
                    };
                    if reversible && emax == -127 {
                        vec![0.0; size]
                    } else {
                        let scale = pow2_f32(emax.saturating_sub(30));
                        // The integers become floats first (rounded), as in C.
                        #[allow(clippy::cast_precision_loss)]
                        iblock.iter().map(|x| scale * (*x as f32)).collect()
                    }
                }
            })
        }
        ZfpType::Double => {
            let (head, bits) = float_head(s, h, 11, 1023)?;
            let l = limits(h, bits);
            Block::F64(match head {
                None => vec![0.0; size],
                Some(None) => {
                    let (iblock, _) = i64c::rev_decode_block(s, l.minbits, l.maxbits, dims)?;
                    iblock
                        .iter()
                        .map(|x| {
                            let u = u64::from_ne_bytes(x.to_ne_bytes());
                            f64::from_bits(if *x < 0 { u ^ 0x7fff_ffff_ffff_ffff } else { u })
                        })
                        .collect()
                }
                Some(Some(emax)) => {
                    let iblock = if reversible {
                        i64c::rev_decode_block(s, l.minbits, l.maxbits, dims)?.0
                    } else {
                        let maxprec = precision(emax, h.maxprec, h.minexp, dims);
                        i64c::decode_block(s, l.minbits, l.maxbits, maxprec, dims)?.0
                    };
                    if reversible && emax == -1023 {
                        vec![0.0; size]
                    } else {
                        let scale = pow2_f64(emax.saturating_sub(62));
                        #[allow(clippy::cast_precision_loss)]
                        iblock.iter().map(|x| scale * (*x as f64)).collect()
                    }
                }
            })
        }
    })
}

/// Decode a ZFP stream (no header) of the field `h` describes into its values' bytes
/// (native-endian, `x` fastest), at most `limit` bytes.
///
/// # Errors
/// [`Error::Corrupt`] for a truncated or malformed stream; [`Error::TooLarge`] past `limit`.
pub fn decompress(stream: &[u8], h: &ZfpHeader, limit: usize) -> Result<Vec<u8>, Error> {
    let nd = h.dims.len();
    if nd == 0 || nd > 4 || h.dims.contains(&0) {
        return Err(err());
    }
    let n = h
        .dims
        .iter()
        .try_fold(1usize, |a, d| a.checked_mul(*d))
        .ok_or_else(err)?;
    let esize = h.ztype.size();
    let total_bytes = n
        .checked_mul(esize)
        .filter(|b| *b <= limit)
        .ok_or_else(|| Error::TooLarge(CODEC.into()))?;
    let mut out = vec![0u8; total_bytes];
    let mut s = Bits {
        data: stream,
        pos: 0,
    };
    let blocks: Vec<usize> = h.dims.iter().map(|d| d.div_ceil(4)).collect();
    let nblocks = blocks
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(err)?;
    // Element strides (x fastest).
    let mut strides = Vec::with_capacity(nd);
    let mut stride = 1usize;
    for d in &h.dims {
        strides.push(stride);
        stride = stride.saturating_mul(*d);
    }
    let size = block_size(nd);
    for b in 0..nblocks {
        // The block's first element, x fastest.
        let mut rem = b;
        let mut origin = Vec::with_capacity(nd);
        for nb in &blocks {
            origin.push(rem.checked_rem(*nb).unwrap_or(0).saturating_mul(4));
            rem = rem.checked_div(*nb).unwrap_or(0);
        }
        let block = decode_block(&mut s, h, nd)?;
        for i in 0..size {
            // The value's position in the field (partial blocks at the edges).
            let mut flat = 0usize;
            let mut inside = true;
            for (k, ((o, d), st)) in origin.iter().zip(&h.dims).zip(&strides).enumerate() {
                let c = i
                    .checked_shr(u32::try_from(k.saturating_mul(2)).unwrap_or(0))
                    .unwrap_or(0)
                    & 3;
                let g = o.saturating_add(c);
                if g >= *d {
                    inside = false;
                    break;
                }
                flat = flat.saturating_add(g.saturating_mul(*st));
            }
            if !inside {
                continue;
            }
            let at = flat.saturating_mul(esize);
            let dst = out.get_mut(at..at.saturating_add(esize)).ok_or_else(err)?;
            block.put(i, dst)?;
        }
    }
    Ok(out)
}
