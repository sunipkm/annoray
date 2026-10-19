//! Lossy quantization of floats before compression, as netCDF-C 4.9 does it
//! (`nc_def_var_quantize`), and netCDF4-python's `least_significant_digit`.
//!
//! Quantization keeps the significant digits (or bits) asked for and sets the rest of each
//! mantissa to a pattern that compresses well; it is recorded in an attribute that names the
//! algorithm (`_QuantizeBitGroomNumberOfSignificantDigits`, …), which readers keep as
//! information only: the stored values are the quantized ones.
//!
//! - **BitGroom** keeps `ceil(nsd · log₂10) + 1` mantissa bits and alternately clears
//!   (even positions) and sets (odd positions) the others, so the errors average out.
//! - **GranularBitRound** works out, for each value, how many bits its `nsd` decimal digits
//!   need, and rounds to them (round half up on the bit pattern).
//! - **BitRound** rounds every value to `nsb` mantissa bits.
//!
//! The results are netCDF-C's bit for bit (checked against netCDF-C 4.9.3), with these
//! deliberate differences:
//!
//! - Where netCDF-C computes `a·b + c` in floating point, a fused multiply-add is used:
//!   netCDF-C's own result at the rare values where this matters depends on whether its
//!   compiler fused the operation (Apple's clang does); annoray's is the same everywhere.
//! - A precision at or beyond what the type holds (7 digits for `f32` in BitGroom) leaves the
//!   values as they are; netCDF-C shifts by a negative amount there (undefined behaviour in C)
//!   and stores arbitrary bits.
//! - NaN, ±∞, ±0 and the fill value are left as they are (netCDF-C's BitGroom sets mantissa
//!   bits of ±∞, making them NaN, and turns −0 into a tiny negative number).
//!
//! `least_significant_digit = d` is netCDF4-python's: values are rounded (half to even) to a
//! multiple of `2⁻ᵇ` with `b = ceil(log₂ 10^d)`, the power of two just finer than `10⁻ᵈ`.

/// A netCDF-C quantization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuantizeMode {
    /// Alternately shave and set the bits beyond the significant digits.
    BitGroom,
    /// Round each value to the bits its significant decimal digits need.
    GranularBitRound,
    /// Round to a number of significant bits.
    BitRound,
}

/// A quantization: the algorithm and its precision (decimal digits; bits for
/// [`QuantizeMode::BitRound`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantize {
    /// The algorithm.
    pub mode: QuantizeMode,
    /// Significant decimal digits (BitGroom, GranularBitRound) or bits (BitRound).
    pub nsd: u8,
}

impl Quantize {
    /// The attribute netCDF-C records the quantization in.
    #[must_use]
    pub fn attribute(&self) -> &'static str {
        match self.mode {
            QuantizeMode::BitGroom => "_QuantizeBitGroomNumberOfSignificantDigits",
            QuantizeMode::GranularBitRound => "_QuantizeGranularBitRoundNumberOfSignificantDigits",
            QuantizeMode::BitRound => "_QuantizeBitRoundNumberOfSignificantBits",
        }
    }

    /// The quantization an attribute records, if `name` is one of netCDF-C's.
    #[must_use]
    pub fn from_attribute(name: &str, nsd: u8) -> Option<Self> {
        let mode = match name {
            "_QuantizeBitGroomNumberOfSignificantDigits" => QuantizeMode::BitGroom,
            "_QuantizeGranularBitRoundNumberOfSignificantDigits" => QuantizeMode::GranularBitRound,
            "_QuantizeBitRoundNumberOfSignificantBits" => QuantizeMode::BitRound,
            _ => return None,
        };
        Some(Self { mode, nsd })
    }

    /// The precision netCDF-C accepts for a type with `mantissa` explicit bits (23 or 52):
    /// 1–7 or 1–15 decimal digits, 1–23 or 1–52 bits.
    #[must_use]
    pub fn valid_for(&self, mantissa: u32) -> bool {
        let max = match self.mode {
            QuantizeMode::BitRound => mantissa,
            _ if mantissa == 23 => 7,
            _ => 15,
        };
        (1..=max).contains(&u32::from(self.nsd))
    }
}

/// netCDF-C's constants (`M_LN10 / M_LN2`, `M_LN2 / M_LN10`): they round differently from
/// `log2(10)` and `log10(2)`, which matters where a digit count falls on an integer.
const BIT_PER_DGT: f64 = std::f64::consts::LN_10 / std::f64::consts::LN_2;
const DGT_PER_BIT: f64 = std::f64::consts::LN_2 / std::f64::consts::LN_10;

/// Mask arithmetic for one float width.
pub(crate) trait Bits: Copy {
    const MANT: u32;
    fn bits(self) -> u64;
    fn from_bits64(b: u64) -> Self;
    fn value(self) -> f64;
}

impl Bits for f32 {
    const MANT: u32 = 23;
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    #[allow(clippy::cast_possible_truncation)] // only the low 32 bits are ever set
    fn from_bits64(b: u64) -> Self {
        Self::from_bits(b as u32)
    }
    fn value(self) -> f64 {
        f64::from(self)
    }
}

impl Bits for f64 {
    const MANT: u32 = 52;
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn from_bits64(b: u64) -> Self {
        Self::from_bits(b)
    }
    fn value(self) -> f64 {
        self
    }
}

/// All ones in the width of `T`.
fn all_ones<T: Bits>() -> u64 {
    if T::MANT == 23 {
        u64::from(u32::MAX)
    } else {
        u64::MAX
    }
}

/// Round half up to `keep` mantissa bits (BitRound's and GranularBitRound's step).
fn round_to<T: Bits>(x: T, keep: i64) -> T {
    let zro = i64::from(T::MANT).saturating_sub(keep);
    if zro <= 0 || zro > i64::from(T::MANT) {
        return x;
    }
    let ones = all_ones::<T>();
    let mask_zro = (ones << zro) & ones;
    let mask_one = !mask_zro & ones;
    let half = mask_one & (mask_zro >> 1);
    T::from_bits64((x.bits().wrapping_add(half) & mask_zro) & ones)
}

fn skip<T: Bits>(x: T, fill: Option<f64>) -> bool {
    let v = x.value();
    !v.is_finite() || v == 0.0 || fill.is_some_and(|f| f == v)
}

/// Quantize `values` in place; `fill` values are left as they are. `odd(i)` says whether
/// value `i` sits at an odd position of the whole array (BitGroom alternates by position, and
/// a slab of a larger array starts wherever it starts).
pub(crate) fn quantize<T: Bits>(
    values: &mut [T],
    q: Quantize,
    fill: Option<f64>,
    odd: impl Fn(usize) -> bool,
) {
    let mant = i64::from(T::MANT);
    let nsd = i64::from(q.nsd);
    match q.mode {
        QuantizeMode::BitGroom => {
            #[allow(clippy::cast_possible_truncation)] // a few dozen bits
            let keep = ((nsd as f64 * BIT_PER_DGT).ceil() as i64).saturating_add(1);
            let zro = mant.saturating_sub(keep);
            if zro <= 0 {
                return;
            }
            let low = (1u64 << zro).wrapping_sub(1);
            for (i, x) in values.iter_mut().enumerate() {
                if skip(*x, fill) {
                    continue;
                }
                let b = if odd(i) {
                    x.bits() | low
                } else {
                    x.bits() & !low
                };
                *x = T::from_bits64(b & all_ones::<T>());
            }
        }
        QuantizeMode::BitRound => {
            for x in values.iter_mut().filter(|x| !skip(**x, fill)) {
                *x = round_to(*x, nsd);
            }
        }
        QuantizeMode::GranularBitRound => {
            for x in values.iter_mut() {
                if skip(*x, fill) {
                    continue;
                }
                let (m, e) = frexp(x.value());
                let m_log10 = m.abs().log10();
                let e = f64::from(e);
                #[allow(clippy::cast_possible_truncation)] // digit and bit counts are small
                let digits = (e.mul_add(DGT_PER_BIT, m_log10).floor() as i64).saturating_add(1);
                #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                let qnt_pwr = (BIT_PER_DGT * (digits.saturating_sub(nsd)) as f64).floor() as i64;
                #[allow(clippy::cast_possible_truncation)]
                let msb = (-BIT_PER_DGT).mul_add(m_log10, e).floor() as i64;
                let keep = msb.saturating_sub(qnt_pwr).abs().saturating_sub(1);
                *x = round_to(*x, keep);
            }
        }
    }
}

/// `x = m · 2^e` with `0.5 ≤ |m| < 1` (C's `frexp`), for finite non-zero `x`.
fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    #[allow(clippy::cast_possible_truncation)] // 11 bits
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        // Subnormal: scale into the normal range first.
        let (m, e) = frexp(x * f64::from_bits(0x4350_0000_0000_0000)); // 2^54
        return (m, e.saturating_sub(54));
    }
    let m = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (m, exp.saturating_sub(1022))
}

/// netCDF4-python's `least_significant_digit` rounding of `values`, in place.
pub(crate) fn least_significant_digit<T: Bits + RoundTo>(
    values: &mut [T],
    digits: i32,
    fill: Option<f64>,
) {
    let precision = 10f64.powi(digits.saturating_neg());
    let exp = precision.log10();
    #[allow(clippy::cast_possible_truncation)]
    let exp = if exp < 0.0 { exp.floor() } else { exp.ceil() } as i32;
    let bits = 10f64.powi(exp.saturating_neg()).log2().ceil();
    let scale = 2f64.powf(bits);
    for x in values.iter_mut() {
        let v = x.value();
        if !v.is_finite() || fill.is_some_and(|f| f == v) {
            continue;
        }
        *x = T::from_f64((scale * v).round_ties_even() / scale);
    }
}

/// Conversion back from `f64` for [`least_significant_digit`].
pub(crate) trait RoundTo {
    fn from_f64(v: f64) -> Self;
}

impl RoundTo for f32 {
    #[allow(clippy::cast_possible_truncation)] // the value came from an f32
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl RoundTo for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_matches_c() {
        assert_eq!(frexp(1.0), (0.5, 1));
        assert_eq!(frexp(-3.0), (-0.75, 2));
        assert_eq!(frexp(0.1).1, -3);
        let (m, e) = frexp(5e-324);
        assert_eq!((m, e), (0.5, -1073));
    }

    #[test]
    fn bitround_keeps_bits() {
        let mut v = [std::f32::consts::PI, -1.0e-3];
        quantize(
            &mut v,
            Quantize {
                mode: QuantizeMode::BitRound,
                nsd: 3,
            },
            None,
            |i| i % 2 == 1,
        );
        assert_eq!(v[0], 3.25); // 1.1001001… rounded to 1.101 × 2¹
        assert!((v[1] + 1.0e-3).abs() < 1.0e-4);
    }

    #[test]
    fn specials_and_fill_are_left_alone() {
        for mode in [
            QuantizeMode::BitGroom,
            QuantizeMode::GranularBitRound,
            QuantizeMode::BitRound,
        ] {
            let mut v = [f64::NAN, f64::INFINITY, -0.0, 0.0, -999.0, 1.234_567];
            quantize(&mut v, Quantize { mode, nsd: 2 }, Some(-999.0), |i| {
                i % 2 == 1
            });
            assert!(v[0].is_nan());
            assert_eq!(v[1], f64::INFINITY);
            assert_eq!(v[2].to_bits(), (-0.0f64).to_bits());
            assert_eq!(v[3].to_bits(), 0);
            assert_eq!(v[4], -999.0);
            assert!((v[5] - 1.234_567).abs() < 0.05);
        }
    }

    #[test]
    fn least_significant_digit_matches_netcdf4_python() {
        // netCDF4-python: _quantize(np.array([1.23456789]), 2) == 1.234375 (2⁻⁷ multiples).
        let mut v = [1.234_567_89f64, -0.001];
        least_significant_digit(&mut v, 2, None);
        assert_eq!(v[0], 1.234_375);
        assert_eq!(v[1], 0.0);
    }
}
