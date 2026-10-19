//! IEEE 754 half precision (`float16`): read as `f32` (every half is exactly an `f32`) and
//! written from `f32`/`f64` by rounding to nearest, ties to even, as numpy's
//! `astype(np.float16)` does (from `f64` directly, not through `f32`, which would round twice).
//! Too large values become ±∞; NaN stays NaN (quiet).

/// The `f32` a half's bits stand for (exact).
#[must_use]
pub fn to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: m · 2⁻²⁴, normalised for f32.
            let shift = m.leading_zeros().saturating_sub(21); // put the top bit at bit 10
            let m = (m << shift) & 0x3ff;
            let e = 127u32
                .saturating_sub(15)
                .saturating_sub(shift)
                .saturating_add(1);
            sign | (e << 23) | (m << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | (e.saturating_add(112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// `m >> shift`, rounded to nearest, ties to even.
fn round_shift(m: u64, shift: u32) -> u64 {
    if shift == 0 {
        return m;
    }
    if shift >= 64 {
        return 0;
    }
    let r = m >> shift;
    let rem = m & (1u64 << shift).wrapping_sub(1);
    let half = 1u64 << shift.saturating_sub(1);
    if rem > half || (rem == half && r & 1 == 1) {
        r.saturating_add(1)
    } else {
        r
    }
}

/// The half nearest to `x` (ties to even).
#[must_use]
pub fn from_f64(x: f64) -> u16 {
    let b = x.to_bits();
    #[allow(clippy::cast_possible_truncation)] // the sign bit only
    let sign = ((b >> 48) & 0x8000) as u16;
    #[allow(clippy::cast_possible_truncation)] // 11 bits
    let exp = ((b >> 52) & 0x7ff) as i32;
    let man = b & 0x000f_ffff_ffff_ffff;
    if exp == 0x7ff {
        return if man == 0 {
            sign | 0x7c00
        } else {
            #[allow(clippy::cast_possible_truncation)] // 10 bits
            let payload = ((man >> 42) & 0x3ff) as u16;
            sign | 0x7e00 | payload
        };
    }
    // The half's biased exponent.
    let e = exp.saturating_sub(1023 - 15);
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if exp == 0 || e < -10 {
        // f64 subnormals and anything below half the smallest half subnormal round to ±0.
        return sign;
    }
    if e <= 0 {
        // A half subnormal: the significand (with its implicit bit) shifted further.
        let m = man | (1u64 << 52);
        #[allow(clippy::cast_sign_loss)] // e ≤ 0
        let shift = 42u32.saturating_add(1i32.saturating_sub(e) as u32);
        #[allow(clippy::cast_possible_truncation)] // at most 0x400
        let r = round_shift(m, shift) as u16;
        return sign | r;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // 1 ≤ e ≤ 30; 11 bits
    let h = ((e as u64) << 10).saturating_add(round_shift(man, 42));
    // A carry out of the mantissa raises the exponent (up to ∞), as it should.
    #[allow(clippy::cast_possible_truncation)]
    let h = h.min(0x7c00) as u16;
    sign | h
}

/// The half nearest to `x` (ties to even), from `f64` so that it rounds once.
#[must_use]
pub fn from_f32(x: f32) -> u16 {
    from_f64(f64::from(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_round_trips() {
        for h in 0..=u16::MAX {
            let x = to_f32(h);
            if x.is_nan() {
                assert!(to_f32(from_f32(x)).is_nan(), "{h:#06x}");
                continue;
            }
            assert_eq!(from_f32(x), h, "{h:#06x} -> {x}");
        }
    }

    #[test]
    fn known_values() {
        assert_eq!(to_f32(0x3c00), 1.0);
        assert_eq!(to_f32(0xc000), -2.0);
        assert_eq!(to_f32(0x7bff), 65504.0);
        assert_eq!(to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(from_f64(65519.0), 0x7bff); // below the halfway point to 65536
        assert_eq!(from_f64(65520.0), 0x7c00); // ties to even: ∞
        assert_eq!(from_f64(1e10), 0x7c00);
        assert_eq!(from_f64(2f64.powi(-25)), 0); // a tie to even: 0
        assert_eq!(from_f64(2f64.powi(-25) * 1.000_001), 1);
        assert_eq!(from_f64(1.0 + 2f64.powi(-11)), 0x3c00); // tie: even mantissa
        assert_eq!(from_f64(1.0 + 3.0 * 2f64.powi(-11)), 0x3c02);
        assert_eq!(from_f64(-0.0), 0x8000);
    }
}
