//! Exact scale factors: `2^p2 · 3^p3 · 5^p5 · π^pπ · residual`.
//!
//! Following whippyunits, scales are kept prime-factorised so that SI prefixes, minutes, hours,
//! days and degrees are exact. Equal scales are equal values (`==`, `Hash`), with no tolerance.

use std::fmt;

use crate::UnitError;
use crate::ck::Ck;

/// An exact signed rational number in lowest terms with a positive denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    num: i128,
    den: i128,
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        let r = a.checked_rem(b).unwrap_or(0);
        a = b;
        b = r;
    }
    i128::try_from(a).unwrap_or(1)
}

impl Rational {
    /// Zero.
    pub const ZERO: Self = Self { num: 0, den: 1 };
    /// One.
    pub const ONE: Self = Self { num: 1, den: 1 };

    /// `num / den`, reduced.
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if `den` is zero or normalisation overflows.
    pub fn new(num: i128, den: i128) -> Result<Self, UnitError> {
        if den == 0 {
            return Err(UnitError::Overflow);
        }
        let g = gcd(num, den);
        let (mut num, mut den) = ((Ck::new(num) / g).get()?, (Ck::new(den) / g).get()?);
        if den < 0 {
            num = (-Ck::new(num)).get()?;
            den = (-Ck::new(den)).get()?;
        }
        Ok(Self { num, den })
    }

    /// An integer.
    #[must_use]
    pub const fn from_int(n: i128) -> Self {
        Self { num: n, den: 1 }
    }

    /// Numerator.
    #[must_use]
    pub const fn numer(&self) -> i128 {
        self.num
    }

    /// Denominator (always positive).
    #[must_use]
    pub const fn denom(&self) -> i128 {
        self.den
    }

    /// Whether this is zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// Nearest `f64`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // conversion to float is the point
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// `mantissa · 10^exp10`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if the result does not fit.
    pub fn from_decimal(mantissa: i128, exp10: i32) -> Result<Self, UnitError> {
        let p = Ck::new(10)
            .0
            .and_then(|t| t.checked_pow(exp10.unsigned_abs()));
        let p = Ck(p).get()?;
        if exp10 >= 0 {
            Self::new((Ck::new(mantissa) * p).get()?, 1)
        } else {
            Self::new(mantissa, p)
        }
    }

    /// `self + rhs`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`].
    pub fn checked_add(self, rhs: Self) -> Result<Self, UnitError> {
        let g = gcd(self.den, rhs.den);
        let l = (Ck::new(self.den) / g).get()?;
        let r = (Ck::new(rhs.den) / g).get()?;
        let num = Ck::new(self.num) * r + Ck::new(rhs.num) * l;
        Self::new(num.get()?, (Ck::new(self.den) * r).get()?)
    }

    /// `self - rhs`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`].
    pub fn checked_sub(self, rhs: Self) -> Result<Self, UnitError> {
        self.checked_add(Self {
            num: (-Ck::new(rhs.num)).get()?,
            den: rhs.den,
        })
    }

    /// `self * rhs`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`].
    pub fn checked_mul(self, rhs: Self) -> Result<Self, UnitError> {
        // Cross-reduce first to keep intermediates small.
        let g1 = gcd(self.num, rhs.den);
        let g2 = gcd(rhs.num, self.den);
        let num = (Ck::new(self.num) / g1) * (Ck::new(rhs.num) / g2);
        let den = (Ck::new(self.den) / g2) * (Ck::new(rhs.den) / g1);
        Self::new(num.get()?, den.get()?)
    }

    /// `1 / self`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if `self` is zero.
    pub fn recip(self) -> Result<Self, UnitError> {
        Self::new(self.den, self.num)
    }

    /// `self ^ n` for any integer `n`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`], including `0 ^ negative`.
    pub fn powi(self, n: i32) -> Result<Self, UnitError> {
        let e = n.unsigned_abs();
        let num = Ck(self.num.checked_pow(e)).get()?;
        let den = Ck(self.den.checked_pow(e)).get()?;
        if n >= 0 {
            Self::new(num, den)
        } else {
            Self::new(den, num)
        }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// A non-zero scale factor `2^p2 · 3^p3 · 5^p5 · π^pπ · residual`, with the residual coprime to 30.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scale {
    p2: i16,
    p3: i16,
    p5: i16,
    ppi: i16,
    residual: Rational,
}

/// Remove all factors of `p` from `n > 0`, returning the reduced value and the count.
fn strip(mut n: i128, p: i128) -> Result<(i128, i16), UnitError> {
    let mut count: i16 = 0;
    while n != 0 && n.checked_rem(p) == Some(0) {
        n = (Ck::new(n) / p).get()?;
        count = count.checked_add(1).ok_or(UnitError::Overflow)?;
    }
    Ok((n, count))
}

fn add16(a: i16, b: i16) -> Result<i16, UnitError> {
    a.checked_add(b).ok_or(UnitError::Overflow)
}

fn mul16(a: i16, n: i32) -> Result<i16, UnitError> {
    (Ck::new(a) * n).to()
}

fn pow10_f64(k: i32) -> f64 {
    // Parsing gives the correctly rounded value, unlike repeated multiplication.
    format!("1e{k}").parse().unwrap_or(f64::NAN)
}

/// Exact integer `n`-th root of `v >= 0`, if there is one.
fn int_root(v: i128, n: u32) -> Option<i128> {
    if v < 2 || n == 1 {
        return Some(v);
    }
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let guess = (v as f64).powf(1.0 / f64::from(n)).round() as i128;
    [guess.checked_sub(1)?, guess, guess.checked_add(1)?]
        .into_iter()
        .find(|c| *c >= 0 && c.checked_pow(n) == Some(v))
}

impl Scale {
    /// The scale `1`.
    pub const ONE: Self = Self {
        p2: 0,
        p3: 0,
        p5: 0,
        ppi: 0,
        residual: Rational::ONE,
    };

    /// `2^p2 · 3^p3 · 5^p5 · π^ppi`.
    #[must_use]
    pub const fn from_exponents(p2: i16, p3: i16, p5: i16, ppi: i16) -> Self {
        Self {
            p2,
            p3,
            p5,
            ppi,
            residual: Rational::ONE,
        }
    }

    /// The scale `π`.
    #[must_use]
    pub const fn pi() -> Self {
        Self {
            p2: 0,
            p3: 0,
            p5: 0,
            ppi: 1,
            residual: Rational::ONE,
        }
    }

    /// A scale from a non-zero rational. Negative scales exist in UDUNITS
    /// (`degree_west = -1 degree_east`); the sign is kept in the residual.
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if `r == 0`.
    pub fn from_rational(r: Rational) -> Result<Self, UnitError> {
        if r.num == 0 {
            return Err(UnitError::Overflow);
        }
        let (n, n2) = strip(r.num, 2)?;
        let (n, n3) = strip(n, 3)?;
        let (n, n5) = strip(n, 5)?;
        let (d, d2) = strip(r.den, 2)?;
        let (d, d3) = strip(d, 3)?;
        let (d, d5) = strip(d, 5)?;
        Ok(Self {
            p2: (Ck::new(n2) - d2).to()?,
            p3: (Ck::new(n3) - d3).to()?,
            p5: (Ck::new(n5) - d5).to()?,
            ppi: 0,
            residual: Rational::new(n, d)?,
        })
    }

    /// `mantissa · 10^exp10`, with `mantissa != 0`. Exact for any exponent in `i16` range.
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if `mantissa == 0` or the exponent is out of range.
    pub fn from_decimal(mantissa: i128, exp10: i32) -> Result<Self, UnitError> {
        let base = Self::from_rational(Rational::from_int(mantissa))?;
        let e = i16::try_from(exp10).map_err(|_| UnitError::Overflow)?;
        Ok(Self {
            p2: add16(base.p2, e)?,
            p5: add16(base.p5, e)?,
            ..base
        })
    }

    /// Whether this is exactly one.
    #[must_use]
    pub fn is_one(&self) -> bool {
        *self == Self::ONE
    }

    /// Exponents of 2, 3, 5 and π.
    #[must_use]
    pub const fn exponents(&self) -> (i16, i16, i16, i16) {
        (self.p2, self.p3, self.p5, self.ppi)
    }

    /// The part not made of 2, 3, 5 or π (e.g. `127/50 → 127` for the inch, after factoring).
    #[must_use]
    pub const fn residual(&self) -> Rational {
        self.residual
    }

    /// `self · rhs`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`].
    pub fn checked_mul(self, rhs: Self) -> Result<Self, UnitError> {
        Ok(Self {
            p2: add16(self.p2, rhs.p2)?,
            p3: add16(self.p3, rhs.p3)?,
            p5: add16(self.p5, rhs.p5)?,
            ppi: add16(self.ppi, rhs.ppi)?,
            residual: self.residual.checked_mul(rhs.residual)?,
        })
    }

    /// `1 / self`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`].
    pub fn recip(self) -> Result<Self, UnitError> {
        self.powi(-1)
    }

    /// `self / rhs`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`].
    pub fn checked_div(self, rhs: Self) -> Result<Self, UnitError> {
        self.checked_mul(rhs.recip()?)
    }

    /// `self ^ n`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`].
    pub fn powi(self, n: i32) -> Result<Self, UnitError> {
        Ok(Self {
            p2: mul16(self.p2, n)?,
            p3: mul16(self.p3, n)?,
            p5: mul16(self.p5, n)?,
            ppi: mul16(self.ppi, n)?,
            residual: self.residual.powi(n)?,
        })
    }

    /// The exact `n`-th root, if every exponent is divisible by `n` and the residual is a
    /// perfect power. `n` must be positive.
    #[must_use]
    pub fn root(self, n: i32) -> Option<Self> {
        let div = |p: i16| -> Option<i16> {
            let n16 = i16::try_from(n).ok()?;
            if n16 <= 0 || p.checked_rem(n16)? != 0 {
                None
            } else {
                p.checked_div(n16)
            }
        };
        let un = n.unsigned_abs();
        if self.residual.num < 0 {
            return None;
        }
        Some(Self {
            p2: div(self.p2)?,
            p3: div(self.p3)?,
            p5: div(self.p5)?,
            ppi: div(self.ppi)?,
            residual: Rational::new(
                int_root(self.residual.num, un)?,
                int_root(self.residual.den, un)?,
            )
            .ok()?,
        })
    }

    /// The value as an exact rational, unless it involves π or overflows `i128`.
    #[must_use]
    pub fn to_rational(&self) -> Option<Rational> {
        if self.ppi != 0 {
            return None;
        }
        let pow = |b: i128, e: i16| Rational::from_int(b).powi(i32::from(e)).ok();
        pow(2, self.p2)?
            .checked_mul(pow(3, self.p3)?)
            .ok()?
            .checked_mul(pow(5, self.p5)?)
            .ok()?
            .checked_mul(self.residual)
            .ok()
    }

    /// Nearest `f64`.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        // Pair 2s with 5s into a power of ten, which parses exactly-rounded.
        let m = i32::from(self.p2.unsigned_abs().min(self.p5.unsigned_abs()));
        let k = match (self.p2.signum(), self.p5.signum()) {
            (1, 1) => m,
            (-1, -1) => 0i32.saturating_sub(m),
            _ => 0,
        };
        let rest2 = i32::from(self.p2).saturating_sub(k);
        let rest5 = i32::from(self.p5).saturating_sub(k);
        pow10_f64(k)
            * 2f64.powi(rest2)
            * 5f64.powi(rest5)
            * 3f64.powi(i32::from(self.p3))
            * std::f64::consts::PI.powi(i32::from(self.ppi))
            * self.residual.to_f64()
    }
}

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_rational() {
            Some(r) if r.den == 1 => write!(f, "{}", r.num),
            _ => write!(f, "{}", self.to_f64()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_scales_are_exact() {
        let milli = Scale::from_decimal(1, -3).unwrap();
        let kilo = Scale::from_decimal(1, 3).unwrap();
        assert!(milli.checked_mul(kilo).unwrap().is_one());
        // 3600 s = 2^4 3^2 5^2
        let hour = Scale::from_rational(Rational::from_int(3600)).unwrap();
        assert_eq!(hour.exponents(), (4, 2, 2, 0));
        assert_eq!(hour.residual(), Rational::ONE);
        // inch = 0.0254 m = 254e-4 = 127 · 2 · 10^-4
        let inch = Scale::from_decimal(254, -4).unwrap();
        assert_eq!(inch.residual(), Rational::from_int(127));
        assert_eq!(inch.to_rational(), Some(Rational::new(127, 5000).unwrap()));
        assert!((inch.to_f64() - 0.0254).abs() < 1e-17);
        assert_eq!(Scale::from_decimal(1, -3).unwrap().to_f64(), 0.001);
        assert_eq!(Scale::from_decimal(1, 24).unwrap().to_f64(), 1e24);
    }

    #[test]
    fn roots_and_powers() {
        let km = Scale::from_decimal(1, 3).unwrap();
        let km2 = km.powi(2).unwrap();
        assert_eq!(km2.root(2), Some(km));
        assert_eq!(km.root(2), None);
        let nine = Scale::from_rational(Rational::new(49, 9).unwrap()).unwrap();
        assert_eq!(
            nine.root(2).unwrap().to_rational(),
            Some(Rational::new(7, 3).unwrap())
        );
        assert!(Scale::pi().to_rational().is_none());
        assert!(Scale::from_rational(Rational::ZERO).is_err());
    }

    #[test]
    fn rational_ops() {
        let a = Rational::new(27315, 100).unwrap();
        assert_eq!(a, Rational::new(5463, 20).unwrap());
        assert_eq!(a.checked_sub(a).unwrap(), Rational::ZERO);
        assert_eq!(Rational::new(1, -2).unwrap(), Rational::new(-1, 2).unwrap());
        assert!(Rational::new(1, 0).is_err());
        assert!(
            Rational::from_int(i128::MAX)
                .checked_add(Rational::ONE)
                .is_err()
        );
        assert_eq!(
            Rational::from_decimal(1234, -2).unwrap(),
            Rational::new(617, 50).unwrap()
        );
    }
}
