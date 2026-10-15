//! Physical dimensions as integer exponents of eight base quantities.

use std::fmt;

use crate::UnitError;

/// A base quantity. Angle is first-class (as in whippyunits) rather than dimensionless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum BaseDim {
    Mass,
    Length,
    Time,
    Current,
    Temperature,
    Amount,
    Luminosity,
    Angle,
}

impl BaseDim {
    /// All base quantities, in canonical display order.
    pub const ALL: [Self; 8] = [
        Self::Mass,
        Self::Length,
        Self::Time,
        Self::Current,
        Self::Temperature,
        Self::Amount,
        Self::Luminosity,
        Self::Angle,
    ];

    /// The SI symbol of the coherent base unit.
    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Mass => "kg",
            Self::Length => "m",
            Self::Time => "s",
            Self::Current => "A",
            Self::Temperature => "K",
            Self::Amount => "mol",
            Self::Luminosity => "cd",
            Self::Angle => "rad",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Exponents of the base quantities: `kg^a m^b s^c …`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Dimension([i8; 8]);

impl Dimension {
    /// Dimensionless.
    pub const NONE: Self = Self([0; 8]);

    /// From the exponents of the base quantities, in [`BaseDim::ALL`] order (mass, length,
    /// time, current, temperature, amount, luminosity, angle).
    #[must_use]
    pub const fn from_exponents(exponents: [i8; 8]) -> Self {
        Self(exponents)
    }

    /// The exponents of the base quantities, in [`BaseDim::ALL`] order.
    #[must_use]
    pub const fn exponents(&self) -> [i8; 8] {
        self.0
    }

    /// A single base quantity to the first power.
    #[must_use]
    pub fn of(base: BaseDim) -> Self {
        let mut d = Self::NONE;
        if let Some(e) = d.0.get_mut(base.index()) {
            *e = 1;
        }
        d
    }

    /// The exponent of `base`.
    #[must_use]
    pub fn exponent(&self, base: BaseDim) -> i8 {
        self.0.get(base.index()).copied().unwrap_or(0)
    }

    /// Whether every exponent is zero.
    #[must_use]
    pub fn is_none(&self) -> bool {
        self.0.iter().all(|e| *e == 0)
    }

    /// The same dimension with the angle exponent removed.
    #[must_use]
    pub fn without_angle(mut self) -> Self {
        if let Some(e) = self.0.get_mut(BaseDim::Angle.index()) {
            *e = 0;
        }
        self
    }

    fn zip_with(self, rhs: Self, f: impl Fn(i8, i8) -> Option<i8>) -> Result<Self, UnitError> {
        let mut out = [0i8; 8];
        for ((o, a), b) in out.iter_mut().zip(self.0).zip(rhs.0) {
            *o = f(a, b).ok_or(UnitError::Overflow)?;
        }
        Ok(Self(out))
    }

    /// Exponents added (for multiplication of units).
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if an exponent leaves `i8`.
    pub fn checked_mul(self, rhs: Self) -> Result<Self, UnitError> {
        self.zip_with(rhs, i8::checked_add)
    }

    /// Exponents subtracted (for division of units).
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if an exponent leaves `i8`.
    pub fn checked_div(self, rhs: Self) -> Result<Self, UnitError> {
        self.zip_with(rhs, i8::checked_sub)
    }

    /// Exponents multiplied by `n`.
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if an exponent leaves `i8`.
    pub fn powi(self, n: i32) -> Result<Self, UnitError> {
        let n = i8::try_from(n).map_err(|_| UnitError::Overflow)?;
        self.zip_with(Self::NONE, |a, _| a.checked_mul(n))
    }

    /// Exponents divided by `n`, if all are divisible.
    #[must_use]
    pub fn root(self, n: i32) -> Option<Self> {
        let n = i8::try_from(n).ok().filter(|n| *n > 0)?;
        self.zip_with(Self::NONE, |a, _| {
            if a.checked_rem(n)? == 0 {
                a.checked_div(n)
            } else {
                None
            }
        })
        .ok()
    }
}

impl fmt::Display for Dimension {
    /// Canonical CF style: `kg m-2 s-1`; `1` when dimensionless.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (base, e) in BaseDim::ALL.iter().zip(self.0) {
            if e == 0 {
                continue;
            }
            if !first {
                f.write_str(" ")?;
            }
            first = false;
            f.write_str(base.symbol())?;
            if e != 1 {
                write!(f, "{e}")?;
            }
        }
        if first {
            f.write_str("1")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algebra_and_display() {
        let m = Dimension::of(BaseDim::Length);
        let s = Dimension::of(BaseDim::Time);
        let kg = Dimension::of(BaseDim::Mass);
        let flux = kg
            .checked_div(m.powi(2).unwrap())
            .unwrap()
            .checked_div(s)
            .unwrap();
        assert_eq!(flux.to_string(), "kg m-2 s-1");
        assert_eq!(Dimension::NONE.to_string(), "1");
        assert_eq!(m.powi(2).unwrap().root(2), Some(m));
        assert_eq!(m.root(2), None);
        assert!(m.powi(200).is_err());
        assert!(Dimension::of(BaseDim::Angle).without_angle().is_none());
    }
}
