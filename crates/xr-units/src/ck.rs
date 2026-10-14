//! `Ck`: checked `i128` arithmetic with operator syntax.
//!
//! Every operation propagates overflow (and division by zero) as `None`, so arithmetic-heavy
//! code such as calendar algorithms reads naturally and still cannot panic. Call
//! [`Ck::get`] at the end to turn an overflow into an error.

use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::UnitError;

/// A checked integer. `None` means an earlier operation overflowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ck(pub(crate) Option<i128>);

impl Ck {
    pub(crate) fn new(v: impl Into<i128>) -> Self {
        Self(Some(v.into()))
    }

    /// The value, or [`UnitError::Overflow`].
    pub(crate) fn get(self) -> Result<i128, UnitError> {
        self.0.ok_or(UnitError::Overflow)
    }

    /// The value as `T`, or [`UnitError::Overflow`] if it does not fit.
    pub(crate) fn to<T: TryFrom<i128>>(self) -> Result<T, UnitError> {
        T::try_from(self.get()?).map_err(|_| UnitError::Overflow)
    }

    pub(crate) fn div_euclid(self, rhs: impl Into<Ck>) -> Self {
        let rhs = rhs.into();
        Self(self.0.zip(rhs.0).and_then(|(a, b)| a.checked_div_euclid(b)))
    }

    pub(crate) fn rem_euclid(self, rhs: impl Into<Ck>) -> Self {
        let rhs = rhs.into();
        Self(self.0.zip(rhs.0).and_then(|(a, b)| a.checked_rem_euclid(b)))
    }
}

macro_rules! from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for Ck {
            fn from(v: $t) -> Self {
                Self(Some(i128::from(v)))
            }
        }
    )*};
}
from_int!(i8, i16, i32, i64, i128, u8, u16, u32, u64);

macro_rules! binop {
    ($tr:ident, $m:ident, $checked:ident) => {
        impl<T: Into<Ck>> $tr<T> for Ck {
            type Output = Ck;
            fn $m(self, rhs: T) -> Ck {
                let rhs = rhs.into();
                Ck(self.0.zip(rhs.0).and_then(|(a, b)| a.$checked(b)))
            }
        }
    };
}
binop!(Add, add, checked_add);
binop!(Sub, sub, checked_sub);
binop!(Mul, mul, checked_mul);
binop!(Div, div, checked_div);
binop!(Rem, rem, checked_rem);

impl Neg for Ck {
    type Output = Ck;
    fn neg(self) -> Ck {
        Ck(self.0.and_then(i128::checked_neg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagates_overflow() {
        assert_eq!((Ck::new(2) * 3 + 1).get(), Ok(7));
        assert_eq!((Ck::new(i128::MAX) + 1 - 5).get(), Err(UnitError::Overflow));
        assert_eq!((Ck::new(1) / 0).get(), Err(UnitError::Overflow));
        assert_eq!(Ck::new(-7).div_euclid(2).get(), Ok(-4));
        assert_eq!(Ck::new(-7).rem_euclid(2).get(), Ok(1));
        assert_eq!(Ck::new(300).to::<u8>(), Err(UnitError::Overflow));
    }
}
