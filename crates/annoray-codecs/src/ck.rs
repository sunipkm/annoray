//! `Ck`: checked integer arithmetic with operator syntax. Overflow (and division by zero)
//! propagates as `None`; nothing panics. Allow-listed in `clippy.toml` as `ck::Ck`.

use std::ops::{Add, Div, Mul, Rem, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ck(pub(crate) Option<u128>);

impl Ck {
    /// The value as `T`, or `None` on overflow.
    pub(crate) fn to<T: TryFrom<u128>>(self) -> Option<T> {
        self.0.and_then(|v| T::try_from(v).ok())
    }
}

impl From<usize> for Ck {
    fn from(v: usize) -> Self {
        Self(u128::try_from(v).ok())
    }
}

macro_rules! from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for Ck {
            fn from(v: $t) -> Self {
                Self(Some(u128::from(v)))
            }
        }
    )*};
}
from_int!(u8, u16, u32, u64, u128);

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

impl Ck {
    /// The value as `usize`, or `None` on overflow.
    pub(crate) fn usize(self) -> Option<usize> {
        self.to()
    }
}
