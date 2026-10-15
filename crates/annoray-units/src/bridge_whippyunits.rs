//! [`StaticUnit`] for `whippyunits` quantities (feature `whippyunits`).
//!
//! whippyunits is the model of this crate's runtime units: the same eight dimensions (SI plus
//! angle) and the same `2^a · 3^b · 5^c · π^d` scales, so units map one to one, values stay in
//! their declared unit, and arrays whose unit matches exactly are viewed without copying
//! (`Quantity` is `repr(transparent)`). Only the default brand `()` is bridged: brands have
//! no runtime counterpart.

use whippyunits::Quantity;
use whippyunits::quantity::{_2, _3, _5, _A, _I, _J, _L, _M, _N, _Pi, _T, _Θ};

use crate::static_unit::{StaticUnit, StaticValue};
use crate::{Dimension, Scale, Unit, UnitError};

fn exponent(e: i16) -> Result<i8, UnitError> {
    i8::try_from(e).map_err(|_| UnitError::Overflow)
}

impl<
    const M: i16,
    const L: i16,
    const T: i16,
    const I: i16,
    const TH: i16,
    const N: i16,
    const J: i16,
    const A: i16,
    const P2: i16,
    const P3: i16,
    const P5: i16,
    const PI: i16,
    V: StaticValue,
> StaticUnit
    for Quantity<
        whippyunits::Unit<
            whippyunits::quantity::Scale<_2<P2>, _3<P3>, _5<P5>, _Pi<PI>>,
            whippyunits::quantity::Dimension<
                _M<M>,
                _L<L>,
                _T<T>,
                _I<I>,
                _Θ<TH>,
                _N<N>,
                _J<J>,
                _A<A>,
            >,
        >,
        V,
        (),
    >
{
    type Value = V;

    fn unit() -> Result<Unit, UnitError> {
        let dim = Dimension::from_exponents([
            exponent(M)?,
            exponent(L)?,
            exponent(T)?,
            exponent(I)?,
            exponent(TH)?,
            exponent(N)?,
            exponent(J)?,
            exponent(A)?,
        ]);
        Ok(Unit::linear(dim, Scale::from_exponents(P2, P3, P5, PI)))
    }

    fn from_raw(v: V) -> Self {
        Self::new(v)
    }

    fn into_raw(self) -> V {
        self.unsafe_value
    }

    fn view_slice(values: &[V]) -> Option<&[Self]> {
        Some(Self::from_slice(values))
    }
}
