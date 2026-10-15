//! The bridge to compile-time unit libraries: [`StaticUnit`], implemented behind the `uom` and
//! `whippyunits` features, and conversions between such quantities and [`Quantity`].

use crate::{Converter, Dyn, Kind, Quantity, Unit, UnitError};

/// A numeric type a [`StaticUnit`] can store: values cross the bridge as `f64`.
pub trait StaticValue: Copy + 'static {
    /// The value as `f64`.
    fn to_f64(self) -> f64;

    /// The nearest value to `x`.
    fn from_f64(x: f64) -> Self;
}

impl StaticValue for f64 {
    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(x: f64) -> Self {
        x
    }
}

impl StaticValue for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(x: f64) -> Self {
        x as Self
    }
}

/// A quantity type whose unit is fixed at compile time (`uom` and `whippyunits` quantities,
/// behind the features of the same names).
///
/// It lets runtime-checked values (a [`Quantity`], an annoray array) be converted to and from
/// compile-time-checked ones, and, when the unit and layout match exactly, viewed in place.
pub trait StaticUnit: Sized {
    /// The underlying numeric type.
    type Value: StaticValue;

    /// The unit values of this type are expressed in, in the runtime unit system.
    ///
    /// # Errors
    /// [`UnitError::Unsupported`] for units the runtime system cannot represent (uom's
    /// information kind); [`UnitError::Overflow`] for exponents beyond its range.
    fn unit() -> Result<Unit, UnitError>;

    /// Wrap a raw value that is already expressed in [`unit()`](Self::unit).
    fn from_raw(v: Self::Value) -> Self;

    /// The raw value, expressed in [`unit()`](Self::unit).
    fn into_raw(self) -> Self::Value;

    /// Whether the library treats angle as dimensionless (uom): conversions then ignore the
    /// angle dimension on both sides, after scaling degrees and the like to radians.
    const ANGLE_DIMENSIONLESS: bool = false;

    /// Whether values are differences (uom's `TemperatureInterval`): an absolute source unit
    /// (°C) then converts by its step size, without its offset.
    fn is_difference() -> bool {
        false
    }

    /// The raw values viewed as this type, without copying, when the layouts are identical
    /// (whippyunits' `Quantity` is `repr(transparent)`); `None` otherwise.
    fn view_slice(values: &[Self::Value]) -> Option<&[Self]> {
        let _ = values;
        None
    }
}

/// The converter taking values in `from` to values of `Q`.
///
/// # Errors
/// [`UnitError::Incompatible`] if `from` measures something else; errors of
/// [`StaticUnit::unit`].
pub fn converter_into<Q: StaticUnit>(from: &Unit) -> Result<Converter, UnitError> {
    let target = Q::unit()?;
    let from = if Q::is_difference() {
        from.delta()
    } else {
        from.clone()
    };
    if Q::ANGLE_DIMENSIONLESS {
        from.erase_angle().converter_to(&target.erase_angle())
    } else {
        from.converter_to(&target)
    }
}

/// The converter taking values of `Q` to values in `to`.
///
/// # Errors
/// As for [`converter_into`].
pub fn converter_from<Q: StaticUnit>(to: &Unit) -> Result<Converter, UnitError> {
    let source = Q::unit()?;
    let to = if Q::is_difference() {
        to.delta()
    } else {
        to.clone()
    };
    if Q::ANGLE_DIMENSIONLESS {
        source.erase_angle().converter_to(&to.erase_angle())
    } else {
        source.converter_to(&to)
    }
}

impl Quantity<Dyn> {
    /// A compile-time quantity (`uom`, `whippyunits`) as a runtime one, in its unit.
    ///
    /// # Errors
    /// Errors of [`StaticUnit::unit`].
    pub fn from_static<Q: StaticUnit>(q: Q) -> Result<Self, UnitError> {
        Self::new(q.into_raw().to_f64(), Q::unit()?)
    }
}

impl<K: Kind> Quantity<K> {
    /// This quantity as a compile-time one (`uom`, `whippyunits`), converted into its unit.
    ///
    /// # Errors
    /// [`UnitError::Incompatible`] if `Q` measures something else; errors of
    /// [`StaticUnit::unit`].
    pub fn to_static<Q: StaticUnit>(&self) -> Result<Q, UnitError> {
        let c = converter_into::<Q>(self.unit())?;
        Ok(Q::from_raw(Q::Value::from_f64(c.apply(self.value()))))
    }
}
