//! The bridge to compile-time unit libraries (`uom`, `whippyunits`), implemented in M2 behind
//! feature flags of the same names.

use crate::Unit;

/// A quantity type whose unit is fixed at compile time.
///
/// Implementations exist (behind features) for `uom` and `whippyunits` quantity types. They
/// let a runtime-checked array be viewed or converted as a compile-time-checked one.
pub trait StaticUnit: Sized {
    /// The underlying numeric type.
    type Value: Copy + 'static;

    /// The unit this type is expressed in, in the runtime unit system.
    fn unit() -> Unit;

    /// Wrap a raw value that is already expressed in [`unit()`](Self::unit).
    fn from_raw(v: Self::Value) -> Self;

    /// The raw value, expressed in [`unit()`](Self::unit).
    fn into_raw(self) -> Self::Value;
}
