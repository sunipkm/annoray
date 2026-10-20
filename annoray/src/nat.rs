//! Missing times ("not a time"), as in numpy: a reserved value of each time type.
//!
//! [`Instant::NAT`] and [`Duration::NAT`] are `i64::MIN` nanoseconds, as numpy's, and
//! [`CfDate::NAT`] reserves its year, so no real time is mistaken for a missing one.
//! Reductions skip NaT like NaN, arithmetic with NaT gives NaT, alignment fills missing labels
//! with NaT, and the CF layer reads and writes it the way xarray does.

use annoray_units::{CfDate, Duration, Instant};

/// Time types with a missing value.
pub trait NaT: Sized + Copy {
    /// The missing value.
    const NAT: Self;

    /// Whether this is the missing value.
    fn is_nat(&self) -> bool;
}

impl NaT for Instant {
    const NAT: Self = Self::NAT;

    fn is_nat(&self) -> bool {
        Self::is_nat(*self)
    }
}

impl NaT for Duration {
    const NAT: Self = Self::NAT;

    fn is_nat(&self) -> bool {
        Self::is_nat(*self)
    }
}

impl NaT for CfDate {
    const NAT: Self = Self::NAT;

    fn is_nat(&self) -> bool {
        Self::is_nat(self)
    }
}
