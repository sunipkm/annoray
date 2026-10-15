//! Missing times ("not a time"), as in numpy: a reserved value of each time type.
//!
//! [`NaiveDateTime::MIN`] and [`TimeDelta::MIN`] are chrono's extremes and [`CfDate::NAT`]
//! reserves its year, so no real time is mistaken for a missing one. Reductions skip NaT like
//! NaN, arithmetic with NaT gives NaT, alignment fills missing labels with NaT, and the CF layer
//! reads and writes it the way xarray does.

use annoray_units::CfDate;
use chrono::{NaiveDateTime, TimeDelta};

/// Time types with a missing value.
pub trait NaT: Sized + Copy {
    /// The missing value.
    const NAT: Self;

    /// Whether this is the missing value.
    fn is_nat(&self) -> bool;
}

impl NaT for NaiveDateTime {
    const NAT: Self = Self::MIN;

    fn is_nat(&self) -> bool {
        *self == Self::MIN
    }
}

impl NaT for TimeDelta {
    const NAT: Self = Self::MIN;

    fn is_nat(&self) -> bool {
        *self == Self::MIN
    }
}

impl NaT for CfDate {
    const NAT: Self = Self::NAT;

    fn is_nat(&self) -> bool {
        Self::is_nat(self)
    }
}
