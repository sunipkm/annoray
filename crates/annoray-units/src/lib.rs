//! Runtime units for annoray, modelled on whippyunits.
//!
//! - **Exact scales.** A [`Scale`] is `2^a · 3^b · 5^c · π^d · residual`, so prefixes, minutes,
//!   hours, days and degrees are exact and unit equality needs no tolerance.
//! - **Values stay in their declared unit.** Mixing scales is an error unless you rescale or opt
//!   into [`UnitPolicy::AutoRescale`].
//! - **Affine, epoch and logarithmic units** (°C, `days since 1970-01-01`, dB) are first-class.
//! - **Angle is a dimension**, erasable with [`Unit::erase_angle`].
//! - **UDUNITS-2 / CF parsing** backed by the full UDUNITS database, plus UCUM-style `kg.m2/s2`.
//! - **Instants and durations** as 64-bit nanosecond counts ([`Instant`], [`Duration`]), the
//!   layout of numpy's `datetime64[ns]` and `timedelta64[ns]`.
//! - **Time zones** ([`TimeZone`]): UTC, fixed offsets and IANA zones, from the system's time
//!   zone database or a bundled copy (feature `tzdb-bundle`).
//! - **All CF calendars** via [`CfDate`], with conversion to and from instants.
//!
//! ```
//! use annoray_units::{Unit, UnitPolicy};
//!
//! let km: Unit = "km".parse()?;
//! let m: Unit = "m".parse()?;
//! assert_eq!(km.converter_to(&m)?.factor(), 1000.0);
//! assert!(km.plan_add(&m, UnitPolicy::Strict).is_err());       // scales differ: rescale first
//! assert_eq!("m/s".parse::<Unit>()?, "m s-1".parse::<Unit>()?); // labels don't affect equality
//!
//! let c: Unit = "degC".parse()?;
//! let k: Unit = "K".parse()?;
//! assert_eq!(c.converter_to(&k)?.apply(20.0), 293.15);
//! # Ok::<(), annoray_units::UnitError>(())
//! ```

#![deny(missing_docs)]
// whippyunits names its temperature exponent `_Θ` (a crate-level lint).
#![cfg_attr(feature = "whippyunits", allow(mixed_script_confusables))]

// The README's examples run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

#[cfg(feature = "uom")]
mod bridge_uom;
#[cfg(feature = "whippyunits")]
mod bridge_whippyunits;
mod ck;
mod dimension;
pub mod dims;
mod error;
mod extra;
mod instant;
mod parse;
mod quantity;
mod registry;
mod scale;
mod static_unit;
mod table;
pub mod time;
mod unit;
mod zone;

use std::str::FromStr;

pub use dimension::{BaseDim, Dimension};
pub use error::UnitError;
pub use instant::{Duration, Instant};
pub use quantity::{Dyn, IntoUnit, Kind, Quantity, SameDimension, Scalar};
pub use scale::{Rational, Scale};
pub use static_unit::{StaticUnit, StaticValue, converter_from, converter_into};
pub use time::{Calendar, CfDate, EpochCodec, RefDate};
pub use unit::{BinaryPlan, Converter, LogBase, LogRef, Unit, UnitKind, UnitPolicy};
pub use zone::{Disambiguation, TimeZone};

impl FromStr for Unit {
    type Err = UnitError;

    /// Parse a UDUNITS-2 / CF unit string. The result is labelled with the trimmed input, so
    /// it displays (and is written back) exactly as given.
    fn from_str(s: &str) -> Result<Self, UnitError> {
        let reg = registry::registry();
        let unit = parse::parse_with(s, &|id| reg.lookup(id))?;
        let label = s.trim();
        Ok(if label.is_empty() {
            unit
        } else {
            unit.with_label(label)
        })
    }
}

impl Unit {
    /// Parse a unit string; same as [`str::parse`].
    ///
    /// # Errors
    /// [`UnitError::Parse`] or [`UnitError::UnknownUnit`].
    pub fn parse(s: &str) -> Result<Self, UnitError> {
        s.parse()
    }
}

#[cfg(test)]
mod tests;
