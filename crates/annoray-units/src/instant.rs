//! Instants on the UTC time line and signed durations, each a 64-bit count of nanoseconds:
//! the memory layout of numpy's `datetime64[ns]` and `timedelta64[ns]`.
//!
//! [`Instant`] counts nanoseconds since 1970-01-01T00:00:00Z, so its range is
//! 1677-09-21T00:12:43.145224193Z to 2262-04-11T23:47:16.854775807Z, as numpy's. Dates outside
//! that range are [`CfDate`]s in the `proleptic_gregorian` calendar. [`Duration`] spans about
//! ±292 years. In both, `i64::MIN` is the missing value ("not a time", `NaT`).
//!
//! Both types are `#[repr(transparent)]` over `i64`; that layout is part of their contract,
//! so slices of them may be reinterpreted as slices of `i64` (and handed to numpy) in place.

use std::fmt;
use std::str::FromStr;

use crate::UnitError;
use crate::time::{Calendar, CfDate, RefDate};

const NS_PER_MICRO: i64 = 1_000;
const NS_PER_MILLI: i64 = 1_000_000;
const NS_PER_SEC: i64 = 1_000_000_000;
const NS_PER_MIN: i64 = 60 * NS_PER_SEC;
const NS_PER_HOUR: i64 = 60 * NS_PER_MIN;
const NS_PER_DAY: i64 = 24 * NS_PER_HOUR;

/// A point on the UTC time line, in nanoseconds since 1970-01-01T00:00:00Z (numpy's
/// `datetime64[ns]`). [`Instant::NAT`] is the missing value; it orders before every instant.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Instant(i64);

/// A signed span of time in nanoseconds (numpy's `timedelta64[ns]`). [`Duration::NAT`] is the
/// missing value; it orders before every duration.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Duration(i64);

impl Instant {
    /// "Not a time": the missing value (numpy's `NaT`).
    pub const NAT: Self = Self(i64::MIN);
    /// 1970-01-01T00:00:00Z.
    pub const UNIX_EPOCH: Self = Self(0);
    /// The earliest instant, 1677-09-21T00:12:43.145224193Z.
    pub const MIN: Self = Self(i64::MIN + 1);
    /// The latest instant, 2262-04-11T23:47:16.854775807Z.
    pub const MAX: Self = Self(i64::MAX);

    /// The instant `ns` nanoseconds after the Unix epoch; `i64::MIN` gives [`NAT`](Self::NAT).
    #[must_use]
    pub const fn from_unix_nanos(ns: i64) -> Self {
        Self(ns)
    }

    /// Nanoseconds since the Unix epoch (`i64::MIN` for [`NAT`](Self::NAT)).
    #[must_use]
    pub const fn unix_nanos(self) -> i64 {
        self.0
    }

    /// Whether this is [`NAT`](Self::NAT).
    #[must_use]
    pub const fn is_nat(self) -> bool {
        self.0 == i64::MIN
    }

    /// Midnight UTC on a proleptic Gregorian date (astronomical year numbering).
    ///
    /// # Errors
    /// [`UnitError::InvalidDate`] if the date does not exist; [`UnitError::InstantRange`] if
    /// it is outside the range of an instant.
    pub fn from_ymd(year: i64, month: u8, day: u8) -> Result<Self, UnitError> {
        Self::try_from(CfDate::from_ymd(
            Calendar::ProlepticGregorian,
            year,
            month,
            day,
        )?)
    }

    /// A UTC date and time of day on the proleptic Gregorian calendar.
    ///
    /// # Errors
    /// As for [`from_ymd`](Self::from_ymd), and [`UnitError::InvalidDate`] for a time of day
    /// out of range.
    pub fn from_ymd_hms_nano(
        (year, month, day): (i64, u8, u8),
        (hour, minute, second): (u8, u8, u8),
        nanosecond: u32,
    ) -> Result<Self, UnitError> {
        Self::try_from(
            CfDate::from_ymd(Calendar::ProlepticGregorian, year, month, day)?
                .and_hms_nano(hour, minute, second, nanosecond)?,
        )
    }

    /// The date and time of day in UTC, as a `proleptic_gregorian` [`CfDate`].
    ///
    /// # Errors
    /// [`UnitError::InstantRange`] for [`NAT`](Self::NAT).
    pub fn to_utc(self) -> Result<CfDate, UnitError> {
        self.in_calendar(Calendar::ProlepticGregorian)
    }

    /// The date and time of day in UTC shifted by `offset_seconds` (the wall-clock time of a
    /// place that far ahead of UTC), as a `proleptic_gregorian` [`CfDate`].
    ///
    /// # Errors
    /// [`UnitError::InstantRange`] for [`NAT`](Self::NAT).
    pub fn to_local(self, offset_seconds: i32) -> Result<CfDate, UnitError> {
        if self.is_nat() {
            return Err(UnitError::InstantRange("NaT".to_owned()));
        }
        CfDate::from_unix_nanos(
            Calendar::ProlepticGregorian,
            i128::from(self.0)
                .checked_add(i128::from(offset_seconds).saturating_mul(i128::from(NS_PER_SEC)))
                .ok_or(UnitError::Overflow)?,
        )
    }

    /// This instant as a date in `calendar`. Real-world calendars keep the instant (so a
    /// `julian` date may differ in its fields); idealized calendars (`noleap`, `360_day`, …)
    /// keep the proleptic Gregorian fields, and fail if the date does not exist in them.
    ///
    /// # Errors
    /// [`UnitError::InvalidDate`]; [`UnitError::InstantRange`] for [`NAT`](Self::NAT).
    pub fn in_calendar(self, calendar: Calendar) -> Result<CfDate, UnitError> {
        if self.is_nat() {
            return Err(UnitError::InstantRange("NaT".to_owned()));
        }
        if calendar.is_real_world() {
            return CfDate::from_unix_nanos(calendar, i128::from(self.0));
        }
        let g = self.to_utc()?;
        CfDate::from_ymd(calendar, g.year(), g.month(), g.day())?.and_hms_nano(
            g.hour(),
            g.minute(),
            g.second(),
            g.nanosecond(),
        )
    }

    /// `self + d`, or `None` on overflow or if either is missing.
    #[must_use]
    pub fn checked_add(self, d: Duration) -> Option<Self> {
        if self.is_nat() || d.is_nat() {
            return None;
        }
        self.0.checked_add(d.0).map(Self).filter(|t| !t.is_nat())
    }

    /// `self - d`, or `None` on overflow or if either is missing.
    #[must_use]
    pub fn checked_sub(self, d: Duration) -> Option<Self> {
        self.checked_add(d.checked_neg()?)
    }

    /// The time from `earlier` to `self`, or `None` on overflow or if either is missing.
    #[must_use]
    pub fn duration_since(self, earlier: Self) -> Option<Duration> {
        if self.is_nat() || earlier.is_nat() {
            return None;
        }
        self.0
            .checked_sub(earlier.0)
            .map(Duration)
            .filter(|d| !d.is_nat())
    }
}

impl TryFrom<CfDate> for Instant {
    type Error = UnitError;

    /// Real-world calendars convert exactly (the same instant); idealized calendars convert
    /// field by field and fail for dates the Gregorian calendar lacks (`360_day` Feb 30).
    /// `NaT` converts to [`Instant::NAT`].
    fn try_from(d: CfDate) -> Result<Self, UnitError> {
        if d.is_nat() {
            return Ok(Self::NAT);
        }
        let range = || UnitError::InstantRange(d.to_string());
        let ns = if d.calendar().is_real_world() {
            d.unix_nanos()?
        } else {
            CfDate::from_ymd(Calendar::ProlepticGregorian, d.year(), d.month(), d.day())
                .map_err(|_| range())?
                .and_hms_nano(d.hour(), d.minute(), d.second(), d.nanosecond())?
                .unix_nanos()?
        };
        i64::try_from(ns)
            .ok()
            .filter(|n| *n != i64::MIN)
            .map(Self)
            .ok_or_else(range)
    }
}

impl FromStr for Instant {
    type Err = UnitError;

    /// Parse `YYYY-MM-DD[( |T)hh[:mm[:ss[.fff]]]][ ][Z|UTC|GMT|±hh[:mm]]`; a time without an
    /// offset is in UTC, one with an offset is converted to UTC.
    fn from_str(s: &str) -> Result<Self, UnitError> {
        Self::try_from(
            s.parse::<RefDate>()?
                .in_calendar(Calendar::ProlepticGregorian)?,
        )
    }
}

/// `2000-01-02 03:04:05`, with a fraction of a second when there is one (3, 6 or 9 digits),
/// or `NaT`.
impl fmt::Display for Instant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nat() {
            return f.write_str("NaT");
        }
        let d = self.to_utc().map_err(|_| fmt::Error)?;
        write_civil(f, &d)
    }
}

/// `YYYY-MM-DD hh:mm:ss[.fff]` for a date's fields.
pub(crate) fn write_civil(f: &mut fmt::Formatter<'_>, d: &CfDate) -> fmt::Result {
    if d.year() < 0 {
        write!(f, "-{:04}", d.year().unsigned_abs())?;
    } else {
        write!(f, "{:04}", d.year())?;
    }
    write!(
        f,
        "-{:02}-{:02} {:02}:{:02}:{:02}",
        d.month(),
        d.day(),
        d.hour(),
        d.minute(),
        d.second()
    )?;
    write_fraction(f, d.nanosecond())
}

/// `.fff`, `.ffffff` or `.fffffffff`: the fewest groups of three digits that hold `ns`.
fn write_fraction(f: &mut fmt::Formatter<'_>, ns: u32) -> fmt::Result {
    if ns == 0 {
        Ok(())
    } else if ns.is_multiple_of(1_000_000) {
        write!(f, ".{:03}", ns / 1_000_000)
    } else if ns.is_multiple_of(1_000) {
        write!(f, ".{:06}", ns / 1_000)
    } else {
        write!(f, ".{ns:09}")
    }
}

impl Duration {
    /// "Not a time": the missing value (numpy's `NaT`).
    pub const NAT: Self = Self(i64::MIN);
    /// No time at all.
    pub const ZERO: Self = Self(0);
    /// The shortest duration, about −292 years.
    pub const MIN: Self = Self(i64::MIN + 1);
    /// The longest duration, about 292 years.
    pub const MAX: Self = Self(i64::MAX);

    /// `ns` nanoseconds; `i64::MIN` gives [`NAT`](Self::NAT).
    #[must_use]
    pub const fn from_nanos(ns: i64) -> Self {
        Self(ns)
    }

    /// The length in nanoseconds (`i64::MIN` for [`NAT`](Self::NAT)).
    #[must_use]
    pub const fn nanos(self) -> i64 {
        self.0
    }

    /// Whether this is [`NAT`](Self::NAT).
    #[must_use]
    pub const fn is_nat(self) -> bool {
        self.0 == i64::MIN
    }

    fn scaled(n: i64, ns: i64) -> Option<Self> {
        n.checked_mul(ns).map(Self).filter(|d| !d.is_nat())
    }

    /// `n` microseconds, or `None` if out of range.
    #[must_use]
    pub fn from_micros(n: i64) -> Option<Self> {
        Self::scaled(n, NS_PER_MICRO)
    }

    /// `n` milliseconds, or `None` if out of range.
    #[must_use]
    pub fn from_millis(n: i64) -> Option<Self> {
        Self::scaled(n, NS_PER_MILLI)
    }

    /// `n` seconds, or `None` if out of range.
    #[must_use]
    pub fn from_secs(n: i64) -> Option<Self> {
        Self::scaled(n, NS_PER_SEC)
    }

    /// `n` minutes, or `None` if out of range.
    #[must_use]
    pub fn from_minutes(n: i64) -> Option<Self> {
        Self::scaled(n, NS_PER_MIN)
    }

    /// `n` hours, or `None` if out of range.
    #[must_use]
    pub fn from_hours(n: i64) -> Option<Self> {
        Self::scaled(n, NS_PER_HOUR)
    }

    /// `n` days of 86 400 s, or `None` if out of range.
    #[must_use]
    pub fn from_days(n: i64) -> Option<Self> {
        Self::scaled(n, NS_PER_DAY)
    }

    /// `self + d`, or `None` on overflow or if either is missing.
    #[must_use]
    pub fn checked_add(self, d: Self) -> Option<Self> {
        if self.is_nat() || d.is_nat() {
            return None;
        }
        self.0.checked_add(d.0).map(Self).filter(|d| !d.is_nat())
    }

    /// `self - d`, or `None` on overflow or if either is missing.
    #[must_use]
    pub fn checked_sub(self, d: Self) -> Option<Self> {
        self.checked_add(d.checked_neg()?)
    }

    /// `-self`, or `None` if missing.
    #[must_use]
    pub fn checked_neg(self) -> Option<Self> {
        // `NAT` is `i64::MIN`, the one value without a negation.
        self.0.checked_neg().map(Self)
    }

    /// `self × n`, or `None` on overflow or if missing.
    #[must_use]
    pub fn checked_mul(self, n: i64) -> Option<Self> {
        if self.is_nat() {
            return None;
        }
        self.0.checked_mul(n).map(Self).filter(|d| !d.is_nat())
    }

    /// The length in seconds, as a float (`NaN` for [`NAT`](Self::NAT)).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_secs_f64(self) -> f64 {
        if self.is_nat() {
            f64::NAN
        } else {
            self.0 as f64 / 1e9
        }
    }
}

/// `1 days 02:03:04`, pandas' spelling, with a fraction of a second when there is one (3, 6 or
/// 9 digits); a negative duration has a leading `-` (`-0 days 01:00:00` is minus one hour), and
/// the missing value is `NaT`.
impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nat() {
            return f.write_str("NaT");
        }
        let a = self.0.unsigned_abs();
        let ns = |n: i64| n.unsigned_abs();
        let days = a.checked_div(ns(NS_PER_DAY)).ok_or(fmt::Error)?;
        let rest = a.checked_rem(ns(NS_PER_DAY)).ok_or(fmt::Error)?;
        let (h, rest) = (
            rest.checked_div(ns(NS_PER_HOUR)).ok_or(fmt::Error)?,
            rest.checked_rem(ns(NS_PER_HOUR)).ok_or(fmt::Error)?,
        );
        let (m, rest) = (
            rest.checked_div(ns(NS_PER_MIN)).ok_or(fmt::Error)?,
            rest.checked_rem(ns(NS_PER_MIN)).ok_or(fmt::Error)?,
        );
        let (s, sub) = (
            rest.checked_div(ns(NS_PER_SEC)).ok_or(fmt::Error)?,
            rest.checked_rem(ns(NS_PER_SEC)).ok_or(fmt::Error)?,
        );
        if self.0 < 0 {
            f.write_str("-")?;
        }
        write!(f, "{days} days {h:02}:{m:02}:{s:02}")?;
        write_fraction(f, u32::try_from(sub).map_err(|_| fmt::Error)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instants() {
        let t: Instant = "2000-01-02T03:04:05.5".parse().unwrap();
        assert_eq!(t.to_string(), "2000-01-02 03:04:05.500");
        assert_eq!(
            t,
            Instant::from_ymd_hms_nano((2000, 1, 2), (3, 4, 5), 500_000_000).unwrap()
        );
        assert_eq!(Instant::UNIX_EPOCH.to_string(), "1970-01-01 00:00:00");
        assert_eq!(Instant::NAT.to_string(), "NaT");
        assert_eq!(Instant::MIN.to_string(), "1677-09-21 00:12:43.145224193");
        assert_eq!(Instant::MAX.to_string(), "2262-04-11 23:47:16.854775807");
        // Offsets convert to UTC.
        let ny: Instant = "2024-05-11 03:00 -04:00".parse().unwrap();
        assert_eq!(ny.to_string(), "2024-05-11 07:00:00");
        assert_eq!(ny.to_local(-4 * 3600).unwrap().hour(), 3);
        assert!(matches!(
            Instant::from_ymd(1600, 1, 1),
            Err(UnitError::InstantRange(_))
        ));
        assert!(Instant::from_ymd(2001, 2, 29).is_err());
        assert!(Instant::NAT < Instant::MIN);
    }

    #[test]
    fn arithmetic() {
        let t = Instant::from_ymd(2000, 1, 1).unwrap();
        let day = Duration::from_days(1).unwrap();
        let u = t.checked_add(day).unwrap();
        assert_eq!(u.to_string(), "2000-01-02 00:00:00");
        assert_eq!(u.duration_since(t), Some(day));
        assert_eq!(u.checked_sub(day), Some(t));
        assert_eq!(Instant::NAT.checked_add(day), None);
        assert_eq!(t.checked_add(Duration::NAT), None);
        assert_eq!(Instant::MAX.checked_add(day), None);
        assert_eq!(Instant::MIN.duration_since(Instant::MAX), None);
        assert_eq!(Duration::NAT.checked_neg(), None);
        assert_eq!(Duration::from_days(i64::MAX), None);
    }

    #[test]
    fn calendars() {
        let t = Instant::from_ymd_hms_nano((1700, 3, 1), (1, 2, 3), 4).unwrap();
        for cal in [
            Calendar::Standard,
            Calendar::ProlepticGregorian,
            Calendar::Julian,
        ] {
            let cf = t.in_calendar(cal).unwrap();
            assert_eq!(Instant::try_from(cf).unwrap(), t, "{cal}");
        }
        // Same instant, different fields: Gregorian 1700-03-01 is Julian 1700-02-19.
        let j = t.in_calendar(Calendar::Julian).unwrap();
        assert_eq!((j.month(), j.day()), (2, 19));
        // Idealized calendars keep the fields.
        let feb30 = CfDate::from_ymd(Calendar::Day360, 2001, 2, 30).unwrap();
        assert!(matches!(
            Instant::try_from(feb30),
            Err(UnitError::InstantRange(_))
        ));
        assert_eq!(
            feb30.to_instant_clamped().unwrap().to_string(),
            "2001-02-28 00:00:00"
        );
        let noleap = CfDate::from_ymd(Calendar::NoLeap, 2001, 3, 1).unwrap();
        assert_eq!(
            Instant::try_from(noleap).unwrap().to_string(),
            "2001-03-01 00:00:00"
        );
        assert!(
            Instant::from_ymd(2000, 2, 29)
                .unwrap()
                .in_calendar(Calendar::NoLeap)
                .is_err()
        );
    }

    #[test]
    fn durations() {
        let d = Duration::from_secs(86_400 + 2 * 3600 + 3 * 60 + 4).unwrap();
        assert_eq!(d.to_string(), "1 days 02:03:04");
        assert_eq!(
            Duration::from_millis(-1500).unwrap().to_string(),
            "-0 days 00:00:01.500"
        );
        assert_eq!(
            Duration::from_nanos(1).to_string(),
            "0 days 00:00:00.000000001"
        );
        assert_eq!(Duration::NAT.to_string(), "NaT");
        assert!((Duration::from_millis(1500).unwrap().as_secs_f64() - 1.5).abs() < 1e-12);
    }
}
