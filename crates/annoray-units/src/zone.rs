//! Time zones: UTC, fixed offsets from UTC, and IANA zones (`America/New_York`), which give
//! the wall-clock time of an [`Instant`] and the instant of a wall-clock time.
//!
//! IANA zones come from the system's time zone database (`/usr/share/zoneinfo`), or from a
//! copy built into the program with the `tzdb-bundle` feature (always on Windows).

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use crate::time::{Calendar, CfDate};
use crate::{Instant, UnitError};

/// A time zone: UTC, a fixed offset from UTC, or an IANA zone. Two zones are equal when their
/// names are.
#[derive(Clone)]
pub struct TimeZone {
    tz: jiff::tz::TimeZone,
    /// `UTC`, `+05:30`, or the IANA name.
    name: Arc<str>,
}

/// How a wall-clock time that a zone's clocks skip (a gap, as clocks go forward) or repeat (a
/// fold, as clocks go back) becomes an instant. The names and the default are those of
/// whenever and the JavaScript Temporal API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Disambiguation {
    /// The earlier of two repeated times; a skipped time moved forward by the gap's length.
    #[default]
    Compatible,
    /// The earlier of two repeated times; for a skipped time, the instant the offset before
    /// the gap gives.
    Earlier,
    /// The later of two repeated times; for a skipped time, the instant the offset after the
    /// gap gives.
    Later,
    /// An error ([`UnitError::LocalTime`]) for skipped and repeated times.
    Raise,
}

impl FromStr for Disambiguation {
    type Err = UnitError;

    /// `"compatible"`, `"earlier"`, `"later"` or `"raise"`.
    fn from_str(s: &str) -> Result<Self, UnitError> {
        Ok(match s {
            "compatible" => Self::Compatible,
            "earlier" => Self::Earlier,
            "later" => Self::Later,
            "raise" => Self::Raise,
            _ => {
                return Err(UnitError::Parse {
                    input: s.to_owned(),
                    pos: 0,
                    reason: "expected compatible, earlier, later or raise".to_owned(),
                });
            }
        })
    }
}

impl TimeZone {
    /// Coordinated Universal Time.
    #[must_use]
    pub fn utc() -> Self {
        Self {
            tz: jiff::tz::TimeZone::UTC,
            name: Arc::from("UTC"),
        }
    }

    /// A fixed offset from UTC, east positive (`+05:30` is 19 800 s), to the second.
    ///
    /// # Errors
    /// [`UnitError::UnknownTimeZone`] for an offset beyond ±25:59:59.
    pub fn fixed(offset_seconds: i32) -> Result<Self, UnitError> {
        let unknown = || UnitError::UnknownTimeZone(format!("{offset_seconds} s"));
        let offset = jiff::tz::Offset::from_seconds(offset_seconds).map_err(|_| unknown())?;
        let a = offset_seconds.unsigned_abs();
        let (h, m, s) = (a / 3600, a / 60 % 60, a % 60);
        let sign = if offset_seconds < 0 { '-' } else { '+' };
        let name = if s == 0 {
            format!("{sign}{h:02}:{m:02}")
        } else {
            format!("{sign}{h:02}:{m:02}:{s:02}")
        };
        Ok(Self {
            tz: jiff::tz::TimeZone::fixed(offset),
            name: Arc::from(name),
        })
    }

    /// An IANA zone by name (case-insensitive), from the time zone database.
    ///
    /// # Errors
    /// [`UnitError::UnknownTimeZone`] if the database has no such zone.
    pub fn iana(name: &str) -> Result<Self, UnitError> {
        let unknown = || UnitError::UnknownTimeZone(name.to_owned());
        let tz = jiff::tz::TimeZone::get(name).map_err(|_| unknown())?;
        let canonical = tz.iana_name().ok_or_else(unknown)?.to_owned();
        Ok(Self {
            tz,
            name: Arc::from(canonical),
        })
    }

    /// The name: `UTC`, an offset such as `+05:30`, or the IANA name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether this is UTC (not an offset of zero, which is a fixed zone of its own).
    #[must_use]
    pub fn is_utc(&self) -> bool {
        &*self.name == "UTC"
    }

    /// The offset from UTC at `t`, in seconds east (0 for NaT).
    #[must_use]
    pub fn offset_at(&self, t: Instant) -> i32 {
        if t.is_nat() {
            return 0;
        }
        jiff::Timestamp::from_nanosecond(i128::from(t.unix_nanos()))
            .map_or(0, |ts| self.tz.to_offset(ts).seconds())
    }

    /// The wall-clock date and time of `t` in this zone, as a `proleptic_gregorian`
    /// [`CfDate`].
    ///
    /// # Errors
    /// [`UnitError::InstantRange`] for NaT.
    pub fn to_local(&self, t: Instant) -> Result<CfDate, UnitError> {
        t.to_local(self.offset_at(t))
    }

    /// The instant at which clocks in this zone show `local` (a date in a real-world calendar,
    /// read as a proleptic Gregorian wall-clock time), resolving skipped and repeated times
    /// with `how`.
    ///
    /// # Errors
    /// [`UnitError::LocalTime`] for a skipped or repeated time with
    /// [`Disambiguation::Raise`]; [`UnitError::InstantRange`] outside the range of an instant.
    pub fn instant_of(&self, local: &CfDate, how: Disambiguation) -> Result<Instant, UnitError> {
        if local.is_nat() {
            return Ok(Instant::NAT);
        }
        let range = || UnitError::InstantRange(local.to_string());
        let g = if local.calendar() == Calendar::ProlepticGregorian {
            *local
        } else {
            Instant::try_from(*local)?.to_utc()?
        };
        let small = |v: u8| i8::try_from(v).map_err(|_| range());
        let dt = jiff::civil::DateTime::new(
            i16::try_from(g.year()).map_err(|_| range())?,
            small(g.month())?,
            small(g.day())?,
            small(g.hour())?,
            small(g.minute())?,
            small(g.second())?,
            i32::try_from(g.nanosecond()).map_err(|_| range())?,
        )
        .map_err(|_| range())?;
        let amb = self.tz.to_ambiguous_timestamp(dt);
        let ts = match how {
            Disambiguation::Compatible => amb.compatible(),
            Disambiguation::Earlier => amb.earlier(),
            Disambiguation::Later => amb.later(),
            Disambiguation::Raise => match amb.offset() {
                jiff::tz::AmbiguousOffset::Unambiguous { .. } => amb.compatible(),
                jiff::tz::AmbiguousOffset::Gap { .. } => {
                    return Err(self.local_error(&g, "skipped"));
                }
                jiff::tz::AmbiguousOffset::Fold { .. } => {
                    return Err(self.local_error(&g, "ambiguous"));
                }
            },
        }
        .map_err(|_| range())?;
        i64::try_from(ts.as_nanosecond())
            .ok()
            .map(Instant::from_unix_nanos)
            .filter(|t| !t.is_nat())
            .ok_or_else(range)
    }

    fn local_error(&self, local: &CfDate, reason: &'static str) -> UnitError {
        UnitError::LocalTime {
            time: local.to_string(),
            zone: self.name.to_string(),
            reason,
        }
    }
}

impl FromStr for TimeZone {
    type Err = UnitError;

    /// `UTC` (or `Z`, `GMT`), an offset (`+05:30`, `-0400`, `+05`), or an IANA name.
    fn from_str(s: &str) -> Result<Self, UnitError> {
        let t = s.trim();
        if ["UTC", "Z", "GMT"]
            .iter()
            .any(|u| t.eq_ignore_ascii_case(u))
        {
            return Ok(Self::utc());
        }
        if let Some(secs) = parse_offset(t) {
            return Self::fixed(secs);
        }
        Self::iana(t)
    }
}

/// Seconds east of UTC for `±hh`, `±hhmm`, `±hh:mm` or `±hh:mm:ss`.
fn parse_offset(s: &str) -> Option<i32> {
    let (neg, rest) = match s.as_bytes().first()? {
        b'+' => (false, s.get(1..)?),
        b'-' => (true, s.get(1..)?),
        _ => return None,
    };
    let num = |p: &str| -> Option<i32> {
        (p.len() == 2 && p.bytes().all(|b| b.is_ascii_digit()))
            .then(|| p.parse().ok())
            .flatten()
    };
    let parts: Vec<&str> = if rest.contains(':') {
        rest.split(':').collect()
    } else if rest.len() == 4 {
        vec![rest.get(..2)?, rest.get(2..)?]
    } else {
        vec![rest]
    };
    let (h, m, sec) = match parts.as_slice() {
        [h] => (num(h)?, 0, 0),
        [h, m] => (num(h)?, num(m)?, 0),
        [h, m, sec] => (num(h)?, num(m)?, num(sec)?),
        _ => return None,
    };
    if m >= 60 || sec >= 60 {
        return None;
    }
    let total = h
        .checked_mul(3600)?
        .checked_add(m.checked_mul(60)?)?
        .checked_add(sec)?;
    if neg {
        total.checked_neg()
    } else {
        Some(total)
    }
}

impl PartialEq for TimeZone {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for TimeZone {}

impl std::hash::Hash for TimeZone {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl fmt::Display for TimeZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl fmt::Debug for TimeZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimeZone({:?})", &*self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(s: &str) -> Instant {
        s.parse().unwrap()
    }

    #[test]
    fn names() {
        assert_eq!("utc".parse::<TimeZone>().unwrap(), TimeZone::utc());
        assert_eq!("+05:30".parse::<TimeZone>().unwrap().name(), "+05:30");
        assert_eq!("-0400".parse::<TimeZone>().unwrap().name(), "-04:00");
        assert_eq!("+05".parse::<TimeZone>().unwrap().name(), "+05:00");
        assert_eq!(TimeZone::fixed(0).unwrap().name(), "+00:00");
        assert!(!TimeZone::fixed(0).unwrap().is_utc());
        let ny: TimeZone = "america/new_york".parse().unwrap();
        assert_eq!(ny.name(), "America/New_York");
        assert!(matches!(
            "Mars/Olympus_Mons".parse::<TimeZone>(),
            Err(UnitError::UnknownTimeZone(_))
        ));
        assert!("+26:00".parse::<TimeZone>().is_err());
    }

    #[test]
    fn offsets() {
        let ny: TimeZone = "America/New_York".parse().unwrap();
        assert_eq!(ny.offset_at(at("2024-07-01")), -4 * 3600);
        assert_eq!(ny.offset_at(at("2024-01-01")), -5 * 3600);
        assert_eq!(ny.to_local(at("2024-07-01 12:00")).unwrap().hour(), 8);
        let ist: TimeZone = "+05:30".parse().unwrap();
        let local = ist.to_local(at("2024-07-01 00:00")).unwrap();
        assert_eq!((local.hour(), local.minute()), (5, 30));
    }

    #[test]
    fn wall_clock_times() {
        let ny: TimeZone = "America/New_York".parse().unwrap();
        let wall = |s: &str| at(s).to_utc().unwrap();
        let plain = ny
            .instant_of(&wall("2024-07-01 08:00"), Disambiguation::Raise)
            .unwrap();
        assert_eq!(plain, at("2024-07-01 12:00"));
        // 02:30 on 2024-03-10 is skipped.
        let gap = wall("2024-03-10 02:30");
        assert!(matches!(
            ny.instant_of(&gap, Disambiguation::Raise),
            Err(UnitError::LocalTime {
                reason: "skipped",
                ..
            })
        ));
        assert_eq!(
            ny.instant_of(&gap, Disambiguation::Compatible).unwrap(),
            at("2024-03-10 07:30")
        );
        // 01:30 on 2024-11-03 happens twice.
        let fold = wall("2024-11-03 01:30");
        assert!(matches!(
            ny.instant_of(&fold, Disambiguation::Raise),
            Err(UnitError::LocalTime {
                reason: "ambiguous",
                ..
            })
        ));
        assert_eq!(
            ny.instant_of(&fold, Disambiguation::Earlier).unwrap(),
            at("2024-11-03 05:30")
        );
        assert_eq!(
            ny.instant_of(&fold, Disambiguation::Later).unwrap(),
            at("2024-11-03 06:30")
        );
    }
}
