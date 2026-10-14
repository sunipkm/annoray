//! CF calendars, calendar-aware dates, and decoding/encoding of `<unit> since <date>` values.
//!
//! chrono only knows the proleptic Gregorian calendar. [`CfDate`] covers every CF calendar and
//! converts to and from [`chrono::NaiveDateTime`] where that is meaningful. Semantics follow
//! cftime: `standard`/`julian` have no year zero; floating-point offsets round to the nearest
//! microsecond (ties to even).

use std::fmt;
use std::iter::Peekable;
use std::str::{Chars, FromStr};

use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike};

use crate::UnitError;
use crate::ck::Ck;

const NS_PER_SEC: i128 = 1_000_000_000;
const NS_PER_DAY: i128 = 86_400 * NS_PER_SEC;
/// `NaiveDate::num_days_from_ce` of 1970-01-01.
const CE_TO_UNIX_DAYS: i128 = 719_163;

/// A CF calendar (CF conventions §4.4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Calendar {
    /// `standard` / `gregorian`: Julian before 1582-10-05, Gregorian from 1582-10-15.
    Standard,
    /// `proleptic_gregorian`: Gregorian rules extended to all dates.
    ProlepticGregorian,
    /// `julian`: a leap year every four years.
    Julian,
    /// `noleap` / `365_day`.
    NoLeap,
    /// `all_leap` / `366_day`.
    AllLeap,
    /// `360_day`: twelve 30-day months.
    Day360,
}

impl Calendar {
    /// The canonical CF name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::ProlepticGregorian => "proleptic_gregorian",
            Self::Julian => "julian",
            Self::NoLeap => "noleap",
            Self::AllLeap => "all_leap",
            Self::Day360 => "360_day",
        }
    }

    /// Whether dates in this calendar correspond to real instants (and so convert to chrono
    /// preserving the instant rather than the fields).
    #[must_use]
    pub const fn is_real_world(self) -> bool {
        matches!(
            self,
            Self::Standard | Self::ProlepticGregorian | Self::Julian
        )
    }

    /// Whether year 0 exists (cftime's default `has_year_zero`).
    #[must_use]
    pub const fn has_year_zero(self) -> bool {
        !matches!(self, Self::Standard | Self::Julian)
    }

    /// Calendar year to astronomical year (where year 0 = 1 BC).
    fn astronomical(self, year: i64) -> Option<i64> {
        match (self.has_year_zero(), year) {
            (true, y) => Some(y),
            (false, 0) => None,
            (false, y) if y < 0 => y.checked_add(1),
            (false, y) => Some(y),
        }
    }

    fn year_from_astronomical(self, a: i64) -> Option<i64> {
        if !self.has_year_zero() && a <= 0 {
            a.checked_sub(1)
        } else {
            Some(a)
        }
    }

    /// Days in `month` of `year` (calendar numbering). `None` for invalid months or year 0
    /// in calendars without one.
    #[must_use]
    pub fn days_in_month(self, year: i64, month: u8) -> Option<u8> {
        let a = self.astronomical(year)?;
        let leap = match self {
            Self::ProlepticGregorian => gregorian_leap(a),
            Self::Julian => a.checked_rem(4)? == 0,
            Self::Standard if a < 1582 => a.checked_rem(4)? == 0,
            Self::Standard => gregorian_leap(a),
            Self::NoLeap => false,
            Self::AllLeap => true,
            Self::Day360 => return (1..=12).contains(&month).then_some(30),
        };
        let days = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 if leap => 29,
            2 => 28,
            _ => return None,
        };
        Some(days)
    }
}

fn gregorian_leap(a: i64) -> bool {
    let div = |n: i64| a.checked_rem(n) == Some(0);
    div(4) && (!div(100) || div(400))
}

impl fmt::Display for Calendar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Calendar {
    type Err = UnitError;

    fn from_str(s: &str) -> Result<Self, UnitError> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "standard" | "gregorian" => Self::Standard,
            "proleptic_gregorian" => Self::ProlepticGregorian,
            "julian" => Self::Julian,
            "noleap" | "365_day" => Self::NoLeap,
            "all_leap" | "366_day" => Self::AllLeap,
            "360_day" => Self::Day360,
            _ => return Err(UnitError::UnknownCalendar(s.to_owned())),
        })
    }
}

// ----- day-number algorithms (H. Hinnant, "chrono-Compatible Low-Level Date Algorithms") -----

/// Days from 1970-01-01 for a proleptic Gregorian date (astronomical year).
fn days_from_gregorian(y: i64, m: u8, d: u8) -> Ck {
    let y = Ck::new(y) - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (Ck::new(m) + 9).rem_euclid(12);
    let doy = (mp * 153 + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn gregorian_from_days(z: i128) -> Result<(i64, u8, u8), UnitError> {
    let z = Ck::new(z) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (yoe * 365 + yoe / 4 - yoe / 100);
    finish_civil(yoe + era * 400, doy)
}

/// Days from 1970-01-01 (Gregorian) for a Julian-calendar date (astronomical year).
fn days_from_julian(y: i64, m: u8, d: u8) -> Ck {
    let y = Ck::new(y) - i64::from(m <= 2);
    let era = y.div_euclid(4);
    let yoe = y - era * 4;
    let mp = (Ck::new(m) + 9).rem_euclid(12);
    let doy = (mp * 153 + 2) / 5 + d - 1;
    era * 1461 + yoe * 365 + doy - 719_470
}

fn julian_from_days(z: i128) -> Result<(i64, u8, u8), UnitError> {
    let z = Ck::new(z) + 719_470;
    let era = z.div_euclid(1461);
    let doe = z - era * 1461;
    let yoe = (doe - doe / 1460) / 365;
    let doy = doe - yoe * 365;
    finish_civil(yoe + era * 4, doy)
}

/// Shared tail: March-based day-of-year to (year, month, day).
fn finish_civil(y: Ck, doy: Ck) -> Result<(i64, u8, u8), UnitError> {
    let mp = (doy * 5 + 2) / 153;
    let d = doy - (mp * 153 + 2) / 5 + 1;
    let m = if mp.get()? < 10 { mp + 3 } else { mp - 9 };
    let y = y + i64::from(m.get()? <= 2);
    Ok((y.to()?, m.to()?, d.to()?))
}

const CUM_NOLEAP: [i16; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
const CUM_LEAP: [i16; 12] = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];

fn fixed_year_from_days(
    z: i128,
    year_len: i64,
    cum: &[i16; 12],
) -> Result<(i64, u8, u8), UnitError> {
    let y = Ck::new(z).div_euclid(year_len);
    let doy = Ck::new(z).rem_euclid(year_len).get()?;
    let (m, start) = cum
        .iter()
        .enumerate()
        .rev()
        .find(|(_, start)| i128::from(**start) <= doy)
        .ok_or(UnitError::Overflow)?;
    Ok((
        y.to()?,
        (Ck::from(u8::try_from(m).map_err(|_| UnitError::Overflow)?) + 1).to()?,
        (Ck::new(doy) - *start + 1).to()?,
    ))
}

/// The first Gregorian day of the `standard` calendar, as a day number.
fn standard_cutover() -> Ck {
    days_from_gregorian(1582, 10, 15)
}

/// Day number of a date: days since 1970-01-01 for real-world calendars (so they are
/// comparable), days since year 0 for idealized ones.
fn day_number(cal: Calendar, a: i64, m: u8, d: u8) -> Result<i128, UnitError> {
    let cum = |t: &[i16; 12]| {
        t.get(usize::from(m).saturating_sub(1))
            .copied()
            .ok_or(UnitError::Overflow)
    };
    match cal {
        Calendar::ProlepticGregorian => days_from_gregorian(a, m, d).get(),
        Calendar::Julian => days_from_julian(a, m, d).get(),
        Calendar::Standard if (a, m, d) >= (1582, 10, 15) => days_from_gregorian(a, m, d).get(),
        Calendar::Standard => days_from_julian(a, m, d).get(),
        Calendar::NoLeap => (Ck::new(a) * 365 + cum(&CUM_NOLEAP)? + d - 1).get(),
        Calendar::AllLeap => (Ck::new(a) * 366 + cum(&CUM_LEAP)? + d - 1).get(),
        Calendar::Day360 => (Ck::new(a) * 360 + (Ck::new(m) - 1) * 30 + d - 1).get(),
    }
}

/// Inverse of [`day_number`]; returns the astronomical year.
fn from_day_number(cal: Calendar, z: i128) -> Result<(i64, u8, u8), UnitError> {
    match cal {
        Calendar::ProlepticGregorian => gregorian_from_days(z),
        Calendar::Julian => julian_from_days(z),
        Calendar::Standard if z >= standard_cutover().get()? => gregorian_from_days(z),
        Calendar::Standard => julian_from_days(z),
        Calendar::NoLeap => fixed_year_from_days(z, 365, &CUM_NOLEAP),
        Calendar::AllLeap => fixed_year_from_days(z, 366, &CUM_LEAP),
        Calendar::Day360 => {
            let y = Ck::new(z).div_euclid(360);
            let r = Ck::new(z).rem_euclid(360);
            Ok((y.to()?, (r / 30 + 1).to()?, (r % 30 + 1).to()?))
        }
    }
}

// ----- CfDate -----

/// A date and time in a specific CF calendar, with nanosecond precision. Always valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CfDate {
    calendar: Calendar,
    year: i64,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
    nanosecond: u32,
}

impl CfDate {
    /// Midnight on the given day.
    ///
    /// # Errors
    /// [`UnitError::InvalidDate`] if the date does not exist in `calendar`.
    pub fn from_ymd(calendar: Calendar, year: i64, month: u8, day: u8) -> Result<Self, UnitError> {
        let d = Self {
            calendar,
            year,
            month,
            day,
            hour: 0,
            minute: 0,
            second: 0,
            nanosecond: 0,
        };
        let valid = calendar
            .days_in_month(year, month)
            .is_some_and(|n| (1..=n).contains(&day))
            && !(calendar == Calendar::Standard
                && year == 1582
                && month == 10
                && (5..=14).contains(&day));
        if valid { Ok(d) } else { Err(d.invalid()) }
    }

    /// The same day at the given time of day.
    ///
    /// # Errors
    /// [`UnitError::InvalidDate`] if a component is out of range.
    pub fn and_hms_nano(
        self,
        hour: u8,
        minute: u8,
        second: u8,
        nanosecond: u32,
    ) -> Result<Self, UnitError> {
        let d = Self {
            hour,
            minute,
            second,
            nanosecond,
            ..self
        };
        if hour < 24 && minute < 60 && second < 60 && i128::from(nanosecond) < NS_PER_SEC {
            Ok(d)
        } else {
            Err(d.invalid())
        }
    }

    fn invalid(&self) -> UnitError {
        UnitError::InvalidDate {
            date: self.to_string(),
            calendar: self.calendar.name(),
        }
    }

    /// The calendar.
    #[must_use]
    pub const fn calendar(&self) -> Calendar {
        self.calendar
    }
    /// Year, in the calendar's numbering (no year 0 for `standard`/`julian`).
    #[must_use]
    pub const fn year(&self) -> i64 {
        self.year
    }
    /// Month, 1–12.
    #[must_use]
    pub const fn month(&self) -> u8 {
        self.month
    }
    /// Day of month, from 1.
    #[must_use]
    pub const fn day(&self) -> u8 {
        self.day
    }
    /// Hour, 0–23.
    #[must_use]
    pub const fn hour(&self) -> u8 {
        self.hour
    }
    /// Minute, 0–59.
    #[must_use]
    pub const fn minute(&self) -> u8 {
        self.minute
    }
    /// Second, 0–59.
    #[must_use]
    pub const fn second(&self) -> u8 {
        self.second
    }
    /// Nanoseconds within the second.
    #[must_use]
    pub const fn nanosecond(&self) -> u32 {
        self.nanosecond
    }

    fn day_number(&self) -> Result<i128, UnitError> {
        let a = self
            .calendar
            .astronomical(self.year)
            .ok_or_else(|| self.invalid())?;
        day_number(self.calendar, a, self.month, self.day)
    }

    fn ns_of_day(&self) -> Ck {
        ((Ck::new(self.hour) * 60 + self.minute) * 60 + self.second) * NS_PER_SEC + self.nanosecond
    }

    /// Nanoseconds since the calendar's internal origin.
    fn total_ns(&self) -> Result<i128, UnitError> {
        (Ck::new(self.day_number()?) * NS_PER_DAY + self.ns_of_day()).get()
    }

    /// Inverse of [`total_ns`](Self::total_ns).
    fn from_total_ns(calendar: Calendar, total: i128) -> Result<Self, UnitError> {
        let days = Ck::new(total).div_euclid(NS_PER_DAY).get()?;
        let tod = Ck::new(total).rem_euclid(NS_PER_DAY);
        let (a, m, d) = from_day_number(calendar, days)?;
        let year = calendar
            .year_from_astronomical(a)
            .ok_or(UnitError::Overflow)?;
        Self::from_ymd(calendar, year, m, d)?.and_hms_nano(
            (tod / (3600 * NS_PER_SEC)).to()?,
            (tod / (60 * NS_PER_SEC) % 60).to()?,
            (tod / NS_PER_SEC % 60).to()?,
            (tod % NS_PER_SEC).to()?,
        )
    }

    /// This date shifted by `ns` nanoseconds, within its calendar.
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if the result is out of range.
    pub fn checked_add_nanos(&self, ns: i128) -> Result<Self, UnitError> {
        Self::from_total_ns(self.calendar, (Ck::new(self.total_ns()?) + ns).get()?)
    }

    /// Nanoseconds from `earlier` to `self`.
    ///
    /// # Errors
    /// [`UnitError::InvalidDate`] if the calendars differ; [`UnitError::Overflow`].
    pub fn nanos_since(&self, earlier: &Self) -> Result<i128, UnitError> {
        if self.calendar != earlier.calendar {
            return Err(earlier.invalid_in(self.calendar));
        }
        (Ck::new(self.total_ns()?) - earlier.total_ns()?).get()
    }

    /// The first instant of the next month (rolling into the next year after December).
    ///
    /// # Errors
    /// [`UnitError::Overflow`] at the end of the representable range.
    pub fn start_of_next_month(&self) -> Result<Self, UnitError> {
        let (y, m) = if self.month == 12 {
            (self.year.checked_add(1).ok_or(UnitError::Overflow)?, 1)
        } else {
            (
                self.year,
                self.month.checked_add(1).ok_or(UnitError::Overflow)?,
            )
        };
        // Skip year 0 in calendars without one.
        let y = if y == 0 && !self.calendar.has_year_zero() {
            1
        } else {
            y
        };
        Self::from_ymd(self.calendar, y, m, 1)
    }

    fn invalid_in(&self, calendar: Calendar) -> UnitError {
        UnitError::InvalidDate {
            date: self.to_string(),
            calendar: calendar.name(),
        }
    }

    /// Convert a chrono datetime into `calendar`.
    ///
    /// Real-world calendars keep the instant (so a `julian` date may differ in its fields);
    /// idealized calendars keep the fields and fail if the date does not exist (e.g. Feb 29
    /// in `noleap`).
    ///
    /// # Errors
    /// [`UnitError::InvalidDate`] or [`UnitError::Overflow`].
    pub fn from_chrono(dt: NaiveDateTime, calendar: Calendar) -> Result<Self, UnitError> {
        let (h, mi, s) = (dt.hour(), dt.minute(), dt.second());
        let ns = dt.nanosecond();
        let to8 = |v: u32| u8::try_from(v).map_err(|_| UnitError::Overflow);
        if calendar.is_real_world() {
            let days = Ck::new(dt.date().num_days_from_ce()) - CE_TO_UNIX_DAYS;
            let tod = ((Ck::new(h) * 60 + mi) * 60 + s) * NS_PER_SEC + ns;
            Self::from_total_ns(calendar, (days * NS_PER_DAY + tod).get()?)
        } else {
            Self::from_ymd(
                calendar,
                i64::from(dt.year()),
                to8(dt.month())?,
                to8(dt.day())?,
            )?
            .and_hms_nano(to8(h)?, to8(mi)?, to8(s)?, ns)
        }
    }

    /// Convert to chrono, clamping days that do not exist in the Gregorian calendar to the
    /// end of the month (e.g. 360_day Feb 30 → Feb 28/29). Real-world calendars are exact.
    ///
    /// # Errors
    /// [`UnitError::ChronoRange`] if the year is outside chrono's range.
    pub fn to_chrono_clamped(&self) -> Result<NaiveDateTime, UnitError> {
        if self.calendar.is_real_world() {
            return NaiveDateTime::try_from(*self);
        }
        let last = Calendar::ProlepticGregorian
            .days_in_month(self.year, self.month)
            .ok_or_else(|| self.chrono_range())?;
        let clamped = Self {
            calendar: Calendar::ProlepticGregorian,
            day: self.day.min(last),
            ..*self
        };
        NaiveDateTime::try_from(clamped)
    }

    fn chrono_range(&self) -> UnitError {
        UnitError::ChronoRange(self.to_string())
    }
}

impl TryFrom<CfDate> for NaiveDateTime {
    type Error = UnitError;

    /// Real-world calendars convert exactly (same instant). Idealized calendars convert
    /// field by field and fail for dates chrono's calendar lacks (360_day Feb 30).
    fn try_from(d: CfDate) -> Result<Self, UnitError> {
        let date = if d.calendar.is_real_world() {
            let ce = (Ck::new(d.day_number()?) + CE_TO_UNIX_DAYS)
                .to::<i32>()
                .map_err(|_| d.chrono_range())?;
            NaiveDate::from_num_days_from_ce_opt(ce)
        } else {
            i32::try_from(d.year)
                .ok()
                .and_then(|y| NaiveDate::from_ymd_opt(y, u32::from(d.month), u32::from(d.day)))
        };
        date.and_then(|date| {
            date.and_hms_nano_opt(
                u32::from(d.hour),
                u32::from(d.minute),
                u32::from(d.second),
                d.nanosecond,
            )
        })
        .ok_or_else(|| d.chrono_range())
    }
}

impl fmt::Display for CfDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.year < 0 {
            write!(f, "-{:04}", self.year.unsigned_abs())?;
        } else {
            write!(f, "{:04}", self.year)?;
        }
        write!(
            f,
            "-{:02}-{:02} {:02}:{:02}:{:02}",
            self.month, self.day, self.hour, self.minute, self.second
        )?;
        if self.nanosecond != 0 {
            write!(f, ".{:09}", self.nanosecond)?;
        }
        Ok(())
    }
}

// ----- reference dates in unit strings -----

/// The reference date of a `<unit> since <date>` string, before a calendar is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub struct RefDate {
    pub year: i64,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub nanosecond: u32,
    /// Offset of the written time from UTC (e.g. `+05:30` → 330).
    pub utc_offset_minutes: i16,
}

impl RefDate {
    /// Resolve in `calendar`, giving the UTC instant as a [`CfDate`].
    ///
    /// # Errors
    /// [`UnitError::InvalidDate`] if the date does not exist in `calendar`.
    pub fn in_calendar(&self, calendar: Calendar) -> Result<CfDate, UnitError> {
        let local = CfDate::from_ymd(calendar, self.year, self.month, self.day)?.and_hms_nano(
            self.hour,
            self.minute,
            self.second,
            self.nanosecond,
        )?;
        if self.utc_offset_minutes == 0 {
            return Ok(local);
        }
        let shift = Ck::new(self.utc_offset_minutes) * 60 * NS_PER_SEC;
        CfDate::from_total_ns(calendar, (Ck::new(local.total_ns()?) - shift).get()?)
    }
}

impl fmt::Display for RefDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.year < 0 {
            write!(f, "-{:04}", self.year.unsigned_abs())?;
        } else {
            write!(f, "{:04}", self.year)?;
        }
        write!(
            f,
            "-{:02}-{:02} {:02}:{:02}:{:02}",
            self.month, self.day, self.hour, self.minute, self.second
        )?;
        if self.nanosecond != 0 {
            write!(f, ".{:09}", self.nanosecond)?;
        }
        if self.utc_offset_minutes != 0 {
            let sign = if self.utc_offset_minutes < 0 {
                '-'
            } else {
                '+'
            };
            let m = self.utc_offset_minutes.unsigned_abs();
            write!(f, " {sign}{:02}:{:02}", m / 60, m % 60)?;
        }
        Ok(())
    }
}

struct Cursor<'a>(Peekable<Chars<'a>>);

impl Cursor<'_> {
    fn eat(&mut self, c: char) -> bool {
        self.0.next_if_eq(&c).is_some()
    }
    fn skip_ws(&mut self) {
        while self.0.next_if(|c| c.is_whitespace()).is_some() {}
    }
    fn peek_digit(&mut self) -> bool {
        self.0.peek().is_some_and(char::is_ascii_digit)
    }
    /// One or more ASCII digits: (value, digit count).
    fn digits(&mut self) -> Option<(i64, u32)> {
        let mut v: i64 = 0;
        let mut n: u32 = 0;
        while let Some(c) = self.0.next_if(char::is_ascii_digit) {
            v = v.checked_mul(10)?.checked_add(i64::from(c.to_digit(10)?))?;
            n = n.checked_add(1)?;
        }
        (n > 0).then_some((v, n))
    }
    fn small<T: TryFrom<i64>>(&mut self) -> Option<T> {
        self.digits().and_then(|(v, _)| T::try_from(v).ok())
    }
    fn at_end(&mut self) -> bool {
        self.0.peek().is_none()
    }
}

impl FromStr for RefDate {
    type Err = UnitError;

    /// Parse `YYYY-MM-DD[( |T)hh[:mm[:ss[.fff]]]][ ][Z|UTC|GMT|±hh[:mm]]`.
    fn from_str(s: &str) -> Result<Self, UnitError> {
        parse_ref_date(s).ok_or_else(|| UnitError::Parse {
            input: s.to_owned(),
            pos: 0,
            reason: "expected a reference date such as 1970-01-01 00:00:00".to_owned(),
        })
    }
}

fn parse_ref_date(s: &str) -> Option<RefDate> {
    let mut c = Cursor(s.chars().peekable());
    c.skip_ws();
    let neg = c.eat('-');
    if !neg {
        c.eat('+');
    }
    let (year, _) = c.digits()?;
    let year = if neg { year.checked_neg()? } else { year };
    if !c.eat('-') {
        return None;
    }
    let month: u8 = c.small()?;
    if !c.eat('-') {
        return None;
    }
    let day: u8 = c.small()?;
    let mut d = RefDate {
        year,
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
        nanosecond: 0,
        utc_offset_minutes: 0,
    };

    // Time of day.
    if !c.eat('T') {
        c.skip_ws();
    }
    if c.peek_digit() {
        d.hour = c.small()?;
        if c.eat(':') {
            d.minute = c.small()?;
            if c.eat(':') {
                d.second = c.small()?;
                if c.eat('.') {
                    let mut ns: u32 = 0;
                    let mut scale: u32 = 100_000_000;
                    while let Some(ch) = c.0.next_if(char::is_ascii_digit) {
                        ns = ns.checked_add(ch.to_digit(10)?.checked_mul(scale)?)?;
                        scale /= 10;
                    }
                    d.nanosecond = ns;
                }
            }
        }
    }

    // Time zone.
    c.skip_ws();
    if c.eat('Z') {
    } else if c.eat('U') {
        (c.eat('T') && c.eat('C')).then_some(())?;
    } else if c.eat('G') {
        (c.eat('M') && c.eat('T')).then_some(())?;
    } else if let Some(sign) = c.0.next_if(|ch| *ch == '+' || *ch == '-') {
        let (hh, n) = c.digits()?;
        let (h, m) = if n > 2 {
            (hh.checked_div(100)?, hh.checked_rem(100)?)
        } else {
            (hh, if c.eat(':') { c.digits()?.0 } else { 0 })
        };
        let total = i16::try_from(h.checked_mul(60)?.checked_add(m)?).ok()?;
        d.utc_offset_minutes = if sign == '-' {
            total.checked_neg()?
        } else {
            total
        };
    }
    c.skip_ws();
    let ranges_ok = (1..=12).contains(&d.month)
        && (1..=31).contains(&d.day)
        && d.hour < 24
        && d.minute < 60
        && d.second < 60;
    (c.at_end() && ranges_ok).then_some(d)
}

/// Whether `s` looks like a reference date (used to tell `K @ 273.15` from `s @ 1970-01-01`).
pub(crate) fn looks_like_date(s: &str) -> Option<RefDate> {
    parse_ref_date(s)
}

// ----- epoch codec -----

/// Converts between numbers in `<step> since <reference>` units and [`CfDate`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochCodec {
    calendar: Calendar,
    step_ns: i128,
    ref_ns: i128,
}

impl EpochCodec {
    /// A codec for `step_ns` nanoseconds per unit since `reference`, in `calendar`.
    ///
    /// # Errors
    /// [`UnitError::InvalidDate`] if the reference date does not exist in `calendar`;
    /// [`UnitError::Overflow`] if `step_ns <= 0`.
    pub fn new(step_ns: i128, reference: &RefDate, calendar: Calendar) -> Result<Self, UnitError> {
        if step_ns <= 0 {
            return Err(UnitError::Overflow);
        }
        let ref_ns = reference.in_calendar(calendar)?.total_ns()?;
        Ok(Self {
            calendar,
            step_ns,
            ref_ns,
        })
    }

    /// The calendar.
    #[must_use]
    pub const fn calendar(&self) -> Calendar {
        self.calendar
    }

    /// Nanoseconds per unit step.
    #[must_use]
    pub const fn step_ns(&self) -> i128 {
        self.step_ns
    }

    /// The reference date as a [`CfDate`].
    ///
    /// # Errors
    /// [`UnitError::Overflow`].
    pub fn reference(&self) -> Result<CfDate, UnitError> {
        CfDate::from_total_ns(self.calendar, self.ref_ns)
    }

    /// Decode an integer offset (exact).
    ///
    /// # Errors
    /// [`UnitError::Overflow`] if the date is out of range.
    pub fn decode_int(&self, n: i64) -> Result<CfDate, UnitError> {
        CfDate::from_total_ns(
            self.calendar,
            (Ck::new(self.ref_ns) + Ck::new(n) * self.step_ns).get()?,
        )
    }

    /// Decode a floating-point offset, rounding to the nearest microsecond (ties to even),
    /// as cftime does.
    ///
    /// # Errors
    /// [`UnitError::NonFiniteTime`] for NaN/∞; [`UnitError::Overflow`] if out of range.
    pub fn decode_float(&self, x: f64) -> Result<CfDate, UnitError> {
        if !x.is_finite() {
            return Err(UnitError::NonFiniteTime(x.to_string()));
        }
        #[allow(clippy::cast_precision_loss)]
        let step_us = self.step_ns as f64 / 1000.0;
        let us = (x * step_us).round_ties_even();
        if !us.is_finite() || us.abs() >= 1e36 {
            return Err(UnitError::Overflow);
        }
        #[allow(clippy::cast_possible_truncation)] // range checked above
        let us = us as i128;
        CfDate::from_total_ns(
            self.calendar,
            (Ck::new(self.ref_ns) + Ck::new(us) * 1000).get()?,
        )
    }

    fn offset_ns(&self, date: &CfDate) -> Result<i128, UnitError> {
        if date.calendar != self.calendar {
            return Err(UnitError::InvalidDate {
                date: date.to_string(),
                calendar: self.calendar.name(),
            });
        }
        (Ck::new(date.total_ns()?) - self.ref_ns).get()
    }

    /// Encode as a (possibly fractional) number of steps.
    ///
    /// # Errors
    /// [`UnitError::InvalidDate`] if `date` is in another calendar.
    #[allow(clippy::cast_precision_loss)]
    pub fn encode_float(&self, date: &CfDate) -> Result<f64, UnitError> {
        let ns = self.offset_ns(date)?;
        let whole = Ck::new(ns).div_euclid(self.step_ns).get()?;
        let rem = Ck::new(ns).rem_euclid(self.step_ns).get()?;
        Ok(whole as f64 + rem as f64 / self.step_ns as f64)
    }

    /// Encode as an exact whole number of steps.
    ///
    /// # Errors
    /// [`UnitError::NotWholeSteps`] if the date falls between steps.
    pub fn encode_int(&self, date: &CfDate) -> Result<i64, UnitError> {
        let ns = self.offset_ns(date)?;
        if Ck::new(ns).rem_euclid(self.step_ns).get()? != 0 {
            return Err(UnitError::NotWholeSteps {
                date: date.to_string(),
                step: format!("{} ns", self.step_ns),
            });
        }
        Ck::new(ns).div_euclid(self.step_ns).to()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Calendar; 6] = [
        Calendar::Standard,
        Calendar::ProlepticGregorian,
        Calendar::Julian,
        Calendar::NoLeap,
        Calendar::AllLeap,
        Calendar::Day360,
    ];

    #[test]
    fn day_numbers_round_trip() {
        for cal in ALL {
            for z in (-800_000i128..800_000).step_by(997).chain(-1500..1500) {
                let (a, m, d) = from_day_number(cal, z).unwrap();
                assert_eq!(day_number(cal, a, m, d).unwrap(), z, "{cal} {a}-{m}-{d}");
            }
        }
        assert_eq!(days_from_gregorian(1970, 1, 1).get(), Ok(0));
        assert_eq!(days_from_julian(1970, 1, 1).get(), Ok(13));
        // The standard calendar jumps from Julian 1582-10-04 to Gregorian 1582-10-15.
        assert_eq!(
            days_from_julian(1582, 10, 4).get().unwrap() + 1,
            days_from_gregorian(1582, 10, 15).get().unwrap()
        );
    }

    #[test]
    fn invalid_dates() {
        assert!(CfDate::from_ymd(Calendar::Standard, 1582, 10, 10).is_err());
        assert!(CfDate::from_ymd(Calendar::Standard, 0, 1, 1).is_err());
        assert!(CfDate::from_ymd(Calendar::ProlepticGregorian, 0, 1, 1).is_ok());
        assert!(CfDate::from_ymd(Calendar::NoLeap, 2000, 2, 29).is_err());
        assert!(CfDate::from_ymd(Calendar::Day360, 2001, 2, 30).is_ok());
        assert!(CfDate::from_ymd(Calendar::Julian, 1900, 2, 29).is_ok());
        assert!(CfDate::from_ymd(Calendar::ProlepticGregorian, 1900, 2, 29).is_err());
        assert!(
            CfDate::from_ymd(Calendar::Standard, 2000, 1, 1)
                .unwrap()
                .and_hms_nano(24, 0, 0, 0)
                .is_err()
        );
    }

    #[test]
    fn ref_date_parsing() {
        let p = |s: &str| s.parse::<RefDate>();
        let d = p("1970-01-01").unwrap();
        assert_eq!((d.year, d.month, d.day, d.hour), (1970, 1, 1, 0));
        let d = p("2000-03-01T06:00:00Z").unwrap();
        assert_eq!(d.hour, 6);
        let d = p("2000-01-01 00:00:00.5 +05:30").unwrap();
        assert_eq!((d.nanosecond, d.utc_offset_minutes), (500_000_000, 330));
        let d = p("1-1-1 0:0:0 UTC").unwrap();
        assert_eq!(d.year, 1);
        assert_eq!(p("-0100-03-01").unwrap().year, -100);
        assert_eq!(p("2000-01-01 -0600").unwrap().utc_offset_minutes, -360);
        assert!(p("2000-13-01").is_err());
        assert!(p("273.15").is_err());
        assert!(p("2000-01-01 junk").is_err());
    }

    #[test]
    fn chrono_round_trip() {
        let dt = NaiveDate::from_ymd_opt(1600, 3, 1)
            .unwrap()
            .and_hms_nano_opt(1, 2, 3, 4)
            .unwrap();
        for cal in [
            Calendar::Standard,
            Calendar::ProlepticGregorian,
            Calendar::Julian,
        ] {
            let cf = CfDate::from_chrono(dt, cal).unwrap();
            assert_eq!(NaiveDateTime::try_from(cf).unwrap(), dt, "{cal}");
        }
        // Same instant, different fields: Gregorian 1600-03-01 is Julian 1600-02-20.
        let j = CfDate::from_chrono(dt, Calendar::Julian).unwrap();
        assert_eq!((j.month(), j.day()), (2, 20));

        let feb30 = CfDate::from_ymd(Calendar::Day360, 2001, 2, 30).unwrap();
        assert!(matches!(
            NaiveDateTime::try_from(feb30),
            Err(UnitError::ChronoRange(_))
        ));
        assert_eq!(
            feb30.to_chrono_clamped().unwrap().date(),
            NaiveDate::from_ymd_opt(2001, 2, 28).unwrap()
        );
        let noleap = CfDate::from_ymd(Calendar::NoLeap, 2001, 3, 1).unwrap();
        assert_eq!(
            NaiveDateTime::try_from(noleap).unwrap().date(),
            NaiveDate::from_ymd_opt(2001, 3, 1).unwrap()
        );
    }

    fn step_ns(step: &str) -> i128 {
        match step {
            "days" => NS_PER_DAY,
            "hours" => 3600 * NS_PER_SEC,
            "minutes" => 60 * NS_PER_SEC,
            "seconds" => NS_PER_SEC,
            "milliseconds" => 1_000_000,
            "microseconds" => 1000,
            other => panic!("unknown step {other}"),
        }
    }

    /// Compare against cftime.num2date (fixtures/gen_cftime.py).
    #[test]
    fn matches_cftime() {
        let table = include_str!("../../../fixtures/cftime.txt");
        let mut n = 0;
        for line in table.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = line.split('|').collect();
            let cal: Calendar = f[0].parse().unwrap();
            let (step, reference) = f[1].split_once(" since ").unwrap();
            let codec = EpochCodec::new(step_ns(step), &reference.parse().unwrap(), cal).unwrap();
            let got = if f[2].contains(['.', 'e']) {
                codec.decode_float(f[2].parse().unwrap())
            } else {
                codec.decode_int(f[2].parse().unwrap())
            };
            if f[3] == "ERR" {
                assert!(got.is_err(), "{line}");
                continue;
            }
            let e: Vec<i64> = f[3].split(' ').map(|x| x.parse().unwrap()).collect();
            let d = got.unwrap_or_else(|err| panic!("{line}: {err}"));
            let us = i64::from(d.nanosecond() / 1000);
            let got = [
                d.year(),
                d.month().into(),
                d.day().into(),
                d.hour().into(),
                d.minute().into(),
                d.second().into(),
                us,
            ];
            assert_eq!(got.as_slice(), e.as_slice(), "{line}");

            // Integer offsets must encode back exactly.
            if !f[2].contains(['.', 'e']) {
                assert_eq!(
                    codec.encode_int(&d).unwrap(),
                    f[2].parse::<i64>().unwrap(),
                    "{line}"
                );
            }
            n += 1;
        }
        assert!(n > 5000, "{n}");
    }

    #[test]
    fn encode_rejects_partial_steps() {
        let r: RefDate = "2000-01-01".parse().unwrap();
        let codec = EpochCodec::new(NS_PER_DAY, &r, Calendar::NoLeap).unwrap();
        let d = CfDate::from_ymd(Calendar::NoLeap, 2000, 1, 2)
            .unwrap()
            .and_hms_nano(12, 0, 0, 0)
            .unwrap();
        assert!(matches!(
            codec.encode_int(&d),
            Err(UnitError::NotWholeSteps { .. })
        ));
        assert_eq!(codec.encode_float(&d).unwrap(), 1.5);
        assert!(codec.decode_float(f64::NAN).is_err());
        assert!(codec.decode_float(1e300).is_err());
    }
}
