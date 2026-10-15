//! Labels and label-based indexers for `sel`.

use std::fmt;
use std::ops::{RangeFrom, RangeFull, RangeInclusive, RangeToInclusive};

use annoray_units::{CfDate, Unit, UnitError};
use chrono::{NaiveDate, NaiveDateTime, TimeDelta};

use crate::CoreError;

/// A coordinate label: what `sel` looks up.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Label {
    /// An integer, in the coordinate's unit.
    Int(i64),
    /// A float, in the coordinate's unit.
    Float(f64),
    /// A string. On date coordinates it is parsed as an ISO 8601 date, possibly partial
    /// (`"2020-01"` selects all of January 2020).
    Str(String),
    /// A real-world date and time.
    DateTime(NaiveDateTime),
    /// A date in a CF calendar.
    CfDate(CfDate),
    /// A duration.
    TimeDelta(TimeDelta),
    /// A number with an explicit unit; converted into the coordinate's unit before lookup.
    Quantity(f64, Unit),
}

impl Label {
    /// A value with a unit, e.g. `Label::quantity(2.0, "km")`.
    ///
    /// # Errors
    /// If `unit` does not parse.
    pub fn quantity(value: f64, unit: &str) -> Result<Self, UnitError> {
        Ok(Self::Quantity(value, unit.parse()?))
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v:?}"),
            Self::DateTime(v) => write!(f, "{v}"),
            Self::CfDate(v) => write!(f, "{v}"),
            Self::TimeDelta(v) => write!(f, "{v}"),
            Self::Quantity(v, u) => write!(f, "{v} {u}"),
        }
    }
}

macro_rules! label_from {
    ($($ty:ty => |$v:ident| $e:expr),* $(,)?) => {$(
        impl From<$ty> for Label {
            fn from($v: $ty) -> Self {
                $e
            }
        }
    )*};
}

label_from!(
    i8 => |v| Self::Int(v.into()),
    i16 => |v| Self::Int(v.into()),
    i32 => |v| Self::Int(v.into()),
    i64 => |v| Self::Int(v),
    u8 => |v| Self::Int(v.into()),
    u16 => |v| Self::Int(v.into()),
    u32 => |v| Self::Int(v.into()),
    f32 => |v| Self::Float(v.into()),
    f64 => |v| Self::Float(v),
    &str => |v| Self::Str(v.to_owned()),
    String => |v| Self::Str(v),
    NaiveDateTime => |v| Self::DateTime(v),
    NaiveDate => |v| Self::DateTime(v.and_time(chrono::NaiveTime::MIN)),
    CfDate => |v| Self::CfDate(v),
    TimeDelta => |v| Self::TimeDelta(v),
);

/// How a scalar label matches coordinate values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    /// Only an exact match.
    #[default]
    Exact,
    /// The closest value (optionally within a tolerance).
    Nearest,
    /// The last value at or before the label (forward fill). Needs a sorted coordinate.
    Pad,
    /// The first value at or after the label (back fill). Needs a sorted coordinate.
    Backfill,
}

/// What to select along one dimension, by label.
#[derive(Debug, Clone, PartialEq)]
pub enum LabelIndexer {
    /// One label. Drops the dimension (except partial date strings, which select a range).
    Scalar {
        /// The label.
        label: Label,
        /// How to match it.
        method: Method,
        /// For `Nearest`: the largest acceptable distance.
        tolerance: Option<Label>,
    },
    /// A label range, inclusive at both ends as in xarray. Keeps the dimension.
    Slice {
        /// First label (inclusive); `None` for the start.
        start: Option<Label>,
        /// Last label (inclusive); `None` for the end.
        stop: Option<Label>,
        /// Positional step.
        step: Option<isize>,
    },
}

impl LabelIndexer {
    /// Exact match.
    pub fn exact(label: impl Into<Label>) -> Self {
        Self::Scalar {
            label: label.into(),
            method: Method::Exact,
            tolerance: None,
        }
    }
    /// Nearest match.
    pub fn nearest(label: impl Into<Label>) -> Self {
        Self::Scalar {
            label: label.into(),
            method: Method::Nearest,
            tolerance: None,
        }
    }
    /// Nearest match within `tolerance`.
    pub fn nearest_within(label: impl Into<Label>, tolerance: impl Into<Label>) -> Self {
        Self::Scalar {
            label: label.into(),
            method: Method::Nearest,
            tolerance: Some(tolerance.into()),
        }
    }
    /// Last value at or before `label`.
    pub fn pad(label: impl Into<Label>) -> Self {
        Self::Scalar {
            label: label.into(),
            method: Method::Pad,
            tolerance: None,
        }
    }
    /// First value at or after `label`.
    pub fn backfill(label: impl Into<Label>) -> Self {
        Self::Scalar {
            label: label.into(),
            method: Method::Backfill,
            tolerance: None,
        }
    }
    /// Inclusive label range.
    pub fn slice(start: Option<Label>, stop: Option<Label>) -> Self {
        Self::Slice {
            start,
            stop,
            step: None,
        }
    }
}

/// Conversion into a [`LabelIndexer`]: scalars give exact matches, inclusive ranges give slices.
pub trait IntoLabelIndexer {
    /// Perform the conversion.
    ///
    /// # Errors
    /// Conversions are infallible today; the `Result` leaves room for checked ones.
    fn into_label_indexer(self) -> Result<LabelIndexer, CoreError>;
}

impl IntoLabelIndexer for LabelIndexer {
    fn into_label_indexer(self) -> Result<LabelIndexer, CoreError> {
        Ok(self)
    }
}

impl IntoLabelIndexer for Label {
    fn into_label_indexer(self) -> Result<LabelIndexer, CoreError> {
        Ok(LabelIndexer::exact(self))
    }
}

impl IntoLabelIndexer for RangeFull {
    fn into_label_indexer(self) -> Result<LabelIndexer, CoreError> {
        Ok(LabelIndexer::slice(None, None))
    }
}

macro_rules! into_label_indexer {
    ($($ty:ty),*) => {$(
        impl IntoLabelIndexer for $ty {
            fn into_label_indexer(self) -> Result<LabelIndexer, CoreError> {
                Ok(LabelIndexer::exact(self))
            }
        }
        impl IntoLabelIndexer for RangeInclusive<$ty> {
            fn into_label_indexer(self) -> Result<LabelIndexer, CoreError> {
                let (a, b) = self.into_inner();
                Ok(LabelIndexer::slice(Some(a.into()), Some(b.into())))
            }
        }
        impl IntoLabelIndexer for RangeFrom<$ty> {
            fn into_label_indexer(self) -> Result<LabelIndexer, CoreError> {
                Ok(LabelIndexer::slice(Some(self.start.into()), None))
            }
        }
        impl IntoLabelIndexer for RangeToInclusive<$ty> {
            fn into_label_indexer(self) -> Result<LabelIndexer, CoreError> {
                Ok(LabelIndexer::slice(None, Some(self.end.into())))
            }
        }
    )*};
}

into_label_indexer!(
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    f32,
    f64,
    &str,
    String,
    NaiveDateTime,
    NaiveDate,
    CfDate,
    TimeDelta
);

/// How precisely a (possibly partial) date string was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Resolution {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

/// A parsed ISO 8601 date string, possibly partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PartialDate {
    pub(crate) year: i64,
    pub(crate) month: u8,
    pub(crate) day: u8,
    pub(crate) hour: u8,
    pub(crate) minute: u8,
    pub(crate) second: u8,
    pub(crate) nanosecond: u32,
    pub(crate) resolution: Resolution,
}

/// Parse `YYYY[-MM[-DD[(T| )hh[:mm[:ss[.fff]]]]]]`.
pub(crate) fn parse_partial_date(s: &str) -> Option<PartialDate> {
    let s = s.trim();
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let (date, time) = match body.split_once(['T', ' ']) {
        Some((d, t)) => (d, Some(t.trim())),
        None => (body, None),
    };
    let mut dp = date.split('-');
    let year: i64 = dp.next()?.parse().ok()?;
    let year = if neg { year.checked_neg()? } else { year };
    let mut d = PartialDate {
        year,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        nanosecond: 0,
        resolution: Resolution::Year,
    };
    if let Some(m) = dp.next() {
        d.month = m.parse().ok()?;
        d.resolution = Resolution::Month;
    }
    if let Some(day) = dp.next() {
        d.day = day.parse().ok()?;
        d.resolution = Resolution::Day;
    }
    if dp.next().is_some() {
        return None;
    }
    if let Some(time) = time {
        if d.resolution != Resolution::Day {
            return None;
        }
        let mut tp = time.split(':');
        d.hour = tp.next()?.parse().ok()?;
        d.resolution = Resolution::Hour;
        if let Some(mi) = tp.next() {
            d.minute = mi.parse().ok()?;
            d.resolution = Resolution::Minute;
        }
        if let Some(sec) = tp.next() {
            let (whole, frac) = sec.split_once('.').unwrap_or((sec, ""));
            d.second = whole.parse().ok()?;
            d.resolution = Resolution::Second;
            if !frac.is_empty() {
                let digits: String = frac.chars().chain(std::iter::repeat('0')).take(9).collect();
                d.nanosecond = digits.parse().ok()?;
            }
        }
        if tp.next().is_some() {
            return None;
        }
    }
    let ok = (1..=12).contains(&d.month)
        && (1..=31).contains(&d.day)
        && d.hour < 24
        && d.minute < 60
        && d.second < 60;
    ok.then_some(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_dates() {
        let p = parse_partial_date("2020-01").unwrap();
        assert_eq!(
            (p.year, p.month, p.day, p.resolution),
            (2020, 1, 1, Resolution::Month)
        );
        let p = parse_partial_date("2020-01-15T06:30:00.25").unwrap();
        assert_eq!(
            (p.hour, p.minute, p.nanosecond, p.resolution),
            (6, 30, 250_000_000, Resolution::Second)
        );
        assert_eq!(parse_partial_date("-0044-03-15").unwrap().year, -44);
        assert_eq!(
            parse_partial_date("1999").unwrap().resolution,
            Resolution::Year
        );
        for bad in [
            "",
            "2020-13",
            "2020-01-01-01",
            "abc",
            "2020-01T05",
            "2020-01-01 25:00",
        ] {
            assert!(parse_partial_date(bad).is_none(), "{bad}");
        }
    }
}
