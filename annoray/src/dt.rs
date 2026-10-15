//! Date-time components: xarray's `.dt` accessor, and the `"time.month"`-style virtual
//! variables `groupby` accepts.

use std::str::FromStr;

use annoray_units::CfDate;
use chrono::{Datelike, NaiveDateTime, Timelike};
use ndarray::{ArrayD, IxDyn};

use crate::{CoreError, NaT, Owned, Repr, Result, TypedArray, Variable, VariableBase};

/// A component of a date.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DtComponent {
    /// The year.
    Year,
    /// The month, 1–12.
    Month,
    /// The day of the month, from 1.
    Day,
    /// The hour, 0–23.
    Hour,
    /// The minute, 0–59.
    Minute,
    /// The second, 0–59.
    Second,
    /// The day of the year, from 1.
    DayOfYear,
    /// The day of the week, Monday = 0 (real-world calendars only).
    DayOfWeek,
    /// The meteorological season: `"DJF"`, `"MAM"`, `"JJA"` or `"SON"`.
    Season,
}

impl FromStr for DtComponent {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "year" => Self::Year,
            "month" => Self::Month,
            "day" => Self::Day,
            "hour" => Self::Hour,
            "minute" => Self::Minute,
            "second" => Self::Second,
            "dayofyear" => Self::DayOfYear,
            "dayofweek" | "weekday" => Self::DayOfWeek,
            "season" => Self::Season,
            _ => {
                return Err(CoreError::InvalidReduction(format!(
                    "unknown date-time component {s:?}"
                )));
            }
        })
    }
}

fn season(month: u32) -> &'static str {
    match month {
        3..=5 => "MAM",
        6..=8 => "JJA",
        9..=11 => "SON",
        _ => "DJF",
    }
}

/// The number of a component, for dates that are not NaT.
trait Parts {
    fn number(&self, c: DtComponent) -> Option<i64>;
    fn month_number(&self) -> u32;
}

impl Parts for NaiveDateTime {
    fn number(&self, c: DtComponent) -> Option<i64> {
        Some(match c {
            DtComponent::Year => i64::from(self.year()),
            DtComponent::Month => i64::from(self.month()),
            DtComponent::Day => i64::from(self.day()),
            DtComponent::Hour => i64::from(self.hour()),
            DtComponent::Minute => i64::from(self.minute()),
            DtComponent::Second => i64::from(self.second()),
            DtComponent::DayOfYear => i64::from(self.ordinal()),
            DtComponent::DayOfWeek => i64::from(self.weekday().num_days_from_monday()),
            DtComponent::Season => return None,
        })
    }

    fn month_number(&self) -> u32 {
        self.month()
    }
}

impl Parts for CfDate {
    fn number(&self, c: DtComponent) -> Option<i64> {
        Some(match c {
            DtComponent::Year => self.year(),
            DtComponent::Month => i64::from(self.month()),
            DtComponent::Day => i64::from(self.day()),
            DtComponent::Hour => i64::from(self.hour()),
            DtComponent::Minute => i64::from(self.minute()),
            DtComponent::Second => i64::from(self.second()),
            DtComponent::DayOfYear => {
                let cal = self.calendar();
                let before: i64 = (1..self.month())
                    .map(|m| i64::from(cal.days_in_month(self.year(), m).unwrap_or(0)))
                    .sum();
                before.saturating_add(i64::from(self.day()))
            }
            DtComponent::DayOfWeek => {
                if !self.calendar().is_real_world() {
                    return None;
                }
                let dt = NaiveDateTime::try_from(*self).ok()?;
                i64::from(dt.weekday().num_days_from_monday())
            }
            DtComponent::Season => return None,
        })
    }

    fn month_number(&self) -> u32 {
        u32::from(self.month())
    }
}

/// A component of every date, with a mask of the NaT positions. Numbers are `i64` (0 where
/// NaT); seasons are strings (empty where NaT).
pub(crate) fn components<S: Repr>(
    v: &VariableBase<S>,
    c: DtComponent,
) -> Result<(TypedArray<Owned>, Vec<bool>)> {
    fn go<T: Parts + NaT>(
        a: &ndarray::ArrayViewD<'_, T>,
        c: DtComponent,
    ) -> Result<(TypedArray<Owned>, Vec<bool>)> {
        let shape = IxDyn(a.shape());
        let missing: Vec<bool> = a.iter().map(NaT::is_nat).collect();
        let bad = || CoreError::InvalidReduction(format!("{c:?} is not defined here"));
        let data = if c == DtComponent::Season {
            let s: Vec<String> = a
                .iter()
                .map(|d| {
                    if d.is_nat() {
                        String::new()
                    } else {
                        season(d.month_number()).to_owned()
                    }
                })
                .collect();
            TypedArray::Str(ArrayD::from_shape_vec(shape, s).map_err(|_| bad())?)
        } else {
            let n = a
                .iter()
                .map(|d| if d.is_nat() { Some(0) } else { d.number(c) })
                .collect::<Option<Vec<i64>>>()
                .ok_or_else(bad)?;
            TypedArray::I64(ArrayD::from_shape_vec(shape, n).map_err(|_| bad())?)
        };
        Ok((data, missing))
    }
    match v.typed()? {
        TypedArray::DateTime(a) => go(&a, c),
        TypedArray::CfDate(a) => go(&a, c),
        other => Err(CoreError::UnsupportedDType {
            op: "date-time components",
            dtype: other.dtype(),
        }),
    }
}

impl<S: Repr> VariableBase<S> {
    /// A component of each date (xarray's `.dt.<component>`): `"year"`, `"month"`, `"day"`,
    /// `"hour"`, `"minute"`, `"second"`, `"dayofyear"`, `"dayofweek"` or `"season"`.
    /// Numbers are `i64`, or `f64` with NaN if there is NaT; seasons are strings (empty for
    /// NaT).
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for data that are not dates;
    /// [`CoreError::InvalidReduction`] for an unknown component, or the day of the week in an
    /// idealized calendar.
    pub fn dt(&self, component: &str) -> Result<Variable> {
        let (data, missing) = components(self, component.parse()?)?;
        let data = match data {
            TypedArray::I64(a) if missing.iter().any(|m| *m) => {
                #[allow(clippy::cast_precision_loss)] // date components are small
                let f: Vec<f64> = a
                    .iter()
                    .zip(&missing)
                    .map(|(v, m)| if *m { f64::NAN } else { *v as f64 })
                    .collect();
                TypedArray::F64(
                    ArrayD::from_shape_vec(a.raw_dim(), f)
                        .map_err(|_| CoreError::InvalidReduction("shape".into()))?,
                )
            }
            other => other,
        };
        Variable::from_typed(self.dims().iter().cloned(), data)
    }
}
