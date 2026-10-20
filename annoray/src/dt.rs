//! Date-time components: xarray's `.dt` accessor, and the `"time.month"`-style virtual
//! variables `groupby` accepts.

use std::str::FromStr;

use annoray_units::{CfDate, Disambiguation, Instant, TimeZone};
use ndarray::{ArrayD, IxDyn};

use crate::{
    CoreError, DType, DataArray, DataArrayBase, NaT, Owned, Repr, Result, TypedArray, Variable,
    VariableBase,
};

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
                // 1970-01-01 was a Thursday (Monday = 0).
                let days = self.unix_nanos().ok()?.div_euclid(86_400_000_000_000);
                i64::try_from(days.checked_add(3)?.rem_euclid(7)).ok()?
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
        TypedArray::DateTime(a) => {
            // The wall-clock fields of each instant in the variable's zone (UTC by default).
            let zone = v.time_zone();
            let fields = a
                .iter()
                .map(|t| {
                    if t.is_nat() {
                        Ok(CfDate::NAT)
                    } else {
                        t.to_local(zone.map_or(0, |z| z.offset_at(*t)))
                    }
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let fields = ArrayD::from_shape_vec(IxDyn(a.shape()), fields)
                .map_err(|_| CoreError::InvalidReduction("shape".into()))?;
            go(&fields.view(), c)
        }
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

fn not_datetimes(op: &'static str, dtype: DType) -> CoreError {
    CoreError::UnsupportedDType { op, dtype }
}

impl<S: Repr> VariableBase<S> {
    /// The same instants with their wall-clock times in `tz` (`None`: UTC), as pandas'
    /// `tz_convert`. Only how the times read changes: their components, resampling bins and
    /// display.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for data that are not date-times.
    pub fn tz_convert(&self, tz: Option<TimeZone>) -> Result<Variable> {
        if self.dtype() != DType::DateTime {
            return Err(not_datetimes("tz_convert", self.dtype()));
        }
        let mut out = self.to_owned();
        out.set_time_zone(tz)?;
        Ok(out)
    }

    /// The instants at which clocks in `tz` (`None`: UTC) show the same wall-clock times as
    /// these do in their own zone, as pandas' `tz_localize` (applied to times read as UTC
    /// that were local times). Times that `tz` skips or repeats are resolved by `how`.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for data that are not date-times; a
    /// [`UnitError::LocalTime`](annoray_units::UnitError::LocalTime) for a skipped or
    /// repeated time with [`Disambiguation::Raise`].
    pub fn tz_localize(&self, tz: Option<TimeZone>, how: Disambiguation) -> Result<Variable> {
        let TypedArray::DateTime(a) = self.typed()? else {
            return Err(not_datetimes("tz_localize", self.dtype()));
        };
        let from = self.time_zone();
        let moved = a
            .iter()
            .map(|t| {
                if t.is_nat() {
                    return Ok(Instant::NAT);
                }
                let wall = t.to_local(from.map_or(0, |z| z.offset_at(*t)))?;
                match &tz {
                    Some(z) => z.instant_of(&wall, how),
                    None => Instant::try_from(wall),
                }
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let data = ArrayD::from_shape_vec(IxDyn(a.shape()), moved)
            .map_err(|_| CoreError::InvalidReduction("shape".into()))?;
        let mut out = self.to_owned();
        out.data_mut::<Instant>()?.assign(&data);
        out.set_time_zone(tz)?;
        Ok(out)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// The time zone of date-time data, if one is set.
    #[must_use]
    pub fn time_zone(&self) -> Option<&TimeZone> {
        self.variable().time_zone()
    }

    /// The data's instants with their wall-clock times in `tz`; see
    /// [`VariableBase::tz_convert`].
    ///
    /// # Errors
    /// As for [`VariableBase::tz_convert`].
    pub fn tz_convert(&self, tz: Option<TimeZone>) -> Result<DataArray> {
        let var = self.variable().tz_convert(tz)?;
        let (name, _, coords) = self.to_owned().into_parts();
        Ok(DataArray::from_parts(name, var, coords))
    }

    /// The data's wall-clock times read in `tz`; see [`VariableBase::tz_localize`].
    ///
    /// # Errors
    /// As for [`VariableBase::tz_localize`].
    pub fn tz_localize(&self, tz: Option<TimeZone>, how: Disambiguation) -> Result<DataArray> {
        let var = self.variable().tz_localize(tz, how)?;
        let (name, _, coords) = self.to_owned().into_parts();
        Ok(DataArray::from_parts(name, var, coords))
    }
}
