//! Converting a time coordinate to another CF calendar (xarray's `convert_calendar`); the
//! rules are on [`CalendarOptions`].

use annoray_units::{Calendar, CfDate};
use chrono::{Datelike, NaiveDateTime, Timelike};

use crate::{
    CoreError, DType, DataArray, DataArrayBase, Dataset, DatasetBase, Name, Repr, Result,
    TypedArray, Variable, VariableBase,
};

/// How dates move to or from a `360_day` calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlignOn {
    /// Field by field; dates that do not exist are dropped.
    Date,
    /// Keep the position within the year (day-of-year scaled).
    Year,
}

/// Options of [`DataArrayBase::convert_calendar`] (xarray's `convert_calendar`).
///
/// - Dates keep their fields (year, month, day, time of day); those that do not exist in the
///   target calendar are dropped with their data (Feb 29 going to `noleap`, the 31st going to
///   `360_day`), as are missing times (NaT).
/// - To or from `360_day` (between different calendars), [`AlignOn`] is required: `Date`
///   converts field by field as above; `Year` keeps the position within the year, mapping
///   day-of-year `d` of a year of `n` days to `round(d · m / n)` in the target's year of `m`
///   days (ties to even, as numpy), the first of any dates that coincide kept.
/// - `missing`: the result is reindexed onto a regular axis from its first to its last date,
///   with the source's (constant) step, the new positions filled with this value (integers
///   become `f64`).
/// - The result's times are `DateTime` for the standard and proleptic Gregorian calendars
///   when every date is representable (xarray's `datetime64` default), unless `use_cftime`;
///   `CfDate` otherwise.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CalendarOptions {
    /// The target calendar.
    pub calendar: Calendar,
    /// Required to or from `360_day`.
    pub align_on: Option<AlignOn>,
    /// `Some(true)`: `CfDate` times even for the standard calendars; `Some(false)`: plain
    /// `DateTime` times (the target must be standard or proleptic Gregorian).
    pub use_cftime: Option<bool>,
    /// Fill the gaps of a regular axis with this value.
    pub missing: Option<f64>,
}

impl CalendarOptions {
    /// Conversion to `calendar`, field by field, no fill.
    #[must_use]
    pub fn new(calendar: Calendar) -> Self {
        Self {
            calendar,
            align_on: None,
            use_cftime: None,
            missing: None,
        }
    }

    /// Set [`align_on`](Self::align_on).
    #[must_use]
    pub fn align_on(mut self, a: AlignOn) -> Self {
        self.align_on = Some(a);
        self
    }

    /// Set [`use_cftime`](Self::use_cftime).
    #[must_use]
    pub fn use_cftime(mut self, u: bool) -> Self {
        self.use_cftime = Some(u);
        self
    }

    /// Set [`missing`](Self::missing).
    #[must_use]
    pub fn missing(mut self, value: f64) -> Self {
        self.missing = Some(value);
        self
    }
}

/// A source time: its fields, in its calendar.
struct Fields {
    year: i64,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
    nano: u32,
}

fn bad(msg: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(msg.into())
}

fn chrono_fields(t: &NaiveDateTime) -> Result<Fields> {
    let to8 = |v: u32| u8::try_from(v).map_err(|_| CoreError::ArithmeticOverflow("date"));
    Ok(Fields {
        year: i64::from(t.year()),
        month: to8(t.month())?,
        day: to8(t.day())?,
        hour: to8(t.hour())?,
        minute: to8(t.minute())?,
        second: to8(t.second())?,
        nano: t.nanosecond(),
    })
}

fn cf_fields(t: &CfDate) -> Fields {
    Fields {
        year: t.year(),
        month: t.month(),
        day: t.day(),
        hour: t.hour(),
        minute: t.minute(),
        second: t.second(),
        nano: t.nanosecond(),
    }
}

/// The source times (`None` for NaT) and their calendar.
fn source_times<S: Repr>(coord: &VariableBase<S>) -> Result<(Vec<Option<Fields>>, Calendar)> {
    match coord.typed()? {
        TypedArray::DateTime(a) => Ok((
            a.iter()
                .map(|t| {
                    if crate::NaT::is_nat(t) {
                        Ok(None)
                    } else {
                        chrono_fields(t).map(Some)
                    }
                })
                .collect::<Result<_>>()?,
            Calendar::Standard,
        )),
        TypedArray::CfDate(a) => {
            let cal = a
                .iter()
                .find(|d| !d.is_nat())
                .map_or(Calendar::Standard, CfDate::calendar);
            Ok((
                a.iter()
                    .map(|t| (!t.is_nat()).then(|| cf_fields(t)))
                    .collect(),
                cal,
            ))
        }
        other => Err(CoreError::UnsupportedDType {
            op: "convert_calendar (needs times)",
            dtype: other.dtype(),
        }),
    }
}

fn days_in_year(cal: Calendar, year: i64) -> Result<i128> {
    (1..=12u8).try_fold(0_i128, |acc, m| {
        cal.days_in_month(year, m)
            .map(|d| acc.saturating_add(i128::from(d)))
            .ok_or_else(|| {
                bad(format!(
                    "year {year} does not exist in the {} calendar",
                    cal.name()
                ))
            })
    })
}

const NS_PER_DAY: i128 = 86_400_000_000_000;

/// `f` in `cal`, or `None` if that date does not exist there.
fn field_date(cal: Calendar, f: &Fields) -> Option<CfDate> {
    CfDate::from_ymd(cal, f.year, f.month, f.day)
        .and_then(|d| d.and_hms_nano(f.hour, f.minute, f.second, f.nano))
        .ok()
}

/// `f` (in `from`) at the same position of its year in `to`.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // day counts are small
fn year_date(from: Calendar, to: Calendar, f: &Fields) -> Result<Option<CfDate>> {
    let Some(src) = field_date(from, f) else {
        return Ok(None);
    };
    let jan1 = CfDate::from_ymd(from, f.year, 1, 1)?;
    let doy = src
        .nanos_since(&jan1)?
        .div_euclid(NS_PER_DAY)
        .saturating_add(1);
    let (n, m) = (days_in_year(from, f.year)?, days_in_year(to, f.year)?);
    let scaled = (m as f64 * doy as f64 / n as f64).round_ties_even() as i128;
    let day = scaled.clamp(1, m).saturating_sub(1);
    let start = CfDate::from_ymd(to, f.year, 1, 1)?;
    let tod = (src.nanos_since(&CfDate::from_ymd(from, f.year, f.month, f.day)?))?;
    Ok(Some(start.checked_add_nanos(
        day.saturating_mul(NS_PER_DAY).saturating_add(tod),
    )?))
}

/// The positions kept and their new dates, in order (for `Year` alignment: sorted, the
/// first of coinciding dates kept, as xarray's `np.unique`).
fn convert(
    fields: &[Option<Fields>],
    from: Calendar,
    opts: &CalendarOptions,
) -> Result<(Vec<usize>, Vec<CfDate>)> {
    let to = opts.calendar;
    let involves_360 = from != to && (from == Calendar::Day360 || to == Calendar::Day360);
    let align = match (involves_360, opts.align_on) {
        (true, None) => {
            return Err(bad(
                "converting to or from a 360_day calendar needs align_on (\"date\" or \"year\")",
            ));
        }
        (true, a) => a,
        (false, _) => None,
    };
    let mut kept: Vec<(usize, CfDate)> = Vec::new();
    for (i, f) in fields.iter().enumerate() {
        let Some(f) = f else { continue };
        let new = match align {
            Some(AlignOn::Year) => year_date(from, to, f)?,
            _ => field_date(to, f),
        };
        if let Some(d) = new {
            kept.push((i, d));
        }
    }
    if align == Some(AlignOn::Year) {
        let epoch = CfDate::from_ymd(to, 1970, 1, 1)?;
        let mut keyed = kept
            .into_iter()
            .map(|(i, d)| Ok((d.nanos_since(&epoch)?, i, d)))
            .collect::<Result<Vec<_>>>()?;
        keyed.sort_by_key(|(k, i, _)| (*k, *i));
        keyed.dedup_by_key(|(k, _, _)| *k);
        kept = keyed.into_iter().map(|(_, i, d)| (i, d)).collect();
    }
    Ok(kept.into_iter().unzip())
}

/// The new times as a variable along `dim`: plain datetimes where possible (see the module
/// notes), else CF dates.
fn time_variable(dim: &Name, dates: Vec<CfDate>, opts: &CalendarOptions) -> Result<Variable> {
    let standard = matches!(
        opts.calendar,
        Calendar::Standard | Calendar::ProlepticGregorian
    );
    if opts.use_cftime == Some(false) && !standard {
        return Err(bad(format!(
            "use_cftime=false needs the standard calendar, not {}",
            opts.calendar.name()
        )));
    }
    if standard && opts.use_cftime != Some(true) {
        let plain: std::result::Result<Vec<NaiveDateTime>, _> =
            dates.iter().map(|d| NaiveDateTime::try_from(*d)).collect();
        match plain {
            Ok(p) => {
                return Variable::from_typed(
                    [dim.clone()],
                    TypedArray::DateTime(ndarray::Array1::from(p).into_dyn()),
                );
            }
            Err(e) if opts.use_cftime == Some(false) => return Err(e.into()),
            Err(_) => {}
        }
    }
    Variable::from_typed(
        [dim.clone()],
        TypedArray::CfDate(ndarray::Array1::from(dates).into_dyn()),
    )
}

/// A regular axis in the target calendar from the first to the last of `dates`, with the
/// source's step.
fn full_axis(fields: &[Option<Fields>], from: Calendar, dates: &[CfDate]) -> Result<Vec<CfDate>> {
    let src: Vec<CfDate> = fields
        .iter()
        .flatten()
        .filter_map(|f| field_date(from, f))
        .collect();
    let steps: Vec<i128> = src
        .windows(2)
        .map(|w| match w {
            [a, b] => b.nanos_since(a).map_err(CoreError::from),
            _ => Ok(0),
        })
        .collect::<Result<_>>()?;
    let step = match steps.first() {
        Some(&s) if s > 0 && steps.iter().all(|x| *x == s) => s,
        _ => {
            return Err(bad(
                "missing= needs regularly spaced source times (a constant step)",
            ));
        }
    };
    let (Some(first), Some(last)) = (dates.first(), dates.last()) else {
        return Ok(Vec::new());
    };
    let mut out = vec![*first];
    let mut t = *first;
    while t.nanos_since(last)? < 0 {
        t = t.checked_add_nanos(step)?;
        out.push(t);
    }
    Ok(out)
}

/// What [`convert_calendar`](DataArrayBase::convert_calendar) needs of an object.
trait Calendared: Sized {
    fn time_coord(&self, dim: &str) -> Result<Variable>;
    fn taken(&self, dim: &str, keep: &[isize]) -> Result<Self>;
    fn with_time(self, dim: &Name, times: Variable) -> Result<Self>;
    fn reindexed(&self, dim: &str, labels: &Variable) -> Result<Self>;
    /// `self` where `present` (along `dim`), `fill` elsewhere.
    fn filled(&self, present: &DataArray, fill: f64) -> Result<Self>;
}

fn converted<T: Calendared>(obj: &T, dim: &str, opts: &CalendarOptions) -> Result<T> {
    let coord = obj.time_coord(dim)?;
    let (fields, from) = source_times(&coord)?;
    let (keep, dates) = convert(&fields, from, opts)?;
    let idx = keep
        .iter()
        .map(|i| isize::try_from(*i).map_err(|_| CoreError::ArithmeticOverflow("index")))
        .collect::<Result<Vec<_>>>()?;
    let name = Name::from(dim);
    let out = obj.taken(dim, &idx)?;
    let Some(fill) = opts.missing else {
        return out.with_time(&name, time_variable(&name, dates, opts)?);
    };
    let axis = full_axis(&fields, from, &dates)?;
    let present: Vec<bool> = {
        let have: std::collections::HashSet<CfDate> = dates.iter().copied().collect();
        axis.iter().map(|d| have.contains(d)).collect()
    };
    let out = out.with_time(&name, time_variable(&name, dates, opts)?)?;
    let labels = time_variable(&name, axis, opts)?;
    let re = out.reindexed(dim, &labels)?;
    let mask = DataArray::new(Variable::new(
        [name.clone()],
        ndarray::Array1::from(present).into_dyn(),
    )?)
    .with_coord(name, labels)?;
    re.filled(&mask, fill)
}

fn index_coord<S: Repr>(c: Option<&VariableBase<S>>, dim: &str) -> Result<Variable> {
    match c {
        Some(c) if c.dims().len() == 1 && c.dims().first().is_some_and(|d| d == dim) => {
            if !matches!(c.dtype(), DType::DateTime | DType::CfDate) {
                return Err(CoreError::UnsupportedDType {
                    op: "convert_calendar (needs times)",
                    dtype: c.dtype(),
                });
            }
            Ok(c.to_owned())
        }
        _ => Err(CoreError::NoIndex(Name::from(dim))),
    }
}

impl Calendared for DataArray {
    fn time_coord(&self, dim: &str) -> Result<Variable> {
        index_coord(self.coords().get(dim), dim)
    }
    fn taken(&self, dim: &str, keep: &[isize]) -> Result<Self> {
        self.take(dim, keep)
    }
    fn with_time(self, dim: &Name, times: Variable) -> Result<Self> {
        self.with_coord(dim.clone(), times)
    }
    fn reindexed(&self, dim: &str, labels: &Variable) -> Result<Self> {
        self.reindex(dim, labels)
    }
    fn filled(&self, present: &DataArray, fill: f64) -> Result<Self> {
        self.where_or(present, fill)
    }
}

impl Calendared for Dataset {
    fn time_coord(&self, dim: &str) -> Result<Variable> {
        index_coord(self.coords().get(dim), dim)
    }
    fn taken(&self, dim: &str, keep: &[isize]) -> Result<Self> {
        self.take(dim, keep)
    }
    fn with_time(mut self, dim: &Name, times: Variable) -> Result<Self> {
        self.set_coord_var(dim.clone(), times)?;
        Ok(self)
    }
    fn reindexed(&self, dim: &str, labels: &Variable) -> Result<Self> {
        self.reindex(dim, labels)
    }
    fn filled(&self, present: &DataArray, fill: f64) -> Result<Self> {
        let dim = present.dims().first().cloned();
        let mut out = self.clone();
        for (k, v) in self.data_vars() {
            if dim.as_ref().is_some_and(|d| v.dims().contains(d)) {
                let a = self.get(k)?.where_or(present, fill)?;
                out.set_array(k.clone(), a)?;
            }
        }
        Ok(out)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// The data with its time coordinate `dim` in another calendar; see the
    /// rules of [`CalendarOptions`].
    ///
    /// # Errors
    /// [`CoreError::NoIndex`] if `dim` has no time coordinate;
    /// [`CoreError::InvalidReduction`] when `align_on` is needed but missing, or `missing` is
    /// given for irregular times.
    pub fn convert_calendar(&self, dim: &str, opts: &CalendarOptions) -> Result<DataArray> {
        converted(&self.to_owned(), dim, opts)
    }
}

impl<S: Repr> DatasetBase<S> {
    /// Every variable along `dim` in another calendar; see
    /// [`DataArrayBase::convert_calendar`].
    ///
    /// # Errors
    /// As for [`DataArrayBase::convert_calendar`].
    pub fn convert_calendar(&self, dim: &str, opts: &CalendarOptions) -> Result<Dataset> {
        converted(&self.to_owned(), dim, opts)
    }
}
