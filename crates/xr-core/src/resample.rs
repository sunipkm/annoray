//! Resampling a time dimension into regular bins (xarray's `resample`), with pandas'
//! frequency strings.
//!
//! - **Fixed steps**: `ns`, `us`, `ms`, `s`, `min`, `h`, `D` (aliases `N`, `U`, `L`, `S`, `T`,
//!   `H`), with a multiple such as `"6h"`. Bins are closed on the left and labelled with their
//!   start, counted from midnight of the first day (pandas' `origin = "start_day"`).
//! - **Calendar periods starting** a month, quarter or year: `MS`, `QS[-MON]`, `YS[-MON]`
//!   (alias `AS`). Closed on the left, labelled with the start; quarters start in January,
//!   April, July and October unless anchored (`QS-FEB`).
//! - **Calendar periods ending** a week, month, quarter or year: `W[-DAY]` (weeks ending on
//!   Sunday by default), `ME` (alias `M`), `QE[-MON]` (alias `Q`), `YE[-MON]` (aliases `Y`,
//!   `A`). Labelled with the last day; every time on that day belongs to the bin, as pandas
//!   does for daily-or-coarser periods.
//!
//! Multiples of calendar periods count from the first period with data. Bins span from the
//! first to the last time, empty ones included. NaT belongs to no bin. Dates in CF calendars
//! (`CfDate`) resample in their own calendar; weeks need a real-world calendar.
//!
//! The result is a [`DataArrayGroupBy`] (or [`DatasetGroupBy`]) whose group dimension is the
//! resampled dimension itself, so `da.resample("time", "MS")?.mean()?` has a monthly `time`.

use std::str::FromStr;

use chrono::NaiveDateTime;
use ndarray::{ArrayD, IxDyn};
use xr_units::{Calendar, CfDate};

use crate::dataarray::DataArrayBase;
use crate::groupby::{DataArrayGroupBy, DatasetGroupBy, Grouper};
use crate::{CoreError, DatasetBase, NaT, Name, Repr, Result, TypedArray, Variable, VariableBase};

const NS_PER_DAY: i128 = 86_400_000_000_000;

fn invalid(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(reason.into())
}

/// What a frequency counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    /// A fixed number of nanoseconds.
    Tick(i128),
    /// Weeks ending on a weekday (0 = Monday).
    Week(u8),
    /// Periods of `step` months, the first starting in month `anchor` (0 = January).
    Start { step: i64, anchor: i64 },
    /// Periods of `step` months, the first ending in month `anchor` (0 = January).
    End { step: i64, anchor: i64 },
}

/// A resampling frequency: a multiple of a unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Freq {
    n: i64,
    unit: Unit,
}

const MONTHS: [&str; 12] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];
const DAYS: [&str; 7] = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"];

impl FromStr for Freq {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self> {
        let bad = || invalid(format!("unknown frequency {s:?}"));
        let digits = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
        let (num, rest) = s.split_at(digits);
        let n: i64 = if num.is_empty() {
            1
        } else {
            num.parse().map_err(|_| bad())?
        };
        if n <= 0 {
            return Err(bad());
        }
        let (base, anchor) = match rest.split_once('-') {
            Some((b, a)) => (b, Some(a)),
            None => (rest, None),
        };
        let month = |default: i64| -> Result<i64> {
            match anchor {
                None => Ok(default),
                Some(a) => MONTHS
                    .iter()
                    .position(|m| *m == a)
                    .and_then(|p| i64::try_from(p).ok())
                    .ok_or_else(bad),
            }
        };
        let tick = |ns: i128| -> Result<Unit> {
            if anchor.is_some() {
                return Err(bad());
            }
            Ok(Unit::Tick(ns))
        };
        let unit = match base {
            "ns" | "N" => tick(1)?,
            "us" | "U" => tick(1_000)?,
            "ms" | "L" => tick(1_000_000)?,
            "s" | "S" => tick(1_000_000_000)?,
            "min" | "T" => tick(60_000_000_000)?,
            "h" | "H" => tick(3_600_000_000_000)?,
            "D" => tick(NS_PER_DAY)?,
            "W" => {
                let day = match anchor {
                    None => 6,
                    Some(a) => DAYS.iter().position(|d| *d == a).ok_or_else(bad)?,
                };
                Unit::Week(u8::try_from(day).map_err(|_| bad())?)
            }
            "MS" if anchor.is_none() => Unit::Start { step: 1, anchor: 0 },
            "ME" | "M" if anchor.is_none() => Unit::End { step: 1, anchor: 0 },
            "QS" => Unit::Start {
                step: 3,
                anchor: month(0)?,
            },
            "QE" | "Q" => Unit::End {
                step: 3,
                anchor: month(11)?,
            },
            "YS" | "AS" => Unit::Start {
                step: 12,
                anchor: month(0)?,
            },
            "YE" | "Y" | "A" => Unit::End {
                step: 12,
                anchor: month(11)?,
            },
            _ => return Err(bad()),
        };
        Ok(Self { n, unit })
    }
}

fn overflow() -> CoreError {
    CoreError::ArithmeticOverflow("resample")
}

/// Months since year 0 (0 = January of year 0).
fn month_index(d: &CfDate) -> Result<i64> {
    d.year()
        .checked_mul(12)
        .and_then(|y| y.checked_add(i64::from(d.month())))
        .and_then(|m| m.checked_sub(1))
        .ok_or_else(overflow)
}

/// The month (1–12) of a month index.
fn month_of(m: i64) -> Result<u8> {
    u8::try_from(m.rem_euclid(12))
        .ok()
        .and_then(|x| x.checked_add(1))
        .ok_or_else(overflow)
}

fn month_start(cal: Calendar, m: i64) -> Result<CfDate> {
    let month = month_of(m)?;
    Ok(CfDate::from_ymd(cal, m.div_euclid(12), month, 1)?)
}

fn month_end(cal: Calendar, m: i64) -> Result<CfDate> {
    let year = m.div_euclid(12);
    let month = month_of(m)?;
    let last = cal.days_in_month(year, month).ok_or_else(overflow)?;
    Ok(CfDate::from_ymd(cal, year, month, last)?)
}

fn midnight(d: &CfDate) -> Result<CfDate> {
    Ok(CfDate::from_ymd(
        d.calendar(),
        d.year(),
        d.month(),
        d.day(),
    )?)
}

/// The period number of `d` for a calendar unit, and the label of a period number.
fn period(unit: Unit, d: &CfDate, week_ref: Option<&CfDate>) -> Result<i64> {
    match unit {
        Unit::Start { step, anchor } => Ok(month_index(d)?
            .checked_sub(anchor)
            .ok_or_else(overflow)?
            .div_euclid(step)),
        Unit::End { step, anchor } => {
            // The period whose last month is the first such month at or after `d`'s.
            let m = month_index(d)?.checked_sub(anchor).ok_or_else(overflow)?;
            Ok(step
                .checked_sub(1)
                .and_then(|s| m.checked_add(s))
                .ok_or_else(overflow)?
                .div_euclid(step))
        }
        Unit::Week(_) => {
            let r = week_ref.ok_or_else(overflow)?;
            let days = midnight(d)?.nanos_since(r)?.div_euclid(NS_PER_DAY);
            // Weeks end on the reference weekday: day 0 ends week 0, days 1..=7 end week 1.
            i64::try_from(days.checked_add(6).ok_or_else(overflow)?.div_euclid(7))
                .map_err(|_| overflow())
        }
        Unit::Tick(_) => Err(overflow()),
    }
}

fn period_label(unit: Unit, cal: Calendar, p: i64, week_ref: Option<&CfDate>) -> Result<CfDate> {
    match unit {
        Unit::Start { step, anchor } => month_start(cal, affine(p, step, anchor)?),
        Unit::End { step, anchor } => month_end(cal, affine(p, step, anchor)?),
        Unit::Week(_) => {
            let r = week_ref.ok_or_else(overflow)?;
            let ns = i128::from(p)
                .checked_mul(7 * NS_PER_DAY)
                .ok_or_else(overflow)?;
            Ok(r.checked_add_nanos(ns)?)
        }
        Unit::Tick(_) => Err(overflow()),
    }
}

/// A day with the given weekday (0 = Monday) in 1970, the reference for weeks.
fn week_reference(cal: Calendar, weekday: u8) -> Result<CfDate> {
    if !cal.is_real_world() {
        return Err(invalid("weekly resampling needs a real-world calendar"));
    }
    // 1970-01-05 was a Monday.
    let day = weekday.checked_add(5).ok_or_else(overflow)?;
    Ok(CfDate::from_ymd(cal, 1970, 1, day)?)
}

/// Bin labels and the bin of each time (`None` for NaT).
fn bins(dates: &[Option<CfDate>], freq: Freq) -> Result<(Vec<CfDate>, Vec<Option<usize>>)> {
    let Some(first) = dates.iter().flatten().next().copied() else {
        return Ok((Vec::new(), vec![None; dates.len()]));
    };
    let cal = first.calendar();
    let n = freq.n;
    let mut ks: Vec<Option<i64>> = Vec::with_capacity(dates.len());
    let label: Box<dyn Fn(i64) -> Result<CfDate>> = match freq.unit {
        Unit::Tick(ns) => {
            let origin = midnight(
                dates
                    .iter()
                    .flatten()
                    .min_by(|a, b| order(a, b))
                    .unwrap_or(&first),
            )?;
            let step = ns.checked_mul(i128::from(n)).ok_or_else(overflow)?;
            for d in dates {
                ks.push(match d {
                    Some(d) => Some(
                        i64::try_from(d.nanos_since(&origin)?.div_euclid(step))
                            .map_err(|_| overflow())?,
                    ),
                    None => None,
                });
            }
            Box::new(move |k| {
                Ok(origin
                    .checked_add_nanos(i128::from(k).checked_mul(step).ok_or_else(overflow)?)?)
            })
        }
        unit => {
            let week_ref = match unit {
                Unit::Week(day) => Some(week_reference(cal, day)?),
                _ => None,
            };
            let periods = dates
                .iter()
                .map(|d| {
                    d.as_ref()
                        .map(|d| period(unit, d, week_ref.as_ref()))
                        .transpose()
                })
                .collect::<Result<Vec<Option<i64>>>>()?;
            let p0 = periods.iter().flatten().min().copied().unwrap_or(0);
            let end = matches!(unit, Unit::End { .. } | Unit::Week(_));
            for p in &periods {
                ks.push(
                    p.map(|p| -> Result<i64> {
                        let off = p.checked_sub(p0).ok_or_else(overflow)?;
                        // Start periods group forward from the first; end periods group
                        // backward from each bin's last period.
                        Ok(if end {
                            n.checked_sub(1)
                                .and_then(|x| off.checked_add(x))
                                .ok_or_else(overflow)?
                                .div_euclid(n)
                        } else {
                            off.div_euclid(n)
                        })
                    })
                    .transpose()?,
                );
            }
            Box::new(move |k| period_label(unit, cal, affine(k, n, p0)?, week_ref.as_ref()))
        }
    };
    let kmax = ks.iter().flatten().max().copied().unwrap_or(0);
    let kmin = ks.iter().flatten().min().copied().unwrap_or(0);
    let labels = (kmin..=kmax).map(&label).collect::<Result<Vec<_>>>()?;
    let bin = ks
        .into_iter()
        .map(|k| k.and_then(|k| usize::try_from(k.checked_sub(kmin)?).ok()))
        .collect();
    Ok((labels, bin))
}

/// `x * a + b`, checked.
fn affine(x: i64, a: i64, b: i64) -> Result<i64> {
    x.checked_mul(a)
        .and_then(|v| v.checked_add(b))
        .ok_or_else(overflow)
}

fn order(a: &CfDate, b: &CfDate) -> std::cmp::Ordering {
    a.nanos_since(b)
        .map_or(std::cmp::Ordering::Equal, |d| d.cmp(&0))
}

/// The grouper resampling `coord` (dimension `dim`) at `freq`.
fn grouper<S: Repr>(coord: &VariableBase<S>, dim: &Name, freq: &str) -> Result<Grouper> {
    if coord.dims() != [dim.clone()] {
        return Err(invalid(format!("{dim:?} has no 1-D time coordinate")));
    }
    let freq: Freq = freq.parse()?;
    let typed = coord.typed()?;
    let (dates, chrono): (Vec<Option<CfDate>>, bool) = match &typed {
        TypedArray::DateTime(a) => (
            a.iter()
                .map(|d| {
                    if d.is_nat() {
                        Ok(None)
                    } else {
                        CfDate::from_chrono(*d, Calendar::ProlepticGregorian).map(Some)
                    }
                })
                .collect::<std::result::Result<_, _>>()?,
            true,
        ),
        TypedArray::CfDate(a) => (
            a.iter().map(|d| (!d.is_nat()).then_some(*d)).collect(),
            false,
        ),
        other => {
            return Err(CoreError::UnsupportedDType {
                op: "resample",
                dtype: other.dtype(),
            });
        }
    };
    let (labels, bin) = bins(&dates, freq)?;
    let mut groups = vec![Vec::new(); labels.len()];
    for (i, b) in bin.iter().enumerate() {
        if let Some(g) = b.and_then(|b| groups.get_mut(b)) {
            g.push(i);
        }
    }
    let shape = IxDyn(&[labels.len()]);
    let data = if chrono {
        let c = labels
            .iter()
            .map(|d| NaiveDateTime::try_from(*d))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        TypedArray::DateTime(ArrayD::from_shape_vec(shape, c).map_err(|_| overflow())?)
    } else {
        TypedArray::CfDate(ArrayD::from_shape_vec(shape, labels).map_err(|_| overflow())?)
    };
    let labels = Variable::from_typed([dim.clone()], data)?;
    Ok(Grouper::new(
        dim.clone(),
        dim.clone(),
        labels,
        groups,
        coord.to_owned(),
    ))
}

impl<S: Repr> DataArrayBase<S> {
    /// Resample the time dimension `dim` at `freq` (`"1D"`, `"6h"`, `"MS"`, `"QE-NOV"`, …; see
    /// the [module docs](crate::resample)). Reduce the result (`mean`, `sum`, …) to get one
    /// value per bin.
    ///
    /// # Errors
    /// [`CoreError::InvalidReduction`] for an unknown frequency or a dimension without a 1-D
    /// time coordinate; [`CoreError::UnsupportedDType`] for coordinates that are not dates.
    pub fn resample(&self, dim: &str, freq: &str) -> Result<DataArrayGroupBy<'_, S>> {
        let dim = Name::from(dim);
        let g = grouper(self.coord(&dim)?, &dim, freq)?;
        Ok(DataArrayGroupBy::from_grouper(self, g))
    }
}

impl<S: Repr> DatasetBase<S> {
    /// Resample the time dimension `dim` at `freq`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::resample`].
    pub fn resample(&self, dim: &str, freq: &str) -> Result<DatasetGroupBy<'_, S>> {
        let dim = Name::from(dim);
        let coord = self
            .coords()
            .get(&dim)
            .ok_or_else(|| CoreError::NoIndex(dim.clone()))?;
        let g = grouper(coord, &dim, freq)?;
        Ok(DatasetGroupBy::from_grouper(self, g))
    }
}
