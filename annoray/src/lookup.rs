//! Resolving [`LabelIndexer`]s against a dimension coordinate, producing positional [`Indexer`]s.
//!
//! The coordinate's values are mapped into one of three key spaces (numbers as `f64`, strings,
//! or times as `i128` nanoseconds), and a single generic algorithm handles exact, nearest,
//! pad and backfill lookups and inclusive slices on ascending, descending or unsorted keys.

use std::cmp::Ordering;

use annoray_units::{Calendar, CfDate, Unit, UnitKind};
use chrono::{Datelike, NaiveDateTime, TimeDelta, Timelike};

use crate::label::{PartialDate, Resolution, parse_partial_date};
use crate::{
    CoreError, Indexer, Label, LabelIndexer, Method, NaT, Name, PySlice, Repr, Result, TypedArray,
    VariableBase,
};

const NS_PER_SEC: i128 = 1_000_000_000;
const NS_PER_DAY: i128 = 86_400 * NS_PER_SEC;

/// What a time key counts nanoseconds from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimeKind {
    DateTime,
    Cf(Calendar),
    Delta,
}

pub(crate) enum Keys {
    Num(Vec<f64>),
    Str(Vec<String>),
    Ns(Vec<i128>, TimeKind),
}

/// A query point, or the half-open range `[lo, hi)` a partial date string covers.
#[derive(Debug, Clone)]
enum Q<K> {
    Point(K),
    Range(K, K),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    Asc,
    Desc,
    Unsorted,
}

struct Ctx<'a> {
    dim: &'a Name,
    unit: Option<&'a Unit>,
}

impl Ctx<'_> {
    fn invalid(&self, reason: impl Into<String>) -> CoreError {
        CoreError::InvalidLabel {
            dim: self.dim.clone(),
            reason: reason.into(),
        }
    }
    fn not_found(&self, label: &Label) -> CoreError {
        CoreError::LabelNotFound {
            dim: self.dim.clone(),
            label: label.to_string(),
        }
    }
    fn overflow(&self) -> CoreError {
        self.invalid("label out of range")
    }
}

// ----- time conversions -----

fn datetime_ns(dt: &NaiveDateTime) -> Option<i128> {
    let days = i128::from(dt.date().num_days_from_ce());
    let tod = i128::from(dt.time().num_seconds_from_midnight())
        .checked_mul(NS_PER_SEC)?
        .checked_add(i128::from(dt.time().nanosecond()))?;
    days.checked_mul(NS_PER_DAY)?.checked_add(tod)
}

/// 1970-01-01 on the nanosecond scale of `kind`'s keys (time deltas: zero).
pub(crate) fn unix_origin_ns(kind: &TimeKind) -> Option<i128> {
    match kind {
        TimeKind::DateTime => {
            datetime_ns(&chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?.and_hms_opt(0, 0, 0)?)
        }
        TimeKind::Cf(_) | TimeKind::Delta => Some(0),
    }
}

fn cf_epoch(cal: Calendar) -> Option<CfDate> {
    CfDate::from_ymd(cal, 1970, 1, 1).ok()
}

fn cf_ns(d: &CfDate, cal: Calendar) -> Option<i128> {
    d.nanos_since(&cf_epoch(cal)?).ok()
}

fn delta_ns(td: &TimeDelta) -> Option<i128> {
    i128::from(td.num_seconds())
        .checked_mul(NS_PER_SEC)?
        .checked_add(i128::from(td.subsec_nanos()))
}

/// A date (and its period end, for partial strings) from `PartialDate` fields.
fn partial_to_chrono(p: &PartialDate) -> Option<Q<i128>> {
    let y = i32::try_from(p.year).ok()?;
    let start = chrono::NaiveDate::from_ymd_opt(y, u32::from(p.month), u32::from(p.day))?
        .and_hms_nano_opt(
            u32::from(p.hour),
            u32::from(p.minute),
            u32::from(p.second),
            p.nanosecond,
        )?;
    let end = match p.resolution {
        Resolution::Second => return Some(Q::Point(datetime_ns(&start)?)),
        Resolution::Year => {
            chrono::NaiveDate::from_ymd_opt(y.checked_add(1)?, 1, 1)?.and_hms_opt(0, 0, 0)?
        }
        Resolution::Month => {
            let (ny, nm) = if p.month == 12 {
                (y.checked_add(1)?, 1)
            } else {
                (y, u32::from(p.month).checked_add(1)?)
            };
            chrono::NaiveDate::from_ymd_opt(ny, nm, 1)?.and_hms_opt(0, 0, 0)?
        }
        Resolution::Day => start.checked_add_signed(TimeDelta::days(1))?,
        Resolution::Hour => start.checked_add_signed(TimeDelta::hours(1))?,
        Resolution::Minute => start.checked_add_signed(TimeDelta::minutes(1))?,
    };
    Some(Q::Range(datetime_ns(&start)?, datetime_ns(&end)?))
}

fn partial_to_cf(p: &PartialDate, cal: Calendar) -> Option<Q<i128>> {
    let start = CfDate::from_ymd(cal, p.year, p.month, p.day)
        .ok()?
        .and_hms_nano(p.hour, p.minute, p.second, p.nanosecond)
        .ok()?;
    let add = |ns: i128| start.checked_add_nanos(ns).ok();
    let end = match p.resolution {
        Resolution::Second => return Some(Q::Point(cf_ns(&start, cal)?)),
        Resolution::Year => {
            let ny = p.year.checked_add(1)?;
            let ny = if ny == 0 && !cal.has_year_zero() {
                1
            } else {
                ny
            };
            CfDate::from_ymd(cal, ny, 1, 1).ok()?
        }
        Resolution::Month => start.start_of_next_month().ok()?,
        Resolution::Day => add(NS_PER_DAY)?,
        Resolution::Hour => add(3600 * NS_PER_SEC)?,
        Resolution::Minute => add(60 * NS_PER_SEC)?,
    };
    Some(Q::Range(cf_ns(&start, cal)?, cf_ns(&end, cal)?))
}

// ----- coordinate → keys -----

#[allow(clippy::cast_precision_loss)] // 64-bit integer coordinates compare as f64
fn keys_of<S: Repr>(coord: &VariableBase<S>, ctx: &Ctx<'_>) -> Result<Keys> {
    if coord.ndim() != 1 {
        return Err(ctx.invalid("the dimension coordinate is not one-dimensional"));
    }
    let num = |it: &mut dyn Iterator<Item = f64>| Keys::Num(it.collect());
    let values = coord.typed()?;
    Ok(match &values {
        TypedArray::F32(a) => num(&mut a.iter().map(|v| f64::from(*v))),
        TypedArray::F64(a) => num(&mut a.iter().copied()),
        TypedArray::I8(a) => num(&mut a.iter().map(|v| f64::from(*v))),
        TypedArray::I16(a) => num(&mut a.iter().map(|v| f64::from(*v))),
        TypedArray::I32(a) => num(&mut a.iter().map(|v| f64::from(*v))),
        TypedArray::I64(a) => num(&mut a.iter().map(|v| *v as f64)),
        TypedArray::U8(a) => num(&mut a.iter().map(|v| f64::from(*v))),
        TypedArray::U16(a) => num(&mut a.iter().map(|v| f64::from(*v))),
        TypedArray::U32(a) => num(&mut a.iter().map(|v| f64::from(*v))),
        TypedArray::U64(a) => num(&mut a.iter().map(|v| *v as f64)),
        TypedArray::Str(a) => Keys::Str(a.iter().cloned().collect()),
        // NaT sorts first and matches only NaT.
        TypedArray::DateTime(a) => Keys::Ns(
            a.iter()
                .map(|d| {
                    if d.is_nat() {
                        Some(i128::MIN)
                    } else {
                        datetime_ns(d)
                    }
                })
                .collect::<Option<_>>()
                .ok_or_else(|| ctx.overflow())?,
            TimeKind::DateTime,
        ),
        TypedArray::CfDate(a) => {
            let cal = a
                .iter()
                .find(|d| !d.is_nat())
                .map_or(Calendar::Standard, CfDate::calendar);
            let ns = a
                .iter()
                .map(|d| {
                    if d.is_nat() {
                        Some(i128::MIN)
                    } else {
                        cf_ns(d, cal)
                    }
                })
                .collect::<Option<_>>();
            Keys::Ns(
                ns.ok_or_else(|| ctx.invalid("mixed calendars in coordinate"))?,
                TimeKind::Cf(cal),
            )
        }
        TypedArray::TimeDelta(a) => Keys::Ns(
            a.iter()
                .map(|d| {
                    if d.is_nat() {
                        Some(i128::MIN)
                    } else {
                        delta_ns(d)
                    }
                })
                .collect::<Option<_>>()
                .ok_or_else(|| ctx.overflow())?,
            TimeKind::Delta,
        ),
        TypedArray::Bool(_) => {
            return Err(ctx.invalid("boolean coordinates cannot be indexed by label"));
        }
    })
}

/// The labels of a dimension coordinate as comparable keys.
pub(crate) fn index_keys<S: Repr>(coord: &VariableBase<S>, dim: &Name) -> Result<Keys> {
    keys_of(
        coord,
        &Ctx {
            dim,
            unit: coord.unit(),
        },
    )
}

// ----- label → key -----

fn num_key(l: &Label, ctx: &Ctx<'_>, as_tolerance: bool) -> Result<f64> {
    #[allow(clippy::cast_precision_loss)]
    match l {
        Label::Int(v) => Ok(*v as f64),
        Label::Float(v) => Ok(*v),
        Label::Quantity(v, u) => {
            let target = ctx.unit.ok_or_else(|| {
                ctx.invalid(format!("label has unit {u} but the coordinate has none"))
            })?;
            // Tolerances are differences: convert the step size, never the offset.
            let (from, to) = if as_tolerance {
                (u.delta(), target.delta())
            } else {
                (u.clone(), target.clone())
            };
            Ok(from.converter_to(&to)?.apply(*v))
        }
        other => Err(ctx.invalid(format!("{other} is not a number"))),
    }
}

fn time_key(l: &Label, kind: TimeKind, ctx: &Ctx<'_>) -> Result<Q<i128>> {
    let ovf = || ctx.overflow();
    let point = |v: Option<i128>| v.map(Q::Point).ok_or_else(ovf);
    match (kind, l) {
        (TimeKind::DateTime, Label::DateTime(dt)) => point(datetime_ns(dt)),
        (TimeKind::DateTime, Label::CfDate(d)) => point(datetime_ns(&NaiveDateTime::try_from(*d)?)),
        (TimeKind::DateTime, Label::Str(s)) => parse_partial_date(s)
            .and_then(|p| partial_to_chrono(&p))
            .ok_or_else(|| ctx.invalid(format!("cannot parse date {s:?}"))),
        (TimeKind::Cf(cal), Label::CfDate(d)) => {
            if d.calendar() != cal {
                return Err(ctx.invalid(format!(
                    "label is in the {} calendar, coordinate in {cal}",
                    d.calendar()
                )));
            }
            point(cf_ns(d, cal))
        }
        (TimeKind::Cf(cal), Label::DateTime(dt)) => {
            point(cf_ns(&CfDate::from_chrono(*dt, cal)?, cal))
        }
        (TimeKind::Cf(cal), Label::Str(s)) => parse_partial_date(s)
            .and_then(|p| partial_to_cf(&p, cal))
            .ok_or_else(|| ctx.invalid(format!("cannot parse {s:?} as a {cal} date"))),
        (TimeKind::Delta, Label::TimeDelta(td)) => point(delta_ns(td)),
        (TimeKind::Delta, Label::Quantity(..)) => duration_ns(l, ctx).map(Q::Point),
        (_, other) => Err(ctx.invalid(format!(
            "{other} is not a valid label for a time coordinate"
        ))),
    }
}

/// A duration label (`TimeDelta`, or a quantity with a time unit) in nanoseconds.
fn duration_ns(l: &Label, ctx: &Ctx<'_>) -> Result<i128> {
    match l {
        Label::TimeDelta(td) => delta_ns(td).ok_or_else(|| ctx.overflow()),
        Label::Quantity(v, u) if *u.kind() == UnitKind::Linear => {
            let ns: Unit = "ns".parse()?;
            let x = u.converter_to(&ns)?.apply(*v).round();
            if !x.is_finite() || x.abs() >= 1e36 {
                return Err(ctx.overflow());
            }
            #[allow(clippy::cast_possible_truncation)] // range checked above
            Ok(x as i128)
        }
        other => Err(ctx.invalid(format!("{other} is not a duration"))),
    }
}

// ----- the generic algorithm -----

fn order<K: PartialOrd>(keys: &[K]) -> Order {
    let (mut asc, mut desc) = (true, true);
    for w in keys.windows(2) {
        if let [a, b] = w {
            match a.partial_cmp(b) {
                Some(Ordering::Less) => desc = false,
                Some(Ordering::Greater) => asc = false,
                Some(Ordering::Equal) => {}
                None => (asc, desc) = (false, false),
            }
        }
    }
    if asc {
        Order::Asc
    } else if desc {
        Order::Desc
    } else {
        Order::Unsorted
    }
}

struct Search<'k, K> {
    keys: &'k [K],
    order: Order,
}

impl<K: PartialOrd> Search<'_, K> {
    fn lt(&self, q: &K) -> usize {
        self.keys.partition_point(|k| k < q)
    }
    fn le(&self, q: &K) -> usize {
        self.keys.partition_point(|k| k <= q)
    }
    fn gt(&self, q: &K) -> usize {
        self.keys.partition_point(|k| k > q)
    }
    fn ge(&self, q: &K) -> usize {
        self.keys.partition_point(|k| k >= q)
    }

    fn exact(&self, q: &K) -> Option<usize> {
        let i = match self.order {
            Order::Asc => self.lt(q),
            Order::Desc => self.gt(q),
            Order::Unsorted => return self.keys.iter().position(|k| k == q),
        };
        (self.keys.get(i) == Some(q)).then_some(i)
    }

    fn nearest(&self, q: &K, dist: &dyn Fn(&K, &K) -> f64) -> Option<(usize, f64)> {
        let candidates: Vec<usize> = match self.order {
            Order::Asc | Order::Desc => {
                let i = if self.order == Order::Asc {
                    self.lt(q)
                } else {
                    self.gt(q)
                };
                i.checked_sub(1)
                    .into_iter()
                    .chain(std::iter::once(i))
                    .collect()
            }
            Order::Unsorted => (0..self.keys.len()).collect(),
        };
        candidates
            .into_iter()
            .filter_map(|i| self.keys.get(i).map(|k| (i, dist(k, q))))
            .fold(None, |best: Option<(usize, f64)>, (i, d)| match best {
                Some((_, bd)) if bd <= d => best,
                _ => Some((i, d)),
            })
    }

    /// pandas semantics: `pad` takes the previous position in index order (on a descending
    /// coordinate, the smallest value at or above `q`); `backfill` the next one.
    fn pad(&self, q: &K) -> Option<usize> {
        match self.order {
            Order::Asc => self.le(q).checked_sub(1),
            Order::Desc => self.ge(q).checked_sub(1),
            Order::Unsorted => None,
        }
    }

    fn backfill(&self, q: &K) -> Option<usize> {
        let i = match self.order {
            Order::Asc => self.lt(q),
            Order::Desc => self.gt(q),
            Order::Unsorted => return None,
        };
        (i < self.keys.len()).then_some(i)
    }

    /// Positions `[i0, i1)` covered by the half-open key range `[lo, hi)`.
    fn range(&self, lo: &K, hi: &K) -> Option<(usize, usize)> {
        match self.order {
            Order::Asc => Some((self.lt(lo), self.lt(hi))),
            Order::Desc => Some((self.ge(hi), self.ge(lo))),
            Order::Unsorted => None,
        }
    }

    fn slice_start(&self, q: &Q<K>) -> Option<usize> {
        match (self.order, q) {
            (Order::Asc, Q::Point(p) | Q::Range(p, _)) => Some(self.lt(p)),
            (Order::Desc, Q::Point(p)) => Some(self.gt(p)),
            (Order::Desc, Q::Range(_, hi)) => Some(self.ge(hi)),
            (Order::Unsorted, Q::Point(p)) => self.exact(p),
            (Order::Unsorted, Q::Range(..)) => None,
        }
    }

    fn slice_stop(&self, q: &Q<K>) -> Option<usize> {
        match (self.order, q) {
            (Order::Asc, Q::Point(p)) => Some(self.le(p)),
            (Order::Asc, Q::Range(_, hi)) => Some(self.lt(hi)),
            (Order::Desc, Q::Point(p)) => Some(self.ge(p)),
            (Order::Desc, Q::Range(lo, _)) => Some(self.ge(lo)),
            (Order::Unsorted, Q::Point(p)) => self.exact(p).and_then(|i| i.checked_add(1)),
            (Order::Unsorted, Q::Range(..)) => None,
        }
    }
}

fn to_isize(i: usize, ctx: &Ctx<'_>) -> Result<isize> {
    isize::try_from(i).map_err(|_| ctx.overflow())
}

/// The positional slice `[i0, i1)` with an optional step (negative steps walk it backwards).
fn positional_slice(i0: usize, i1: usize, step: Option<isize>, ctx: &Ctx<'_>) -> Result<Indexer> {
    let i1 = i1.max(i0);
    let s = match step {
        Some(st) if st < 0 => {
            if i1 == i0 {
                PySlice::new(Some(0), Some(0), None)
            } else {
                let last = to_isize(i1.saturating_sub(1), ctx)?;
                let before = i0.checked_sub(1).map(|b| to_isize(b, ctx)).transpose()?;
                PySlice::new(Some(last), before, Some(st))
            }
        }
        _ => PySlice::new(Some(to_isize(i0, ctx)?), Some(to_isize(i1, ctx)?), step),
    };
    Ok(Indexer::Slice(s))
}

fn resolve_keys<K: PartialOrd + std::fmt::Debug>(
    keys: &[K],
    ix: &LabelIndexer,
    to_key: &dyn Fn(&Label) -> Result<Q<K>>,
    tolerance: &dyn Fn(&Label) -> Result<f64>,
    dist: &dyn Fn(&K, &K) -> f64,
    ctx: &Ctx<'_>,
) -> Result<Indexer> {
    let s = Search {
        keys,
        order: order(keys),
    };
    let unsorted = || CoreError::NotMonotonic(ctx.dim.clone());
    match ix {
        LabelIndexer::Scalar {
            label,
            method,
            tolerance: tol,
        } => {
            let q = to_key(label)?;
            let not_found = || ctx.not_found(label);
            // A partial date covers a range of keys: select all of it (keeping the dimension).
            if let (Q::Range(lo, hi), Method::Exact) = (&q, method) {
                let (i0, i1) = s.range(lo, hi).ok_or_else(unsorted)?;
                if i1 <= i0 {
                    return Err(not_found());
                }
                return positional_slice(i0, i1, None, ctx);
            }
            let (Q::Point(p) | Q::Range(p, _)) = &q;
            let pos = match method {
                Method::Exact => s.exact(p),
                Method::Nearest => {
                    let tol = tol.as_ref().map(tolerance).transpose()?;
                    s.nearest(p, dist)
                        .filter(|(_, d)| tol.is_none_or(|t| *d <= t))
                        .map(|(i, _)| i)
                }
                Method::Pad | Method::Backfill if s.order == Order::Unsorted => {
                    return Err(unsorted());
                }
                Method::Pad => s.pad(p),
                Method::Backfill => s.backfill(p),
            };
            Ok(Indexer::Int(to_isize(pos.ok_or_else(not_found)?, ctx)?))
        }
        LabelIndexer::Slice { start, stop, step } => {
            let bound = |l: &Option<Label>,
                         f: &dyn Fn(&Q<K>) -> Option<usize>,
                         default: usize|
             -> Result<usize> {
                match l {
                    None => Ok(default),
                    Some(l) => {
                        let q = to_key(l)?;
                        f(&q).ok_or_else(|| {
                            if s.order == Order::Unsorted {
                                ctx.not_found(l)
                            } else {
                                unsorted()
                            }
                        })
                    }
                }
            };
            let i0 = bound(start, &|q| s.slice_start(q), 0)?;
            let i1 = bound(stop, &|q| s.slice_stop(q), keys.len())?;
            positional_slice(i0, i1, *step, ctx)
        }
    }
}

/// Translate a label indexer on `dim` into a positional one, using the dimension coordinate.
pub(crate) fn resolve<S: Repr>(
    coord: &VariableBase<S>,
    dim: &Name,
    ix: &LabelIndexer,
) -> Result<Indexer> {
    let ctx = Ctx {
        dim,
        unit: coord.unit(),
    };
    #[allow(clippy::cast_precision_loss)]
    let ns_dist = |a: &i128, b: &i128| a.abs_diff(*b) as f64;
    match keys_of(coord, &ctx)? {
        Keys::Num(k) => resolve_keys(
            &k,
            ix,
            &|l| num_key(l, &ctx, false).map(Q::Point),
            &|l| num_key(l, &ctx, true),
            &|a, b| (a - b).abs(),
            &ctx,
        ),
        Keys::Str(k) => resolve_keys(
            &k,
            ix,
            &|l| match l {
                Label::Str(s) => Ok(Q::Point(s.clone())),
                other => Err(ctx.invalid(format!("{other} is not a string"))),
            },
            &|_| Err(ctx.invalid("string coordinates have no distance")),
            &|a, b| if a == b { 0.0 } else { f64::INFINITY },
            &ctx,
        ),
        #[allow(clippy::cast_precision_loss)]
        Keys::Ns(k, kind) => resolve_keys(
            &k,
            ix,
            &|l| time_key(l, kind, &ctx),
            &|l| duration_ns(l, &ctx).map(|ns| ns as f64),
            &ns_dist,
            &ctx,
        ),
    }
}
