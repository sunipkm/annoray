//! Filling missing values along a dimension: `ffill`, `bfill` (xarray's, bottleneck's
//! `push`) and `interpolate_na`.
//!
//! - `ffill`/`bfill` carry the last (next) valid value over NaN/NaT, at most `limit` steps
//!   into each run of missing values. Types without a missing value are returned unchanged.
//! - `interpolate_na` interpolates each lane through its valid points: `Linear`/`Slinear`,
//!   `Nearest` (ties to the lower point), `Zero` (the previous value), `Cubic` (scipy's
//!   not-a-knot cubic spline), `Pchip` (Fritsch–Butland, shape-preserving), `Akima` and
//!   `Makima`, matching scipy. Positions are the dimension's coordinate (numbers or times,
//!   strictly increasing), another coordinate along it, or `0..n`.
//! - `limit` fills at most the first `limit` values of each run of NaN; `max_gap` fills only
//!   runs whose bracketing valid points are at most that far apart (in the coordinate's
//!   unit: a number, a quantity, a time delta or a unit string such as `"3 h"`); leading and
//!   trailing runs measure from the first or last coordinate, as xarray.
//! - Outside the valid points nothing is filled unless `extrapolate` (xarray's
//!   `fill_value="extrapolate"`), which extends the edge interval's curve (the edge value
//!   for `Nearest`/`Zero`). Lanes with fewer than two valid points are left as they are.
//! - Units and attributes are kept; `f32` stays `f32`; integers, booleans and strings (no
//!   missing value) are returned unchanged; time data is refused.
//! - Deviations from xarray: `Cubic` leaves lanes with fewer than four valid points unfilled
//!   (scipy raises, failing the whole array); `max_gap` works without a coordinate (in
//!   positions), where xarray raises; a bare number for `max_gap` along a time coordinate is
//!   refused (xarray reads it as nanoseconds).
//! - Lazy data is read in slabs along another dimension.

use annoray_units::Unit;
use ndarray::{ArrayD, Axis};

use crate::calculus::{Origin, positions};
use crate::slabs::by_lanes;
use crate::{
    CoreError, DType, DataArray, DataArrayBase, Dataset, DatasetBase, Label, NaT, Name, Repr,
    Result, TypedArray, Variable,
};

fn invalid(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(reason.into())
}

/// Carry valid values over missing ones along `axis`, at most `limit` steps.
fn push<T: Clone>(
    a: &mut ArrayD<T>,
    axis: usize,
    limit: Option<usize>,
    backward: bool,
    missing: impl Fn(&T) -> bool,
) {
    for mut lane in a.lanes_mut(Axis(axis)) {
        let mut last: Option<T> = None;
        let mut run = 0usize;
        let mut step = |x: &mut T| {
            if missing(x) {
                run = run.saturating_add(1);
                if let Some(v) = &last
                    && limit.is_none_or(|l| run <= l)
                {
                    *x = v.clone();
                }
            } else {
                last = Some(x.clone());
                run = 0;
            }
        };
        if backward {
            lane.iter_mut().rev().for_each(&mut step);
        } else {
            lane.iter_mut().for_each(&mut step);
        }
    }
}

fn carry<S: Repr>(
    da: &DataArrayBase<S>,
    dim: &str,
    limit: Option<usize>,
    backward: bool,
) -> Result<DataArray> {
    if limit == Some(0) {
        return Err(invalid("limit must be at least 1 (or None)"));
    }
    da.variable().axis(dim)?;
    by_lanes(da, &Name::from(dim), |slab| {
        let v = slab.variable();
        let axis = v.axis(dim)?;
        let mut t = v.typed()?.to_owned();
        match &mut t {
            TypedArray::F64(a) => push(a, axis, limit, backward, |x: &f64| x.is_nan()),
            TypedArray::F32(a) => push(a, axis, limit, backward, |x: &f32| x.is_nan()),
            TypedArray::DateTime(a) => push(a, axis, limit, backward, NaT::is_nat),
            TypedArray::CfDate(a) => push(a, axis, limit, backward, NaT::is_nat),
            TypedArray::TimeDelta(a) => push(a, axis, limit, backward, NaT::is_nat),
            _ => {}
        }
        Ok(DataArrayBase::from_parts(
            slab.name().cloned(),
            like(v, t)?,
            slab.coords().clone(),
        ))
    })
}

/// A variable of `data` with `v`'s dimensions, unit and attributes.
fn like(v: &Variable, data: TypedArray<crate::Owned>) -> Result<Variable> {
    let mut out = Variable::from_typed(v.dims().to_vec(), data)?;
    out.set_unit(v.unit().cloned());
    out.attrs_mut()
        .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    Ok(out)
}

/// How [`interpolate_na`](DataArrayBase::interpolate_na) interpolates (scipy's names).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FillMethod {
    /// Straight lines (the default).
    #[default]
    Linear,
    /// The nearest valid value (the lower one on ties).
    Nearest,
    /// The previous valid value (a step function).
    Zero,
    /// Straight lines (scipy's first-order spline; the same as `Linear`).
    Slinear,
    /// A not-a-knot cubic spline (at least four valid points).
    Cubic,
    /// Piecewise cubic Hermite, shape-preserving (no overshoot).
    Pchip,
    /// Akima's piecewise cubic.
    Akima,
    /// The modified Akima piecewise cubic (less overshoot on flat stretches).
    Makima,
}

#[derive(Debug, Clone, Default)]
enum Positions {
    #[default]
    Dimension,
    Index,
    Coordinate(Name),
}

/// Options of [`interpolate_na`](DataArrayBase::interpolate_na), as xarray's keywords.
#[derive(Debug, Clone, Default)]
pub struct InterpNaOptions {
    method: FillMethod,
    limit: Option<usize>,
    max_gap: Option<Label>,
    positions: Positions,
    extrapolate: bool,
}

impl InterpNaOptions {
    /// Interpolate with `method` along the dimension's coordinate, filling every interior
    /// run of NaN.
    #[must_use]
    pub fn new(method: FillMethod) -> Self {
        Self {
            method,
            ..Self::default()
        }
    }

    /// Fill at most the first `limit` values of each run of NaN.
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Fill only runs whose bracketing valid points are at most `gap` apart: a number in the
    /// coordinate's unit, a [`Label::Quantity`], a [`Label::TimeDelta`] or a unit string.
    #[must_use]
    pub fn max_gap(mut self, gap: Label) -> Self {
        self.max_gap = Some(gap);
        self
    }

    /// Interpolate along positions `0..n` instead of a coordinate (`use_coordinate=False`).
    #[must_use]
    pub fn use_positions(mut self) -> Self {
        self.positions = Positions::Index;
        self
    }

    /// Interpolate along the coordinate `name` (one-dimensional, along the dimension).
    #[must_use]
    pub fn use_coordinate(mut self, name: impl Into<Name>) -> Self {
        self.positions = Positions::Coordinate(name.into());
        self
    }

    /// Extend the edge intervals' curves over leading and trailing NaN.
    #[must_use]
    pub fn extrapolate(mut self, extrapolate: bool) -> Self {
        self.extrapolate = extrapolate;
        self
    }
}

/// The interpolating curve through a lane's valid points.
enum Curve {
    Linear,
    Nearest,
    Zero,
    /// Second derivatives at the points.
    Spline(Vec<f64>),
    /// First derivatives at the points (cubic Hermite).
    Hermite(Vec<f64>),
}

struct Interp<'a> {
    xs: &'a [f64],
    ys: &'a [f64],
    curve: Curve,
}

fn steps(x: &[f64]) -> Vec<f64> {
    x.iter().zip(x.iter().skip(1)).map(|(a, b)| b - a).collect()
}

fn secants(x: &[f64], y: &[f64]) -> Vec<f64> {
    steps(y)
        .iter()
        .zip(steps(x))
        .map(|(dy, h)| dy / h)
        .collect()
}

/// numpy's `sign` (0 for 0).
fn sgn(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Second derivatives of the not-a-knot cubic spline (`n ≥ 4`).
fn not_a_knot(x: &[f64], y: &[f64]) -> Option<Vec<f64>> {
    let h = steps(x);
    let d = secants(x, y);
    let n = x.len();
    let k = n.checked_sub(2).filter(|k| *k >= 2)?;
    let (&h0, &h1) = (h.first()?, h.get(1)?);
    let (&ha, &hb) = (h.get(n.checked_sub(3)?)?, h.last()?);
    // Rows for M_1 … M_{n−2}: (sub, diag, super, rhs); the end rows absorb the not-a-knot
    // conditions on M_0 and M_{n−1}.
    let mut rows: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(k);
    for i in 1..=k {
        let (hl, hr) = (*h.get(i.checked_sub(1)?)?, *h.get(i)?);
        let r = 6.0 * (d.get(i)? - d.get(i.checked_sub(1)?)?);
        let (mut a, mut b, mut c) = (hl, 2.0 * (hl + hr), hr);
        if i == 1 {
            b += h0 * (h0 + h1) / h1;
            c -= h0 * h0 / h1;
            a = 0.0;
        }
        if i == k {
            a -= hb * hb / ha;
            b += hb * (ha + hb) / ha;
            c = 0.0;
        }
        rows.push((a, b, c, r));
    }
    // Thomas algorithm.
    let mut cp = Vec::with_capacity(k);
    let mut dp = Vec::with_capacity(k);
    let (mut c_prev, mut d_prev) = (0.0, 0.0);
    for (a, b, c, r) in &rows {
        let m = b - a * c_prev;
        c_prev = c / m;
        d_prev = (r - a * d_prev) / m;
        cp.push(c_prev);
        dp.push(d_prev);
    }
    let mut inner = vec![0.0; k];
    let mut next = 0.0;
    for ((slot, c), d) in inner.iter_mut().zip(&cp).zip(&dp).rev() {
        next = d - c * next;
        *slot = next;
    }
    let (&m1, &m2) = (inner.first()?, inner.get(1)?);
    let (&ma, &mb) = (inner.get(k.checked_sub(2)?)?, inner.last()?);
    let m0 = ((h0 + h1) * m1 - h0 * m2) / h1;
    let mn = ((ha + hb) * mb - hb * ma) / ha;
    Some(
        std::iter::once(m0)
            .chain(inner)
            .chain(std::iter::once(mn))
            .collect(),
    )
}

/// scipy's `PchipInterpolator` derivatives (`n ≥ 3`).
fn pchip_slopes(x: &[f64], y: &[f64]) -> Option<Vec<f64>> {
    let h = steps(x);
    let m = secants(x, y);
    let edge = |h0: f64, h1: f64, m0: f64, m1: f64| {
        let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
        if sgn(d) != sgn(m0) {
            0.0
        } else if sgn(m0) != sgn(m1) && d.abs() > 3.0 * m0.abs() {
            3.0 * m0
        } else {
            d
        }
    };
    let interior = h
        .windows(2)
        .zip(m.windows(2))
        .map(|(hw, mw)| match (hw, mw) {
            ([hl, hr], [ml, mr]) => {
                if sgn(*ml) != sgn(*mr) || *ml == 0.0 || *mr == 0.0 {
                    0.0
                } else {
                    let w1 = 2.0 * hr + hl;
                    let w2 = hr + 2.0 * hl;
                    (w1 + w2) / (w1 / ml + w2 / mr)
                }
            }
            _ => f64::NAN,
        });
    let first = edge(*h.first()?, *h.get(1)?, *m.first()?, *m.get(1)?);
    let n = h.len();
    let last = edge(
        *h.last()?,
        *h.get(n.checked_sub(2)?)?,
        *m.last()?,
        *m.get(n.checked_sub(2)?)?,
    );
    Some(
        std::iter::once(first)
            .chain(interior)
            .chain(std::iter::once(last))
            .collect(),
    )
}

/// scipy's `Akima1DInterpolator` derivatives (`n ≥ 3`), plain or modified.
fn akima_slopes(x: &[f64], y: &[f64], modified: bool) -> Option<Vec<f64>> {
    let s = secants(x, y);
    let (&s0, &s1) = (s.first()?, s.get(1)?);
    let (&sa, &sb) = (s.get(s.len().checked_sub(2)?)?, s.last()?);
    let l1 = 2.0 * s0 - s1;
    let l0 = 2.0 * l1 - s0;
    let r1 = 2.0 * sb - sa;
    let r2 = 2.0 * r1 - sb;
    let m: Vec<f64> = [l0, l1].into_iter().chain(s).chain([r1, r2]).collect();
    let dm: Vec<f64> = steps(&m).iter().map(|v| v.abs()).collect();
    let pm: Vec<f64> = m
        .iter()
        .zip(m.iter().skip(1))
        .map(|(a, b)| (a + b).abs())
        .collect();
    let weight = |j: usize| -> Option<f64> {
        let base = *dm.get(j)?;
        Some(if modified {
            base + 0.5 * pm.get(j)?
        } else {
            base
        })
    };
    let n = x.len();
    let f: Vec<(f64, f64)> = (0..n)
        .map(|i| Some((weight(i.checked_add(2)?)?, weight(i)?)))
        .collect::<Option<_>>()?;
    let top = f
        .iter()
        .map(|(a, b)| a + b)
        .fold(f64::NEG_INFINITY, f64::max);
    (0..n)
        .zip(&f)
        .map(|(i, (f1, f2))| {
            let f12 = f1 + f2;
            if f12 > 1e-9 * top {
                Some((f1 * m.get(i.checked_add(1)?)? + f2 * m.get(i.checked_add(2)?)?) / f12)
            } else {
                Some(0.5 * (m.get(i.checked_add(3)?)? + m.get(i)?))
            }
        })
        .collect()
}

impl<'a> Interp<'a> {
    /// The curve through `(xs, ys)` (`xs` increasing, at least two points); `None` when the
    /// method needs more points.
    fn new(method: FillMethod, xs: &'a [f64], ys: &'a [f64]) -> Option<Self> {
        let n = xs.len();
        if n < 2 {
            return None;
        }
        let curve = match method {
            FillMethod::Linear | FillMethod::Slinear => Curve::Linear,
            FillMethod::Nearest => Curve::Nearest,
            FillMethod::Zero => Curve::Zero,
            FillMethod::Cubic => Curve::Spline(not_a_knot(xs, ys)?),
            // Two points: a straight line, as scipy.
            FillMethod::Pchip | FillMethod::Akima | FillMethod::Makima if n == 2 => Curve::Linear,
            FillMethod::Pchip => Curve::Hermite(pchip_slopes(xs, ys)?),
            FillMethod::Akima => Curve::Hermite(akima_slopes(xs, ys, false)?),
            FillMethod::Makima => Curve::Hermite(akima_slopes(xs, ys, true)?),
        };
        Some(Self { xs, ys, curve })
    }

    /// The value at `t` (the edge interval's curve outside the points).
    fn at(&self, t: f64) -> f64 {
        let n = self.xs.len();
        let i = self
            .xs
            .partition_point(|v| *v <= t)
            .saturating_sub(1)
            .min(n.saturating_sub(2));
        let j = i.saturating_add(1);
        let (Some(&x0), Some(&x1), Some(&y0), Some(&y1)) = (
            self.xs.get(i),
            self.xs.get(j),
            self.ys.get(i),
            self.ys.get(j),
        ) else {
            return f64::NAN;
        };
        let h = x1 - x0;
        match &self.curve {
            Curve::Linear => y0 + (y1 - y0) * (t - x0) / h,
            Curve::Nearest => {
                if t - x0 <= x1 - t {
                    y0
                } else {
                    y1
                }
            }
            Curve::Zero => {
                if t >= x1 {
                    y1
                } else {
                    y0
                }
            }
            Curve::Spline(m) => {
                let (Some(&m0), Some(&m1)) = (m.get(i), m.get(j)) else {
                    return f64::NAN;
                };
                let (a, b) = (x1 - t, t - x0);
                m0 * a * a * a / (6.0 * h)
                    + m1 * b * b * b / (6.0 * h)
                    + (y0 / h - m0 * h / 6.0) * a
                    + (y1 / h - m1 * h / 6.0) * b
            }
            Curve::Hermite(d) => {
                let (Some(&d0), Some(&d1)) = (d.get(i), d.get(j)) else {
                    return f64::NAN;
                };
                let u = (t - x0) / h;
                let v = 1.0 - u;
                (1.0 + 2.0 * u) * v * v * y0
                    + u * v * v * h * d0
                    + u * u * (3.0 - 2.0 * u) * y1
                    + u * u * (u - 1.0) * h * d1
            }
        }
    }
}

/// What a lane needs besides its values.
struct Plan {
    x: Vec<f64>,
    method: FillMethod,
    limit: Option<usize>,
    max_gap: Option<f64>,
    extrapolate: bool,
}

impl Plan {
    fn lane(&self, y: &[f64]) -> Vec<f64> {
        let mut out = y.to_vec();
        let (xs, ys): (Vec<f64>, Vec<f64>) = self
            .x
            .iter()
            .zip(y)
            .filter(|(_, v)| !v.is_nan())
            .map(|(x, v)| (*x, *v))
            .unzip();
        if xs.len() == y.len() {
            return out;
        }
        let Some(curve) = Interp::new(self.method, &xs, &ys) else {
            return out;
        };
        // The next valid position after each one.
        let mut next = vec![None; y.len()];
        let mut seen = None;
        for (slot, (i, v)) in next.iter_mut().zip(y.iter().enumerate()).rev() {
            *slot = seen;
            if !v.is_nan() {
                seen = Some(i);
            }
        }
        let (x_first, x_last) = (
            self.x.first().copied().unwrap_or(f64::NAN),
            self.x.last().copied().unwrap_or(f64::NAN),
        );
        let mut prev: Option<usize> = None;
        let mut run = 0usize;
        for (i, (o, v)) in out.iter_mut().zip(y).enumerate() {
            if !v.is_nan() {
                prev = Some(i);
                run = 0;
                continue;
            }
            run = run.saturating_add(1);
            if self.limit.is_some_and(|l| run > l) {
                continue;
            }
            let after = next.get(i).copied().flatten();
            if let Some(g) = self.max_gap {
                let lo = prev.and_then(|p| self.x.get(p)).copied().unwrap_or(x_first);
                let hi = after.and_then(|q| self.x.get(q)).copied().unwrap_or(x_last);
                if hi - lo > g {
                    continue;
                }
            }
            if (prev.is_none() || after.is_none()) && !self.extrapolate {
                continue;
            }
            if let Some(t) = self.x.get(i) {
                *o = curve.at(*t);
            }
        }
        out
    }
}

/// `gap` in the unit of positions `unit` (`time`: a time coordinate, in seconds).
fn gap_in(gap: &Label, unit: Option<&Unit>, time: bool) -> Result<f64> {
    let quantity = |v: f64, u: &Unit| -> Result<f64> {
        let to = unit
            .ok_or_else(|| invalid(format!("max_gap has unit {u} but the positions have none")))?;
        Ok(u.delta().converter_to(&to.delta())?.apply(v))
    };
    #[allow(clippy::cast_precision_loss)]
    match gap {
        Label::Int(_) | Label::Float(_) if time => Err(invalid(
            "max_gap along a time coordinate needs a duration (a time delta, quantity or unit string)",
        )),
        Label::Int(v) => Ok(*v as f64),
        Label::Float(v) => Ok(*v),
        Label::Quantity(v, u) => quantity(*v, u),
        Label::TimeDelta(td) => {
            let secs = td.num_seconds() as f64 + f64::from(td.subsec_nanos()) * 1e-9;
            quantity(secs, &"s".parse::<Unit>()?)
        }
        Label::Str(s) => quantity(1.0, &s.parse::<Unit>()?),
        other => Err(invalid(format!("{other} is not a gap length"))),
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// Carry the last valid value forward over NaN/NaT along `dim`, at most `limit` steps
    /// into each run (all of it with `None`).
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::InvalidReduction`] for `limit` 0.
    pub fn ffill(&self, dim: &str, limit: Option<usize>) -> Result<DataArray> {
        carry(self, dim, limit, false)
    }

    /// Carry the next valid value backward over NaN/NaT along `dim`; see
    /// [`ffill`](Self::ffill).
    ///
    /// # Errors
    /// As for [`ffill`](Self::ffill).
    pub fn bfill(&self, dim: &str, limit: Option<usize>) -> Result<DataArray> {
        carry(self, dim, limit, true)
    }

    /// Fill NaN along `dim` by interpolating through the valid points; see the module notes.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::InvalidReduction`] for positions that are not
    /// strictly increasing, a coordinate not along `dim`, or a `max_gap` of the wrong kind;
    /// unit errors; [`CoreError::UnsupportedDType`] for time data.
    pub fn interpolate_na(&self, dim: &str, opts: &InterpNaOptions) -> Result<DataArray> {
        let n = self.variable().len_of(dim)?;
        match self.variable().dtype() {
            DType::F64 | DType::F32 => {}
            d @ (DType::DateTime | DType::CfDate | DType::TimeDelta) => {
                return Err(CoreError::UnsupportedDType {
                    op: "interpolate_na",
                    dtype: d,
                });
            }
            _ => return Ok(self.to_owned()),
        }
        let name = match &opts.positions {
            Positions::Dimension if self.coords().contains_key(dim) => Some(Name::from(dim)),
            Positions::Dimension | Positions::Index => None,
            Positions::Coordinate(c) => Some(c.clone()),
        };
        let (x, unit, time) = match &name {
            Some(c) => {
                let coord = self.coords().get(c);
                let time = coord.is_some_and(|v| {
                    matches!(
                        v.dtype(),
                        DType::DateTime | DType::CfDate | DType::TimeDelta
                    )
                });
                let p = positions(c, coord, None, Origin::First)?;
                if p.dim != dim {
                    return Err(invalid(format!("{c:?} is not a coordinate along {dim:?}")));
                }
                if !p.x.windows(2).all(|w| matches!(w, [a, b] if a < b)) {
                    return Err(invalid(format!(
                        "{c:?} must be strictly increasing (without NaN) to interpolate along it"
                    )));
                }
                (p.x, p.unit, time)
            }
            #[allow(clippy::cast_precision_loss)]
            None => ((0..n).map(|i| i as f64).collect(), None, false),
        };
        let max_gap = match &opts.max_gap {
            Some(g) => Some(gap_in(g, unit.as_ref(), time)?),
            None => None,
        };
        let plan = Plan {
            x,
            method: opts.method,
            limit: opts.limit,
            max_gap,
            extrapolate: opts.extrapolate,
        };
        if plan.limit == Some(0) {
            return Err(invalid("limit must be at least 1 (or None)"));
        }
        by_lanes(self, &Name::from(dim), |slab| {
            let v = slab.variable();
            let axis = v.axis(dim)?;
            let typed = v.typed()?;
            let f32 = typed.dtype() == DType::F32;
            let mut data = crate::ops::to_f64(&typed, None)?;
            for mut lane in data.lanes_mut(Axis(axis)) {
                let filled = plan.lane(&lane.to_vec());
                lane.iter_mut().zip(filled).for_each(|(d, f)| *d = f);
            }
            #[allow(clippy::cast_possible_truncation)] // f32 data stays f32
            let t = if f32 {
                TypedArray::F32(data.mapv(|x| x as f32))
            } else {
                TypedArray::F64(data)
            };
            Ok(DataArrayBase::from_parts(
                slab.name().cloned(),
                like(v, t)?,
                slab.coords().clone(),
            ))
        })
    }
}

impl<S: Repr> DatasetBase<S> {
    /// [`DataArrayBase::ffill`] of every data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::ffill`].
    pub fn ffill(&self, dim: &str, limit: Option<usize>) -> Result<Dataset> {
        crate::reshape::per_variable(self, dim, |da| da.ffill(dim, limit))
    }

    /// [`DataArrayBase::bfill`] of every data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::bfill`].
    pub fn bfill(&self, dim: &str, limit: Option<usize>) -> Result<Dataset> {
        crate::reshape::per_variable(self, dim, |da| da.bfill(dim, limit))
    }

    /// [`DataArrayBase::interpolate_na`] of every data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::interpolate_na`].
    pub fn interpolate_na(&self, dim: &str, opts: &InterpNaOptions) -> Result<Dataset> {
        crate::reshape::per_variable(self, dim, |da| da.interpolate_na(dim, opts))
    }
}
