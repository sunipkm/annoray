//! Exponentially weighted moving statistics along a dimension (xarray's `rolling_exp`,
//! numbagg's `move_exp_nan*`).
//!
//! Each step decays every earlier weight by `1 − α` and gives the new value weight 1; missing
//! values (NaN) are skipped, their positions still decaying the others. `α` comes from the
//! window as in pandas: `span` → `2 / (span + 1)`, `com` → `1 / (1 + com)`, `halflife` →
//! `1 − exp(ln ½ / halflife)`, or `α` itself. A result is missing until the decayed sum of
//! the `α` of the values seen reaches `min_weight` (numbagg's rule).
//!
//! - `mean`, `sum`, `count` follow numbagg's recurrences.
//! - `var`, `std`, `cov`, `corr` are numbagg's estimators (the weighted variance with the
//!   bias correction `1 − Σw² / (Σw)²`), computed on values shifted by the lane's first
//!   valid value (exact for nearby values, Sterbenz's lemma) with West's incremental
//!   updates of the weighted mean and of the sum of squared deviations — not as
//!   `E[x²] − E[x]²`, which cancels catastrophically when values are large against their
//!   spread (temperatures in K, timestamps): there numbagg loses every digit (relative
//!   errors of 10³ at an offset of 10⁶ times the spread), while these stay at rounding
//!   level. Equal in exact arithmetic.
//! - Arithmetic is in `f64` (results in `f32` for `f32` data).
//!
//! Units: `mean`, `std` keep the unit (`std` of °C is a difference, in K); `var` is its
//! square; `sum` refuses units that cannot be added (°C); `count` and `corr` have none;
//! `cov` is the product of the two (difference) units. Lazy data is read in slabs (whole
//! lanes along the dimension).

use ndarray::{ArrayD, ArrayViewMut1, Axis};

use crate::reduce::reduced_unit;
use crate::{
    CoreError, DataArray, DataArrayBase, Name, Reduction, Repr, Result, TypedArray, Variable,
};

/// How the decay of [`DataArrayBase::rolling_exp`] is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WindowType {
    /// `α = 2 / (span + 1)`, `span ≥ 1`.
    Span,
    /// Centre of mass: `α = 1 / (1 + com)`, `com ≥ 0`.
    Com,
    /// `α = 1 − exp(ln ½ / halflife)`, `halflife > 0`.
    Halflife,
    /// `α` itself, `0 < α ≤ 1`.
    Alpha,
}

fn alpha_of(window: f64, kind: WindowType) -> Result<f64> {
    let bad = |what: &str| CoreError::InvalidReduction(format!("rolling_exp: {what}"));
    let a = match kind {
        WindowType::Span if window >= 1.0 => 2.0 / (window + 1.0),
        WindowType::Span => return Err(bad("span must be at least 1")),
        WindowType::Com if window >= 0.0 => 1.0 / (1.0 + window),
        WindowType::Com => return Err(bad("com must be at least 0")),
        WindowType::Halflife if window > 0.0 => 1.0 - (0.5_f64.ln() / window).exp(),
        WindowType::Halflife => return Err(bad("halflife must be positive")),
        WindowType::Alpha if window > 0.0 && window <= 1.0 => window,
        WindowType::Alpha => return Err(bad("alpha must be in (0, 1]")),
    };
    if a.is_finite() {
        Ok(a)
    } else {
        Err(bad("the window gives no finite decay"))
    }
}

/// Exponentially weighted moving statistics of a data array; see the module notes.
#[derive(Debug)]
pub struct RollingExp<'a, S: Repr> {
    da: &'a DataArrayBase<S>,
    dim: Name,
    alpha: f64,
    min_weight: f64,
}

/// The statistic of one lane.
#[derive(Debug, Clone, Copy)]
enum Stat {
    Mean,
    Sum,
    Count,
    Var,
    Std,
}

/// The running state of one lane (numbagg's recurrences; West's updates, on values less the
/// lane's first valid one, for the spread).
#[derive(Clone, Copy, Default)]
struct State {
    /// Decayed sum of α (numbagg's `weight`, for `min_weight`).
    weight: f64,
    /// Decayed count, and sum of squared weights.
    w: f64,
    w2: f64,
    /// Weighted mean and sum of squared deviations, of values less `shift`.
    mean: f64,
    m2: f64,
    /// Decayed sum (of the values themselves).
    sum: f64,
    /// The lane's first valid value; `seen` once there is one.
    shift: f64,
    seen: bool,
}

impl State {
    #[inline]
    fn step(&mut self, raw: f64, alpha: f64, decay: f64) {
        self.weight *= decay;
        self.w *= decay;
        self.w2 *= decay * decay;
        self.m2 *= decay;
        self.sum *= decay;
        if raw.is_nan() {
            return;
        }
        if !self.seen {
            self.seen = true;
            self.shift = raw;
        }
        let a = raw - self.shift;
        self.weight += alpha;
        self.sum += raw;
        let w_new = self.w + 1.0;
        let delta = a - self.mean;
        self.mean += delta / w_new;
        self.m2 += delta * (a - self.mean);
        self.w = w_new;
        self.w2 += 1.0;
    }

    #[inline]
    fn value(&self, stat: Stat, min_weight: f64) -> f64 {
        let ok = self.weight >= min_weight;
        let bias = if self.w > 0.0 {
            1.0 - self.w2 / (self.w * self.w)
        } else {
            0.0
        };
        match stat {
            Stat::Mean if ok && self.w > 0.0 => self.mean + self.shift,
            Stat::Sum if ok && self.seen => self.sum,
            Stat::Count if ok => self.w,
            Stat::Var if ok && bias > 0.0 => (self.m2 / self.w) / bias,
            Stat::Std if ok && bias > 0.0 => ((self.m2 / self.w) / bias).max(0.0).sqrt(),
            _ => f64::NAN,
        }
    }
}

/// Columns per parallel task.
const BLOCK: usize = 512;

/// `f` over rows of `x` along `axis` (the axis moved first, so each step reads one
/// contiguous row), in parallel blocks of columns; the result in `x`'s axis order.
fn blocked<T: Copy + Send + Sync, O: Copy + Send + Sync>(
    x: &ndarray::ArrayViewD<'_, T>,
    axis: usize,
    fill: O,
    f: impl Fn(ndarray::ArrayView2<'_, T>, ndarray::ArrayViewMut2<'_, O>) + Sync + Send,
) -> Result<ArrayD<O>> {
    let nd = x.ndim();
    let mut perm: Vec<usize> = vec![axis];
    perm.extend((0..nd).filter(|a| *a != axis));
    let moved = x.view().permuted_axes(ndarray::IxDyn(&perm));
    let moved_shape = moved.shape().to_vec();
    let contiguous = moved.as_standard_layout();
    let n = moved_shape.first().copied().unwrap_or(1);
    let m = moved_shape.iter().skip(1).product::<usize>();
    let bad = |e: ndarray::ShapeError| CoreError::InvalidReduction(format!("rolling_exp: {e}"));
    let input = contiguous
        .view()
        .into_shape_with_order((n, m))
        .map_err(bad)?;
    // The kernel writes every element: a zero `fill` lets the allocator hand out untouched
    // pages, filled in parallel as the blocks write them.
    let mut out =
        ndarray::Array2::from_shape_vec((n, m), vec![fill; n.saturating_mul(m)]).map_err(bad)?;
    let work = |(k, block): (usize, ndarray::ArrayViewMut2<'_, O>)| {
        let c0 = k.saturating_mul(BLOCK);
        let part = input.slice(ndarray::s![.., c0..c0.saturating_add(block.ncols())]);
        f(part, block);
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        out.axis_chunks_iter_mut(Axis(1), BLOCK)
            .into_par_iter()
            .enumerate()
            .for_each(work);
    }
    #[cfg(not(feature = "rayon"))]
    out.axis_chunks_iter_mut(Axis(1), BLOCK)
        .enumerate()
        .for_each(work);
    let shaped = out
        .into_shape_with_order(ndarray::IxDyn(&moved_shape))
        .map_err(bad)?;
    let mut inverse = vec![0; nd];
    for (i, p) in perm.iter().enumerate() {
        if let Some(slot) = inverse.get_mut(*p) {
            *slot = i;
        }
    }
    let in_order = shaped.permuted_axes(ndarray::IxDyn(&inverse));
    Ok(if in_order.is_standard_layout() {
        in_order
    } else {
        in_order.as_standard_layout().into_owned()
    })
}

/// `stat` of every lane of `x` along `axis`, in the data's own type (no `f64` copy).
fn lanes_stat<T: Copy + Send + Sync, O: Copy + Send + Sync>(
    x: &ndarray::ArrayViewD<'_, T>,
    axis: usize,
    to: impl Fn(T) -> f64 + Sync + Send,
    from: impl Fn(f64) -> O + Sync + Send,
    fill: O,
    (stat, alpha, min_weight): (Stat, f64, f64),
) -> Result<ArrayD<O>> {
    let decay = 1.0 - alpha;
    blocked(x, axis, fill, |input, mut out| {
        let mut states = vec![State::default(); input.ncols()];
        for (row, mut dst) in input.rows().into_iter().zip(out.rows_mut()) {
            for ((v, st), o) in row.iter().zip(states.iter_mut()).zip(dst.iter_mut()) {
                st.step(to(*v), alpha, decay);
                *o = from(st.value(stat, min_weight));
            }
        }
    })
}

/// The covariance (or correlation) of two lanes, written into `x`.
fn lane_pair(
    x: &mut ArrayViewMut1<'_, f64>,
    y: ndarray::ArrayView1<'_, f64>,
    corr: bool,
    alpha: f64,
    min_weight: f64,
) {
    let decay = 1.0 - alpha;
    let (mut weight, mut w, mut w2) = (0.0, 0.0, 0.0);
    let (mut mx, mut my, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let first = x
        .iter()
        .zip(y.iter())
        .find(|(a, b)| !(a.is_nan() || b.is_nan()));
    let (kx, ky) = first.map_or((0.0, 0.0), |(a, b)| (*a, *b));
    for (v, b) in x.iter_mut().zip(y.iter()) {
        let (a, b) = (*v - kx, *b - ky);
        weight *= decay;
        w *= decay;
        w2 *= decay * decay;
        sxx *= decay;
        syy *= decay;
        sxy *= decay;
        if !(a.is_nan() || b.is_nan()) {
            weight += alpha;
            let w_new = w + 1.0;
            let (dx, dy) = (a - mx, b - my);
            mx += dx / w_new;
            my += dy / w_new;
            sxx += dx * (a - mx);
            syy += dy * (b - my);
            sxy += dx * (b - my);
            w = w_new;
            w2 += 1.0;
        }
        let bias = if w > 0.0 { 1.0 - w2 / (w * w) } else { 0.0 };
        *v = if weight >= min_weight && bias > 0.0 {
            if corr {
                let den = (sxx * syy).sqrt();
                if den > 0.0 { sxy / den } else { f64::NAN }
            } else {
                (sxy / w) / bias
            }
        } else {
            f64::NAN
        };
    }
}

fn with_values(
    da: &DataArray,
    data: crate::TypedArray<crate::Owned>,
    unit: Option<annoray_units::Unit>,
) -> Result<DataArray> {
    let mut v = Variable::from_typed(da.dims().iter().cloned(), data)?;
    v.set_unit(unit);
    v.attrs_mut()
        .extend(da.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    let mut out = DataArray::new(v);
    for (k, c) in da.coords() {
        out = out.with_coord(k.clone(), c.clone())?;
    }
    Ok(match da.name() {
        Some(n) => out.with_name(n.clone()),
        None => out,
    })
}

impl<S: Repr> DataArrayBase<S> {
    /// Exponentially weighted moving statistics along `dim` (xarray's `rolling_exp`); see
    /// the module notes. `min_weight` is numbagg's (0 by default in xarray).
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::InvalidReduction`] for a window out of range.
    pub fn rolling_exp(
        &self,
        dim: &str,
        window: f64,
        window_type: WindowType,
        min_weight: f64,
    ) -> Result<RollingExp<'_, S>> {
        self.variable().axis(dim)?;
        Ok(RollingExp {
            da: self,
            dim: Name::from(dim),
            alpha: alpha_of(window, window_type)?,
            min_weight,
        })
    }
}

impl<S: Repr> RollingExp<'_, S> {
    fn stat(&self, stat: Stat, op: Reduction) -> Result<DataArray> {
        let unit = reduced_unit(self.da.unit(), op, 1)?;
        let (alpha, min_weight) = (self.alpha, self.min_weight);
        let dim = self.dim.clone();
        let p = (stat, alpha, min_weight);
        crate::slabs::by_lanes(self.da, &self.dim, |part| {
            let axis = part.variable().axis(&dim)?;
            #[allow(clippy::cast_possible_truncation)] // f32 data gives f32 results
            let data = match part.variable().typed()? {
                TypedArray::F32(x) => {
                    TypedArray::F32(lanes_stat(&x, axis, f64::from, |v| v as f32, 0.0_f32, p)?)
                }
                TypedArray::F64(x) => TypedArray::F64(lanes_stat(&x, axis, |v| v, |v| v, 0.0, p)?),
                _ => {
                    let x = part.variable().to_f64()?;
                    TypedArray::F64(lanes_stat(&x.view(), axis, |v| v, |v| v, 0.0, p)?)
                }
            };
            with_values(part, data, unit.clone())
        })
    }

    /// The exponentially weighted moving mean.
    ///
    /// # Errors
    /// Read errors; [`CoreError::UnsupportedDType`] for non-numeric data.
    pub fn mean(&self) -> Result<DataArray> {
        self.stat(Stat::Mean, Reduction::Mean)
    }

    /// The exponentially weighted moving sum.
    ///
    /// # Errors
    /// As for [`mean`](Self::mean); unit errors for units that cannot be added (°C).
    pub fn sum(&self) -> Result<DataArray> {
        self.stat(Stat::Sum, Reduction::Sum)
    }

    /// The decayed count of values.
    ///
    /// # Errors
    /// As for [`mean`](Self::mean).
    pub fn count(&self) -> Result<DataArray> {
        self.stat(Stat::Count, Reduction::Count)
    }

    /// The exponentially weighted moving variance (bias-corrected).
    ///
    /// # Errors
    /// As for [`mean`](Self::mean).
    pub fn var(&self) -> Result<DataArray> {
        self.stat(Stat::Var, Reduction::Var)
    }

    /// The exponentially weighted moving standard deviation (the square root of
    /// [`var`](Self::var), as numbagg and pandas).
    ///
    /// # Errors
    /// As for [`mean`](Self::mean).
    pub fn std(&self) -> Result<DataArray> {
        self.stat(Stat::Std, Reduction::Std)
    }

    fn pair<R: Repr>(&self, other: &DataArrayBase<R>, corr: bool) -> Result<DataArray> {
        let both = crate::broadcast(&[&self.da.to_owned(), &other.to_owned()], &[])?;
        let (Some(a), Some(b)) = (both.first(), both.get(1)) else {
            return Err(CoreError::InvalidReduction("rolling_exp: broadcast".into()));
        };
        let order: Vec<&str> = a.dims().iter().map(Name::as_str).collect();
        let b = b.transpose(&order)?;
        let axis = a.variable().axis(&self.dim)?;
        let y: ArrayD<f64> = b.to_f64()?;
        let mut x = a.to_f64()?;
        let (alpha, min_weight) = (self.alpha, self.min_weight);
        let zip = ndarray::Zip::from(x.lanes_mut(Axis(axis))).and(y.lanes(Axis(axis)));
        let each = |mut lx: ArrayViewMut1<'_, f64>, ly: ndarray::ArrayView1<'_, f64>| {
            lane_pair(&mut lx, ly, corr, alpha, min_weight);
        };
        #[cfg(feature = "rayon")]
        zip.par_for_each(each);
        #[cfg(not(feature = "rayon"))]
        zip.for_each(each);
        let unit = if corr {
            None
        } else {
            match (a.unit(), b.unit()) {
                (None, None) => None,
                (Some(u), None) | (None, Some(u)) => Some(u.delta()),
                (Some(u), Some(v)) => Some(u.delta().mul(&v.delta())?),
            }
        };
        with_values(a, crate::TypedArray::F64(x), unit)
    }

    /// The exponentially weighted moving covariance with `other` (aligned and broadcast;
    /// positions where either is missing are skipped).
    ///
    /// # Errors
    /// Alignment errors; read errors.
    pub fn cov<R: Repr>(&self, other: &DataArrayBase<R>) -> Result<DataArray> {
        self.pair(other, false)
    }

    /// The exponentially weighted moving correlation with `other`; unitless.
    ///
    /// # Errors
    /// As for [`cov`](Self::cov).
    pub fn corr<R: Repr>(&self, other: &DataArrayBase<R>) -> Result<DataArray> {
        self.pair(other, true)
    }
}
