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
use crate::{CoreError, DType, DataArray, DataArrayBase, Name, Reduction, Repr, Result, Variable};

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

/// `stat` of one lane, in place (numbagg's recurrences; West's updates for the spread).
fn lane(x: &mut ArrayViewMut1<'_, f64>, stat: Stat, alpha: f64, min_weight: f64) {
    let decay = 1.0 - alpha;
    // weight: decayed sum of α (numbagg's `weight`); w, w2: decayed count and sum of
    // squared weights; mean, m2: weighted mean and sum of squared deviations; sum: the
    // decayed sum; seen: any value so far (for `sum`).
    let (mut weight, mut w, mut w2, mut mean, mut m2, mut sum) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut seen = false;
    // The mean and spread are tracked for values less this shift.
    let shift = x.iter().copied().find(|a| !a.is_nan()).unwrap_or(0.0);
    for v in x.iter_mut() {
        let raw = *v;
        let a = raw - shift;
        weight *= decay;
        w *= decay;
        w2 *= decay * decay;
        m2 *= decay;
        sum *= decay;
        if !a.is_nan() {
            seen = true;
            weight += alpha;
            sum += raw;
            let w_new = w + 1.0;
            let delta = a - mean;
            mean += delta / w_new;
            m2 += delta * (a - mean);
            w = w_new;
            w2 += 1.0;
        }
        let ok = weight >= min_weight;
        let bias = if w > 0.0 { 1.0 - w2 / (w * w) } else { 0.0 };
        *v = match stat {
            Stat::Mean if ok && w > 0.0 => mean + shift,
            Stat::Sum if ok && seen => sum,
            Stat::Count if ok => w,
            Stat::Var if ok && bias > 0.0 => (m2 / w) / bias,
            Stat::Std if ok && bias > 0.0 => ((m2 / w) / bias).max(0.0).sqrt(),
            _ => f64::NAN,
        };
    }
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

/// `values` (as `f64`) of a variable, lane-computed along `axis` by `f`, back in the input's
/// float type.
fn along(
    v: &Variable,
    axis: usize,
    f: impl Fn(&mut ArrayViewMut1<'_, f64>),
) -> Result<crate::TypedArray<crate::Owned>> {
    let mut x = v.to_f64()?;
    for mut l in x.lanes_mut(Axis(axis)) {
        f(&mut l);
    }
    #[allow(clippy::cast_possible_truncation)] // f32 data gives f32 results
    Ok(if v.dtype() == DType::F32 {
        crate::TypedArray::F32(x.mapv(|a| a as f32))
    } else {
        crate::TypedArray::F64(x)
    })
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
        crate::slabs::by_lanes(self.da, &self.dim, |part| {
            let axis = part.variable().axis(&dim)?;
            let data = along(part.variable(), axis, |l| lane(l, stat, alpha, min_weight))?;
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
        for (mut lx, ly) in x.lanes_mut(Axis(axis)).into_iter().zip(y.lanes(Axis(axis))) {
            lane_pair(&mut lx, ly, corr, self.alpha, self.min_weight);
        }
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
