//! `quantile`, `idxmin`/`idxmax` and weighted reductions, as in xarray.
//!
//! - **Quantiles** follow numpy's `quantile` (its interpolation formula exactly, so results
//!   agree to the bit): methods linear (the default), lower, higher, nearest (ties to even)
//!   and midpoint. A list of probabilities adds a leading `quantile` dimension; a single one
//!   gives a scalar `quantile` coordinate. Units are kept (an order statistic of °C is °C).
//! - **`idxmin`/`idxmax`** give the coordinate label of the extreme along a dimension; an
//!   all-NaN lane gives NaN/NaT. Without `skipna`, NaN counts as the extreme (numpy's
//!   `argmin`), so a lane with NaN gives the label of its first NaN, as in xarray.
//! - **Weighted** sums, means, variances and standard deviations: weights broadcast by
//!   dimension name, must not be NaN, and count only where the data is not NaN. Units follow
//!   the arithmetic: a weighted sum has the unit of data × weights, a mean that of the data.
//!   Weighted quantiles use xarray's algorithm (Kish's effective sample size, linear
//!   interpolation; zero weights ignored).
//!
//! Lazy data: quantiles and `idxmin`/`idxmax` read it in slabs along a kept dimension within
//! the streaming budget (a quantile over every dimension needs all values at once and reads
//! it whole); weighted reductions stay lazy and stream like the plain reductions.

use ndarray::{Array2, ArrayD, IxDyn};

use crate::{
    CoreError, DataArray, DataArrayBase, Dataset, DatasetBase, Name, ReduceDims, ReduceOptions,
    Reduction, Repr, Result, Variable, VariableBase,
};

/// How to interpolate between order statistics (numpy's `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum QuantileMethod {
    /// Linear between the two neighbouring order statistics (the default).
    #[default]
    Linear,
    /// The lower neighbour.
    Lower,
    /// The higher neighbour.
    Higher,
    /// The nearest neighbour (ties to the even index, as numpy).
    Nearest,
    /// Halfway between the two neighbours.
    Midpoint,
}

/// numpy's `_lerp`: `a + (b - a) t`, computed from the nearer end for monotonicity.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

/// The `q`-quantile of sorted, NaN-free values.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // lengths far below 2^53; indices from non-negative floors within the length
fn quantile_sorted(v: &[f64], q: f64, method: QuantileMethod) -> f64 {
    let Some(last) = v.len().checked_sub(1) else {
        return f64::NAN;
    };
    let pos = last as f64 * q;
    let lo = (pos.floor() as usize).min(last);
    let hi = (pos.ceil() as usize).min(last);
    let at = |i: usize| v.get(i).copied().unwrap_or(f64::NAN);
    match method {
        QuantileMethod::Linear => lerp(at(lo), at(hi), pos - pos.floor()),
        QuantileMethod::Lower => at(lo),
        QuantileMethod::Higher => at(hi),
        QuantileMethod::Nearest => at((pos.round_ties_even() as usize).min(last)),
        QuantileMethod::Midpoint => {
            if lo == hi {
                at(lo)
            } else {
                lerp(at(lo), at(hi), 0.5)
            }
        }
    }
}

/// The `qs`-quantiles of NaN-free `v` (reordered in place): by selection (O(n) per
/// quantile) for one or two, by sorting for more. The same order statistics as sorting, so
/// the same results.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // as quantile_sorted
fn row_quantiles(v: &mut [f64], qs: &[f64], method: QuantileMethod) -> Vec<f64> {
    if qs.len() > 2 || v.len() < 64 {
        v.sort_by(f64::total_cmp);
        return qs.iter().map(|q| quantile_sorted(v, *q, method)).collect();
    }
    let Some(last) = v.len().checked_sub(1) else {
        return vec![f64::NAN; qs.len()];
    };
    qs.iter()
        .map(|q| {
            let pos = last as f64 * q;
            let lo = (pos.floor() as usize).min(last);
            let hi = (pos.ceil() as usize).min(last);
            let (_, lo_v, upper) = v.select_nth_unstable_by(lo, f64::total_cmp);
            let lo_v = *lo_v;
            // The next order statistic is the least of the upper part.
            let hi_v = if hi == lo {
                lo_v
            } else {
                upper.iter().copied().fold(f64::INFINITY, f64::min)
            };
            let near = (pos.round_ties_even() as usize).min(last);
            match method {
                QuantileMethod::Linear => lerp(lo_v, hi_v, pos - pos.floor()),
                QuantileMethod::Lower => lo_v,
                QuantileMethod::Higher => hi_v,
                QuantileMethod::Nearest => {
                    if near == lo {
                        lo_v
                    } else {
                        hi_v
                    }
                }
                QuantileMethod::Midpoint => {
                    if lo == hi {
                        lo_v
                    } else {
                        lerp(lo_v, hi_v, 0.5)
                    }
                }
            }
        })
        .collect()
}

/// The `qs`-quantiles of every lane of `v` along `dim` (kept dimensions in order), lanes in
/// parallel and read where they lie; `None` for non-float data (the row path converts).
fn lane_quantiles<S: Repr>(
    v: &VariableBase<S>,
    dim: &Name,
    qs: &[f64],
    method: QuantileMethod,
    skipna: bool,
) -> Result<Option<ArrayD<Vec<f64>>>> {
    fn go<T: Copy + Send + Sync>(
        x: &ndarray::ArrayViewD<'_, T>,
        axis: usize,
        to: impl Fn(T) -> f64 + Sync + Send,
        qs: &[f64],
        method: QuantileMethod,
        skipna: bool,
    ) -> ArrayD<Vec<f64>> {
        let one = |lane: ndarray::ArrayView1<'_, T>| -> Vec<f64> {
            let mut v: Vec<f64> = Vec::with_capacity(lane.len());
            let mut has_nan = false;
            for x in lane {
                let f = to(*x);
                if f.is_nan() {
                    has_nan = true;
                } else {
                    v.push(f);
                }
            }
            if has_nan && !skipna {
                v.clear();
            }
            row_quantiles(&mut v, qs, method)
        };
        let zip = ndarray::Zip::from(x.lanes(ndarray::Axis(axis)));
        #[cfg(feature = "rayon")]
        return zip.par_map_collect(one);
        #[cfg(not(feature = "rayon"))]
        zip.map_collect(one)
    }
    if v.is_lazy() {
        return Ok(None);
    }
    let axis = v.axis(dim)?;
    Ok(match v.typed()? {
        crate::TypedArray::F32(x) => Some(go(&x, axis, f64::from, qs, method, skipna)),
        crate::TypedArray::F64(x) => Some(go(&x, axis, |f| f, qs, method, skipna)),
        _ => None,
    })
}

/// The data as rows: kept dimensions first (their names and shape), the reduced ones
/// flattened into each row.
struct Rows {
    kept: Vec<Name>,
    shape: Vec<usize>,
    values: Array2<f64>,
}

fn all_dims(dims: &[Name], chosen: Option<Vec<Name>>) -> Result<Vec<Name>> {
    let chosen = chosen.unwrap_or_else(|| dims.to_vec());
    for d in &chosen {
        if !dims.contains(d) {
            return Err(CoreError::NoSuchDim(d.clone()));
        }
    }
    Ok(chosen)
}

fn rows<S: Repr>(v: &VariableBase<S>, reduced: &[Name]) -> Result<Rows> {
    let kept: Vec<Name> = v
        .dims()
        .iter()
        .filter(|d| !reduced.contains(d))
        .cloned()
        .collect();
    let order: Vec<&str> = kept
        .iter()
        .chain(v.dims().iter().filter(|d| reduced.contains(d)))
        .map(Name::as_str)
        .collect();
    let t = if v.dims().iter().map(Name::as_str).eq(order.iter().copied()) {
        v.to_owned()
    } else {
        v.transpose(&order)?
    };
    let shape: Vec<usize> = t.shape().iter().take(kept.len()).copied().collect();
    let nrows: usize = shape.iter().product();
    let n: usize = t.shape().iter().skip(kept.len()).product();
    let flat = t.to_f64()?.as_standard_layout().to_owned();
    let values = flat
        .into_shape_with_order((nrows, n))
        .map_err(|e| CoreError::InvalidReduction(e.to_string()))?;
    Ok(Rows {
        kept,
        shape,
        values,
    })
}

fn shaped(shape: &[usize], v: Vec<f64>) -> Result<ArrayD<f64>> {
    ArrayD::from_shape_vec(IxDyn(shape), v).map_err(|e| CoreError::InvalidReduction(e.to_string()))
}

/// Coordinates that survive reducing `reduced` (none of their dimensions reduced).
fn kept_coords<S: Repr>(
    da: &DataArrayBase<S>,
    reduced: &[Name],
) -> indexmap::IndexMap<Name, Variable> {
    da.coords()
        .iter()
        .filter(|(_, c)| !c.dims().iter().any(|d| reduced.contains(d)))
        .map(|(k, c)| (k.clone(), c.to_owned()))
        .collect()
}

fn quantile_coord(qs: &[f64], scalar: bool) -> Result<Variable> {
    if scalar {
        Variable::new(Vec::<Name>::new(), shaped(&[], qs.to_vec())?)
    } else {
        Variable::new(["quantile"], shaped(&[qs.len()], qs.to_vec())?)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// The `qs` quantiles over `dims` (`..` for all). With `scalar_q`, `qs` must hold one
    /// probability and no `quantile` dimension is added. NaNs are skipped with `skipna`
    /// (otherwise a lane with a NaN gives NaN).
    ///
    /// # Errors
    /// [`CoreError::InvalidReduction`] for probabilities outside `[0, 1]` (or several with
    /// `scalar_q`); [`CoreError::NoSuchDim`]; [`CoreError::UnsupportedDType`] for
    /// non-numeric data.
    pub fn quantile(
        &self,
        qs: &[f64],
        scalar_q: bool,
        dims: impl ReduceDims,
        method: QuantileMethod,
        skipna: bool,
    ) -> Result<DataArray> {
        let reduced = all_dims(self.dims(), dims.reduce_dims())?;
        // Lazy data: slabs along a kept dimension (over all dimensions, it is read whole).
        match self.dims().iter().find(|d| !reduced.contains(d)) {
            Some(kept) => crate::slabs::by_slabs(self, kept, 1, |s| {
                s.quantile_eager(qs, scalar_q, &reduced, method, skipna)
            }),
            None => self.quantile_eager(qs, scalar_q, &reduced, method, skipna),
        }
    }

    fn quantile_eager(
        &self,
        qs: &[f64],
        scalar_q: bool,
        reduced: &[Name],
        method: QuantileMethod,
        skipna: bool,
    ) -> Result<DataArray> {
        let reduced = reduced.to_vec();
        if qs.iter().any(|q| !(0.0..=1.0).contains(q)) || (scalar_q && qs.len() != 1) {
            return Err(CoreError::InvalidReduction(
                "quantiles must be in [0, 1] (exactly one for a scalar quantile)".into(),
            ));
        }
        // One reduced dimension: each lane gathered by its own task from the data as it is.
        if let [dim] = reduced.as_slice()
            && let Some(per_lane) = lane_quantiles(self.variable(), dim, qs, method, skipna)?
        {
            let kept: Vec<Name> = self.dims().iter().filter(|d| *d != dim).cloned().collect();
            let shape = per_lane.shape().to_vec();
            let per_row: Vec<Vec<f64>> = per_lane.into_iter().collect();
            return self.quantile_result(qs, scalar_q, &reduced, (kept, shape), &per_row);
        }
        let r = rows(self.variable(), &reduced)?;
        // Each row's quantiles, rows in parallel.
        let one = |row: ndarray::ArrayView1<'_, f64>| -> Vec<f64> {
            let has_nan = row.iter().any(|x| x.is_nan());
            let mut v: Vec<f64> = row.iter().copied().filter(|x| !x.is_nan()).collect();
            if has_nan && !skipna {
                v.clear();
            }
            row_quantiles(&mut v, qs, method)
        };
        #[cfg(feature = "rayon")]
        let per_row: Vec<Vec<f64>> = {
            use rayon::prelude::*;
            r.values
                .axis_iter(ndarray::Axis(0))
                .into_par_iter()
                .map(one)
                .collect()
        };
        #[cfg(not(feature = "rayon"))]
        let per_row: Vec<Vec<f64>> = r.values.axis_iter(ndarray::Axis(0)).map(one).collect();
        self.quantile_result(qs, scalar_q, &reduced, (r.kept, r.shape), &per_row)
    }

    /// The quantile array from each kept position's quantiles (in the kept dimensions'
    /// order), with a leading `quantile` dimension for several.
    fn quantile_result(
        &self,
        qs: &[f64],
        scalar_q: bool,
        reduced: &[Name],
        (kept, kept_shape): (Vec<Name>, Vec<usize>),
        per_row: &[Vec<f64>],
    ) -> Result<DataArray> {
        let mut out = Vec::with_capacity(qs.len().saturating_mul(per_row.len()));
        for i in 0..qs.len() {
            out.extend(
                per_row
                    .iter()
                    .map(|v| v.get(i).copied().unwrap_or(f64::NAN)),
            );
        }
        let (dims, shape): (Vec<Name>, Vec<usize>) = if scalar_q {
            (kept, kept_shape)
        } else {
            (
                std::iter::once(Name::from("quantile"))
                    .chain(kept)
                    .collect(),
                std::iter::once(qs.len()).chain(kept_shape).collect(),
            )
        };
        let mut var = Variable::new(dims, shaped(&shape, out)?)?;
        var.set_unit(self.unit().cloned());
        let mut coords = indexmap::IndexMap::new();
        coords.insert(Name::from("quantile"), quantile_coord(qs, scalar_q)?);
        coords.extend(kept_coords(self, reduced));
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords)
            .with_attrs_from(self.attrs()))
    }

    /// The label (coordinate value) of the minimum along `dim`.
    ///
    /// # Errors
    /// [`CoreError::NoIndex`] if `dim` has no coordinate; as for [`quantile`](Self::quantile).
    pub fn idxmin(&self, dim: &str, skipna: bool) -> Result<DataArray> {
        self.idx_extreme(dim, skipna, false)
    }

    /// The label (coordinate value) of the maximum along `dim`.
    ///
    /// # Errors
    /// As for [`idxmin`](Self::idxmin).
    pub fn idxmax(&self, dim: &str, skipna: bool) -> Result<DataArray> {
        self.idx_extreme(dim, skipna, true)
    }

    fn idx_extreme(&self, dim: &str, skipna: bool, max: bool) -> Result<DataArray> {
        match self.dims().iter().find(|d| d.as_str() != dim) {
            Some(kept) => crate::slabs::by_slabs(self, kept, 1, |s| s.idx_eager(dim, skipna, max)),
            None => self.idx_eager(dim, skipna, max),
        }
    }

    fn idx_eager(&self, dim: &str, skipna: bool, max: bool) -> Result<DataArray> {
        let name = Name::from(dim);
        let coord = self
            .coords()
            .get(&name)
            .filter(|c| c.dims() == [name.clone()])
            .ok_or_else(|| CoreError::NoIndex(name.clone()))?;
        let reduced = vec![name.clone()];
        let r = rows(self.variable(), &reduced)?;
        let better = |x: f64, best: f64| if max { x > best } else { x < best };
        let found: Vec<Option<usize>> = r
            .values
            .rows()
            .into_iter()
            .map(|row| {
                if !skipna && let Some(i) = row.iter().position(|x| x.is_nan()) {
                    // numpy's argmin/argmax: NaN is the extreme (xarray ignores the fill).
                    return Some(i);
                }
                let mut best: Option<(usize, f64)> = None;
                for (i, x) in row.iter().enumerate() {
                    if !x.is_nan() && best.is_none_or(|(_, b)| better(*x, b)) {
                        best = Some((i, *x));
                    }
                }
                best.map(|(i, _)| i)
            })
            .collect();
        let pos = found
            .iter()
            .map(|p| {
                isize::try_from(p.unwrap_or(0)).map_err(|_| CoreError::ArithmeticOverflow("index"))
            })
            .collect::<Result<Vec<isize>>>()?;
        // The labels at those positions, shaped like the kept dimensions.
        let labels = coord.take(dim, &pos)?.into_typed()?;
        let labels = crate::array::typed_dispatch!(labels, a => crate::array::Element::wrap::<crate::Owned>(
            a.into_shape_with_order(IxDyn(&r.shape)).map_err(|e| CoreError::InvalidReduction(e.to_string()))?
        ));
        let mut var = Variable::from_typed(r.kept.clone(), labels)?;
        var.set_unit(coord.unit().cloned());
        let out = DataArrayBase::from_parts(self.name().cloned(), var, kept_coords(self, &reduced))
            .with_attrs_from(self.attrs());
        if found.iter().all(Option::is_some) {
            return Ok(out);
        }
        let valid: Vec<bool> = found.iter().map(Option::is_some).collect();
        let cond = DataArray::from_array(
            r.kept.clone(),
            ArrayD::from_shape_vec(IxDyn(&r.shape), valid)
                .map_err(|e| CoreError::InvalidReduction(e.to_string()))?,
        )?;
        out.where_(&cond)
    }

    /// Reductions weighted by `weights` (broadcast by dimension name; no NaN).
    ///
    /// # Errors
    /// [`CoreError::InvalidReduction`] if the weights contain NaN.
    pub fn weighted<R: Repr>(&self, weights: &DataArrayBase<R>) -> Result<Weighted> {
        check_weights(weights)?;
        Ok(Weighted {
            data: self.to_owned(),
            weights: weights.to_owned(),
        })
    }
}

/// Weights must not be NaN (as xarray).
fn check_weights<R: Repr>(weights: &DataArrayBase<R>) -> Result<()> {
    // The minimum of the validity mask (streamed for lazy weights): below 1 means a NaN.
    let valid = DataArrayBase::from_parts(
        None,
        weights.variable().valid_f64()?,
        indexmap::IndexMap::new(),
    );
    let lowest = valid.reduce(Reduction::Min, .., &ReduceOptions::default())?;
    if lowest.to_f64()?.iter().any(|v| *v < 1.0) {
        return Err(CoreError::InvalidReduction(
            "weights cannot contain missing values".into(),
        ));
    }
    Ok(())
}

/// Weighted reductions of a data array: from [`DataArrayBase::weighted`].
#[derive(Debug, Clone)]
pub struct Weighted {
    data: DataArray,
    weights: DataArray,
}

impl Weighted {
    fn dims(&self, dims: impl ReduceDims) -> Result<Vec<Name>> {
        let mut all: Vec<Name> = self.data.dims().to_vec();
        for d in self.weights.dims() {
            if !all.contains(d) {
                all.push(d.clone());
            }
        }
        all_dims(&all, dims.reduce_dims())
    }

    fn sum_of(&self, x: &DataArray, dims: &[Name]) -> Result<DataArray> {
        x.reduce(Reduction::Sum, dims.to_vec(), &ReduceOptions::default())
    }

    /// The sum of the weights where the data is not NaN (NaN where it is zero).
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; errors of the arithmetic.
    pub fn sum_of_weights(&self, dims: impl ReduceDims) -> Result<DataArray> {
        let dims = self.dims(dims)?;
        self.weight_total(&dims)
    }

    fn weight_total(&self, dims: &[Name]) -> Result<DataArray> {
        // 1 where the data is valid, 0 where it is NaN: weights × mask broadcasts by name.
        let mask = DataArrayBase::from_parts(
            None,
            self.data.variable().valid_f64()?,
            self.data.coords().clone(),
        );
        let valid = self.weights.try_mul(&mask)?;
        let total = self.sum_of(&valid, dims)?;
        let nonzero = Variable::new(total.dims().to_vec(), total.to_f64()?.mapv(|w| w != 0.0))?;
        total.where_(&DataArrayBase::from_parts(
            None,
            nonzero,
            indexmap::IndexMap::new(),
        ))
    }

    /// `Σ w·x` over `dims` (NaN data skipped), in the unit of data × weights.
    ///
    /// # Errors
    /// As for [`sum_of_weights`](Self::sum_of_weights).
    pub fn sum(&self, dims: impl ReduceDims) -> Result<DataArray> {
        let out: Result<DataArray> = (|| {
            let dims = self.dims(dims)?;
            self.sum_of(&self.data.try_mul(&self.weights)?, &dims)
        })();
        out.map(|r| r.with_attrs_from(self.data.attrs()))
    }

    /// `Σ w·x / Σ w` over `dims`, counting weights only where the data is not NaN.
    ///
    /// # Errors
    /// As for [`sum_of_weights`](Self::sum_of_weights).
    pub fn mean(&self, dims: impl ReduceDims) -> Result<DataArray> {
        let out: Result<DataArray> = (|| {
            let dims = self.dims(dims)?;
            self.mean_over(&dims)
        })();
        out.map(|r| r.with_attrs_from(self.data.attrs()))
    }

    fn mean_over(&self, dims: &[Name]) -> Result<DataArray> {
        let sum = self.sum_of(&self.data.try_mul(&self.weights)?, dims)?;
        sum.try_div(&self.weight_total(dims)?)
    }

    /// The weighted variance `Σ w·(x − mean)² / Σ w` (no degrees-of-freedom correction, as
    /// xarray), in the data's unit squared.
    ///
    /// # Errors
    /// As for [`sum_of_weights`](Self::sum_of_weights).
    pub fn var(&self, dims: impl ReduceDims) -> Result<DataArray> {
        let out: Result<DataArray> = (|| {
            let dims = self.dims(dims)?;
            self.var_over(&dims)
        })();
        out.map(|r| r.with_attrs_from(self.data.attrs()))
    }

    fn var_over(&self, dims: &[Name]) -> Result<DataArray> {
        let dev = self.data.try_sub(&self.mean_over(dims)?)?;
        let sq = dev.try_mul(&dev)?;
        let sum = self.sum_of(&sq.try_mul(&self.weights)?, dims)?;
        sum.try_div(&self.weight_total(dims)?)
    }

    /// The square root of [`var`](Self::var), in the data's unit.
    ///
    /// # Errors
    /// As for [`sum_of_weights`](Self::sum_of_weights).
    pub fn std(&self, dims: impl ReduceDims) -> Result<DataArray> {
        let out: Result<DataArray> = (|| {
            let dims = self.dims(dims)?;
            let var = self.var_over(&dims)?;
            let values = var.to_f64()?.mapv(f64::sqrt);
            let mut v = Variable::new(var.dims().to_vec(), values)?;
            v.set_unit(var.unit().map(|u| u.root(2)).transpose()?);
            Ok(DataArrayBase::from_parts(
                var.name().cloned(),
                v,
                var.coords()
                    .iter()
                    .map(|(k, c)| (k.clone(), c.clone()))
                    .collect(),
            ))
        })();
        out.map(|r| r.with_attrs_from(self.data.attrs()))
    }
}

/// xarray's weighted quantile of one lane (adapted there from Akinshin's reference
/// implementation): zero weights dropped, Kish's effective sample size
/// `n = (Σw)² / Σw²`, Hyndman–Fan type 7 (linear) positions `h = (n − 1) q + 1`.
#[allow(clippy::cast_precision_loss)]
fn weighted_quantile_1d(data: &[f64], weights: &[f64], qs: &[f64], skipna: bool) -> Vec<f64> {
    let nan = || vec![f64::NAN; qs.len()];
    if !skipna && data.iter().any(|x| x.is_nan()) {
        return nan();
    }
    let mut pairs: Vec<(f64, f64)> = data
        .iter()
        .zip(weights)
        .filter(|(x, w)| !x.is_nan() && **w != 0.0)
        .map(|(x, w)| (*x, *w))
        .collect();
    if pairs.is_empty() {
        return nan();
    }
    let total: f64 = pairs.iter().map(|(_, w)| w).sum();
    let squares: f64 = pairs.iter().map(|(_, w)| w * w).sum();
    let nw = total * total / squares;
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut cum = Vec::with_capacity(pairs.len().saturating_add(1));
    cum.push(0.0);
    let mut acc = 0.0;
    for (_, w) in &pairs {
        acc += w / total;
        cum.push(acc);
    }
    qs.iter()
        .map(|q| {
            let h = ((nw - 1.0) * q + 1.0).clamp(1.0, nw);
            let (lo, hi) = ((h - 1.0) / nw, h / nw);
            let v: Vec<f64> = cum
                .iter()
                .map(|c| (lo.max(hi.min(*c))) * nw - h + 1.0)
                .collect();
            pairs
                .iter()
                .zip(v.windows(2))
                .map(|((x, _), d)| {
                    x * (d.get(1).copied().unwrap_or(0.0) - d.first().copied().unwrap_or(0.0))
                })
                .sum()
        })
        .collect()
}

/// `a` broadcast against `b` by dimension name: `a × ones(b)` (units of `a` kept).
fn broadcast_against(a: &DataArray, b: &DataArray) -> Result<DataArray> {
    let ones = Variable::new(
        b.dims().to_vec(),
        ArrayD::from_elem(IxDyn(b.variable().shape()), 1.0),
    )?;
    let ones = DataArrayBase::from_parts(
        None,
        ones,
        b.coords()
            .iter()
            .filter(|(_, c)| c.dims().iter().all(|d| b.dims().contains(d)))
            .map(|(k, c)| (k.clone(), c.clone()))
            .collect(),
    );
    a.try_mul(&ones)
}

impl Weighted {
    /// Weighted quantiles over `dims` (xarray's algorithm: Kish's effective sample size and
    /// linear interpolation; zero weights ignored). `scalar_q` as for
    /// [`DataArrayBase::quantile`].
    ///
    /// # Errors
    /// As for [`DataArrayBase::quantile`].
    pub fn quantile(
        &self,
        qs: &[f64],
        scalar_q: bool,
        dims: impl ReduceDims,
        skipna: bool,
    ) -> Result<DataArray> {
        if qs.iter().any(|q| !(0.0..=1.0).contains(q)) || (scalar_q && qs.len() != 1) {
            return Err(CoreError::InvalidReduction(
                "quantiles must be in [0, 1] (exactly one for a scalar quantile)".into(),
            ));
        }
        let reduced = self.dims(dims)?;
        // Data and weights on the same (union) dimensions, in the same order.
        let data = broadcast_against(&self.data, &self.weights)?;
        let order: Vec<&str> = data.dims().iter().map(Name::as_str).collect();
        let weights = broadcast_against(&self.weights, &self.data)?.transpose(&order)?;
        let rd = rows(data.variable(), &reduced)?;
        let rw = rows(weights.variable(), &reduced)?;
        let per_row: Vec<Vec<f64>> = rd
            .values
            .rows()
            .into_iter()
            .zip(rw.values.rows())
            .map(|(d, w)| {
                let (d, w): (Vec<f64>, Vec<f64>) = (d.to_vec(), w.to_vec());
                weighted_quantile_1d(&d, &w, qs, skipna)
            })
            .collect();
        let mut out = Vec::with_capacity(qs.len().saturating_mul(per_row.len()));
        for i in 0..qs.len() {
            out.extend(
                per_row
                    .iter()
                    .map(|r| r.get(i).copied().unwrap_or(f64::NAN)),
            );
        }
        let (dims, shape): (Vec<Name>, Vec<usize>) = if scalar_q {
            (rd.kept.clone(), rd.shape.clone())
        } else {
            (
                std::iter::once(Name::from("quantile"))
                    .chain(rd.kept.iter().cloned())
                    .collect(),
                std::iter::once(qs.len())
                    .chain(rd.shape.iter().copied())
                    .collect(),
            )
        };
        let mut var = Variable::new(dims, shaped(&shape, out)?)?;
        var.set_unit(self.data.unit().cloned());
        let mut coords = indexmap::IndexMap::new();
        coords.insert(Name::from("quantile"), quantile_coord(qs, scalar_q)?);
        coords.extend(kept_coords(&data, &reduced));
        Ok(DataArrayBase::from_parts(
            self.data.name().cloned(),
            var,
            coords,
        ))
    }
}

/// Weighted reductions of every numeric data variable: from [`DatasetBase::weighted`].
#[derive(Debug, Clone)]
pub struct DatasetWeighted {
    ds: Dataset,
    weights: DataArray,
}

impl<S: Repr> DatasetBase<S> {
    /// Reductions of every numeric data variable weighted by `weights`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::weighted`].
    pub fn weighted<R: Repr>(&self, weights: &DataArrayBase<R>) -> Result<DatasetWeighted> {
        check_weights(weights)?;
        Ok(DatasetWeighted {
            ds: self.to_owned(),
            weights: weights.to_owned(),
        })
    }
}

impl DatasetWeighted {
    fn each(&self, f: impl Fn(&Weighted) -> Result<DataArray>) -> Result<Dataset> {
        let mut out = Dataset::new();
        let mut coords: indexmap::IndexMap<Name, Variable> = indexmap::IndexMap::new();
        let mut vars = Vec::new();
        for k in self.ds.data_vars().keys() {
            let da = self.ds.get(k)?.to_owned();
            if matches!(
                da.variable().dtype(),
                crate::DType::Str
                    | crate::DType::Bool
                    | crate::DType::DateTime
                    | crate::DType::CfDate
            ) {
                continue;
            }
            let w = Weighted {
                data: da,
                weights: self.weights.clone(),
            };
            let r = f(&w)?;
            for (ck, c) in r.coords() {
                coords.entry(ck.clone()).or_insert_with(|| c.clone());
            }
            vars.push((k.clone(), r.variable().to_owned()));
        }
        for (k, c) in coords {
            out.add_coord(k, c)?;
        }
        for (k, v) in vars {
            out.add_var(k, v)?;
        }
        *out.attrs_mut() = self.ds.attrs().clone();
        Ok(out)
    }

    fn names(dims: impl ReduceDims) -> Option<Vec<Name>> {
        dims.reduce_dims()
    }

    /// [`Weighted::sum_of_weights`] of every variable.
    ///
    /// # Errors
    /// As for [`Weighted::sum_of_weights`].
    pub fn sum_of_weights(&self, dims: impl ReduceDims) -> Result<Dataset> {
        let d = Self::names(dims);
        self.each(|w| match &d {
            Some(d) => w.sum_of_weights(d.clone()),
            None => w.sum_of_weights(..),
        })
    }

    /// [`Weighted::sum`] of every variable.
    ///
    /// # Errors
    /// As for [`Weighted::sum`].
    pub fn sum(&self, dims: impl ReduceDims) -> Result<Dataset> {
        let d = Self::names(dims);
        self.each(|w| match &d {
            Some(d) => w.sum(d.clone()),
            None => w.sum(..),
        })
    }

    /// [`Weighted::mean`] of every variable.
    ///
    /// # Errors
    /// As for [`Weighted::mean`].
    pub fn mean(&self, dims: impl ReduceDims) -> Result<Dataset> {
        let d = Self::names(dims);
        self.each(|w| match &d {
            Some(d) => w.mean(d.clone()),
            None => w.mean(..),
        })
    }

    /// [`Weighted::var`] of every variable.
    ///
    /// # Errors
    /// As for [`Weighted::var`].
    pub fn var(&self, dims: impl ReduceDims) -> Result<Dataset> {
        let d = Self::names(dims);
        self.each(|w| match &d {
            Some(d) => w.var(d.clone()),
            None => w.var(..),
        })
    }

    /// [`Weighted::std`] of every variable.
    ///
    /// # Errors
    /// As for [`Weighted::std`].
    pub fn std(&self, dims: impl ReduceDims) -> Result<Dataset> {
        let d = Self::names(dims);
        self.each(|w| match &d {
            Some(d) => w.std(d.clone()),
            None => w.std(..),
        })
    }

    /// [`Weighted::quantile`] of every variable.
    ///
    /// # Errors
    /// As for [`Weighted::quantile`].
    pub fn quantile(
        &self,
        qs: &[f64],
        scalar_q: bool,
        dims: impl ReduceDims,
        skipna: bool,
    ) -> Result<Dataset> {
        let d = Self::names(dims);
        self.each(|w| match &d {
            Some(d) => w.quantile(qs, scalar_q, d.clone(), skipna),
            None => w.quantile(qs, scalar_q, .., skipna),
        })
    }
}

/// Apply a DataArray operation to every numeric data variable that has any of `dims`.
fn per_variable<S: Repr>(
    ds: &DatasetBase<S>,
    dims: &[Name],
    f: impl Fn(&DataArrayBase<crate::View<'_>>) -> Result<DataArray>,
) -> Result<Dataset> {
    let mut out = Dataset::new();
    let mut vars = Vec::new();
    let mut coords: indexmap::IndexMap<Name, Variable> = indexmap::IndexMap::new();
    for k in ds.data_vars().keys() {
        let da = ds.get(k)?;
        if !da.dims().iter().any(|d| dims.contains(d)) {
            vars.push((k.clone(), da.variable().to_owned()));
            continue;
        }
        if !matches!(
            da.variable().dtype(),
            crate::DType::Str | crate::DType::Bool | crate::DType::DateTime | crate::DType::CfDate
        ) {
            let r = f(&da)?;
            for (ck, c) in r.coords() {
                coords.entry(ck.clone()).or_insert_with(|| c.clone());
            }
            vars.push((k.clone(), r.variable().to_owned()));
        }
    }
    for (k, c) in ds.coords() {
        if !c.dims().iter().any(|d| dims.contains(d)) && !coords.contains_key(k) {
            coords.insert(k.clone(), c.to_owned());
        }
    }
    for (k, c) in coords {
        out.add_coord(k, c)?;
    }
    for (k, v) in vars {
        out.add_var(k, v)?;
    }
    *out.attrs_mut() = ds.attrs().clone();
    Ok(out)
}

impl<S: Repr> DatasetBase<S> {
    /// [`DataArrayBase::quantile`] of every numeric data variable with any of `dims`
    /// (variables without them are kept as they are; non-numeric ones are dropped).
    ///
    /// # Errors
    /// As for [`DataArrayBase::quantile`].
    pub fn quantile(
        &self,
        qs: &[f64],
        scalar_q: bool,
        dims: impl ReduceDims,
        method: QuantileMethod,
        skipna: bool,
    ) -> Result<Dataset> {
        let all: Vec<Name> = self.dims().into_iter().collect();
        let dims = all_dims(&all, dims.reduce_dims())?;
        per_variable(self, &dims, |da| {
            let own: Vec<Name> = da
                .dims()
                .iter()
                .filter(|d| dims.contains(d))
                .cloned()
                .collect();
            da.quantile(qs, scalar_q, own, method, skipna)
        })
    }

    /// [`DataArrayBase::idxmin`] of every numeric data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::idxmin`].
    pub fn idxmin(&self, dim: &str, skipna: bool) -> Result<Dataset> {
        per_variable(self, &[Name::from(dim)], |da| da.idxmin(dim, skipna))
    }

    /// [`DataArrayBase::idxmax`] of every numeric data variable along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::idxmin`].
    pub fn idxmax(&self, dim: &str, skipna: bool) -> Result<Dataset> {
        per_variable(self, &[Name::from(dim)], |da| da.idxmax(dim, skipna))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_methods() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile_sorted(&v, 0.5, QuantileMethod::Linear), 2.5);
        assert_eq!(quantile_sorted(&v, 0.5, QuantileMethod::Lower), 2.0);
        assert_eq!(quantile_sorted(&v, 0.5, QuantileMethod::Higher), 3.0);
        // Ties to the even index, as numpy: 1.5 → 2, 0.5 → 0, 2.5 → 2.
        assert_eq!(quantile_sorted(&v, 0.5, QuantileMethod::Nearest), 3.0);
        assert_eq!(
            quantile_sorted(&[1.0, 2.0, 3.0], 0.25, QuantileMethod::Nearest),
            1.0
        );
        let six = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(quantile_sorted(&six, 0.5, QuantileMethod::Nearest), 3.0);
        assert_eq!(quantile_sorted(&v, 0.5, QuantileMethod::Midpoint), 2.5);
        assert!(quantile_sorted(&[], 0.5, QuantileMethod::Linear).is_nan());
    }
}
