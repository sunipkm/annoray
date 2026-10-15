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
//!
//! These read lazy data (weighted reductions stream where the plain ones do).

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
        if qs.iter().any(|q| !(0.0..=1.0).contains(q)) || (scalar_q && qs.len() != 1) {
            return Err(CoreError::InvalidReduction(
                "quantiles must be in [0, 1] (exactly one for a scalar quantile)".into(),
            ));
        }
        let reduced = all_dims(self.dims(), dims.reduce_dims())?;
        let r = rows(self.variable(), &reduced)?;
        let mut out = Vec::with_capacity(qs.len().saturating_mul(r.values.nrows()));
        let mut per_row: Vec<Vec<f64>> = Vec::with_capacity(r.values.nrows());
        for row in r.values.rows() {
            let has_nan = row.iter().any(|x| x.is_nan());
            let mut v: Vec<f64> = row.iter().copied().filter(|x| !x.is_nan()).collect();
            if has_nan && !skipna {
                v.clear();
            }
            v.sort_by(f64::total_cmp);
            per_row.push(v);
        }
        for q in qs {
            out.extend(per_row.iter().map(|v| quantile_sorted(v, *q, method)));
        }
        let (dims, shape): (Vec<Name>, Vec<usize>) = if scalar_q {
            (r.kept.clone(), r.shape.clone())
        } else {
            (
                std::iter::once(Name::from("quantile"))
                    .chain(r.kept.iter().cloned())
                    .collect(),
                std::iter::once(qs.len())
                    .chain(r.shape.iter().copied())
                    .collect(),
            )
        };
        let mut var = Variable::new(dims, shaped(&shape, out)?)?;
        var.set_unit(self.unit().cloned());
        let mut coords = indexmap::IndexMap::new();
        coords.insert(Name::from("quantile"), quantile_coord(qs, scalar_q)?);
        coords.extend(kept_coords(self, &reduced));
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
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
        let out = DataArrayBase::from_parts(self.name().cloned(), var, kept_coords(self, &reduced));
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
        let any_nan = weights
            .isnull()?
            .reduce(Reduction::Any, .., &ReduceOptions::default())?;
        if any_nan.to_f64()?.iter().any(|v| *v != 0.0) {
            return Err(CoreError::InvalidReduction(
                "weights cannot contain missing values".into(),
            ));
        }
        Ok(Weighted {
            data: self.to_owned(),
            weights: weights.to_owned(),
        })
    }
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
        let notnull = self.data.notnull()?;
        let mask = DataArrayBase::from_parts(
            None,
            Variable::new(notnull.dims().to_vec(), notnull.to_f64()?)?,
            notnull.coords().clone(),
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
        let dims = self.dims(dims)?;
        self.sum_of(&self.data.try_mul(&self.weights)?, &dims)
    }

    /// `Σ w·x / Σ w` over `dims`, counting weights only where the data is not NaN.
    ///
    /// # Errors
    /// As for [`sum_of_weights`](Self::sum_of_weights).
    pub fn mean(&self, dims: impl ReduceDims) -> Result<DataArray> {
        let dims = self.dims(dims)?;
        self.mean_over(&dims)
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
        let dims = self.dims(dims)?;
        self.var_over(&dims)
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
