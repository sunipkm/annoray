//! Interpolation along dimension coordinates (xarray's `interp`, as numpy's `interp` and
//! scipy's `interp1d` do it).
//!
//! - **Linear** and **nearest** (ties go to the lower coordinate). Targets outside the
//!   coordinate's range, and NaN targets, give NaN; there is no extrapolation.
//! - Coordinates may be unsorted or descending; numbers or dates (times interpolate in
//!   nanoseconds). A target exactly on a coordinate takes that value, even next to a NaN.
//! - Integers become `f64`; `f32` stays `f32`. Units are kept.
//! - Numeric coordinates along the dimension are interpolated too; other ones are dropped.
//!   (xarray back-fills string coordinates, which gives labels that mean nothing at the new
//!   positions; and its `nearest` gives the last value for a NaN target.)

use indexmap::IndexMap;
use ndarray::{ArrayD, Axis, IxDyn};

use crate::dataarray::DataArrayBase;
use crate::lookup::{Keys, index_keys};
use crate::ops::to_f64;
use crate::{
    CoreError, DType, DataArray, Dataset, DatasetBase, Name, Repr, Result, TypedArray, Variable,
    VariableBase,
};

/// How to interpolate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum InterpMethod {
    /// Straight lines between neighbouring points (the default).
    #[default]
    Linear,
    /// The value at the nearest coordinate (the lower one on ties).
    Nearest,
}

fn invalid(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(reason.into())
}

/// Positions as numbers: values, or nanoseconds relative to `origin` for times.
fn positions(keys: Keys, origin: Option<i128>) -> Result<(Vec<f64>, Option<i128>)> {
    #[allow(clippy::cast_precision_loss)] // nanosecond offsets in f64, as numpy does
    Ok(match keys {
        Keys::Num(v) => (v, None),
        Keys::Ns(v, _) => {
            let o = origin
                .or_else(|| v.iter().copied().find(|t| *t != i128::MIN))
                .unwrap_or(0);
            let p = v
                .iter()
                .map(|t| {
                    if *t == i128::MIN {
                        f64::NAN
                    } else {
                        t.saturating_sub(o) as f64
                    }
                })
                .collect();
            (p, Some(o))
        }
        Keys::Str(_) => return Err(invalid("cannot interpolate along string labels")),
    })
}

/// For each target: the two source positions and the weight of the second (`None` for NaN).
type Weights = Vec<Option<(usize, usize, f64)>>;

fn weights(x: &[f64], targets: &[f64], method: InterpMethod) -> Weights {
    // Sorted source positions (NaN coordinates are left out).
    let mut order: Vec<usize> = (0..x.len())
        .filter(|i| x.get(*i).is_some_and(|v| !v.is_nan()))
        .collect();
    order.sort_by(|a, b| {
        let (p, q) = (
            x.get(*a).copied().unwrap_or(f64::NAN),
            x.get(*b).copied().unwrap_or(f64::NAN),
        );
        p.total_cmp(&q)
    });
    let xs: Vec<f64> = order.iter().filter_map(|i| x.get(*i).copied()).collect();
    let (Some(&first), Some(&last)) = (xs.first(), xs.last()) else {
        return vec![None; targets.len()];
    };
    targets
        .iter()
        .map(|&t| {
            if t.is_nan() || t < first || t > last {
                return None;
            }
            // `hi`: the first sorted position strictly above `t` (or the last one).
            let hi = xs
                .partition_point(|v| *v <= t)
                .min(xs.len().saturating_sub(1));
            let lo = hi.saturating_sub(1);
            let (xl, xh) = (xs.get(lo).copied()?, xs.get(hi).copied()?);
            let (il, ih) = (order.get(lo).copied()?, order.get(hi).copied()?);
            if t == xh {
                return Some((ih, ih, 0.0));
            }
            if t == xl {
                return Some((il, il, 0.0));
            }
            match method {
                InterpMethod::Linear => Some((il, ih, (t - xl) / (xh - xl))),
                InterpMethod::Nearest => {
                    let to_hi = xh - t < t - xl;
                    let i = if to_hi { ih } else { il };
                    Some((i, i, 0.0))
                }
            }
        })
        .collect()
}

/// Interpolate one variable's values along `axis`.
fn apply<S: Repr>(
    v: &VariableBase<S>,
    axis: usize,
    w: &Weights,
) -> Result<TypedArray<crate::Owned>> {
    let typed = v.typed()?;
    let f32_data = typed.dtype() == DType::F32;
    let data = to_f64(&typed, None)?;
    let mut shape = data.shape().to_vec();
    if let Some(s) = shape.get_mut(axis) {
        *s = w.len();
    }
    let mut out = ArrayD::from_elem(IxDyn(&shape), f64::NAN);
    for (k, wk) in w.iter().enumerate() {
        let Some((lo, hi, t)) = *wk else { continue };
        let mut dst = out.index_axis_mut(Axis(axis), k);
        let a = data.index_axis(Axis(axis), lo);
        if lo == hi {
            dst.assign(&a);
        } else {
            let b = data.index_axis(Axis(axis), hi);
            ndarray::Zip::from(&mut dst)
                .and(&a)
                .and(&b)
                .for_each(|d, p, q| *d = p + (q - p) * t);
        }
    }
    #[allow(clippy::cast_possible_truncation)] // f32 data stays f32
    Ok(if f32_data {
        TypedArray::F32(out.mapv(|x| x as f32))
    } else {
        TypedArray::F64(out)
    })
}

fn numeric(d: DType) -> bool {
    !matches!(
        d,
        DType::Str | DType::Bool | DType::DateTime | DType::CfDate | DType::TimeDelta
    )
}

/// The weights for interpolating dimension `dim` (whose coordinate is `coord`) at `at`.
fn plan<C: Repr, R: Repr>(
    coord: &VariableBase<C>,
    dim: &Name,
    at: &VariableBase<R>,
    method: InterpMethod,
) -> Result<(Weights, Variable)> {
    if at.ndim() != 1 {
        return Err(invalid("interpolation targets must be one-dimensional"));
    }
    if coord.dtype() != at.dtype() && !(numeric(coord.dtype()) && numeric(at.dtype())) {
        return Err(invalid(format!(
            "targets of type {} for a coordinate of type {}",
            at.dtype(),
            coord.dtype()
        )));
    }
    let (x, origin) = positions(index_keys(coord, dim)?, None)?;
    let (mut t, _) = positions(index_keys(at, dim)?, origin)?;
    // Targets in another unit of the same kind are converted to the coordinate's.
    if let (Some(from), Some(to)) = (at.unit(), coord.unit())
        && from != to
    {
        let c = from.converter_to(to)?;
        t = t.into_iter().map(|v| c.apply(v)).collect();
    }
    let mut new_coord = Variable::from_typed([dim.clone()], at.typed()?.to_owned())?;
    new_coord.set_unit(coord.unit().cloned());
    if at.unit().is_some() && coord.unit().is_some() && at.unit() != coord.unit() {
        // Values were converted: store them in the coordinate's unit.
        new_coord = Variable::from_typed(
            [dim.clone()],
            TypedArray::F64(
                ArrayD::from_shape_vec(IxDyn(&[t.len()]), t.clone())
                    .map_err(|_| invalid("shape"))?,
            ),
        )?;
        new_coord.set_unit(coord.unit().cloned());
    }
    Ok((weights(&x, &t, method), new_coord))
}

/// `v` interpolated along `dim` with `w`, keeping its metadata.
fn interp_var<S: Repr>(v: &VariableBase<S>, dim: &Name, w: &Weights) -> Result<Variable> {
    let axis = v.axis(dim)?;
    let mut out = Variable::from_typed(v.dims().iter().cloned(), apply(v, axis, w)?)?;
    out.set_unit(v.unit().cloned());
    out.attrs_mut()
        .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    Ok(out)
}

/// The coordinates after interpolating `dim`: its index replaced, numeric ones along it
/// interpolated, other ones along it dropped.
fn coords_after<'c, C: Repr + 'c>(
    coords: impl Iterator<Item = (&'c Name, &'c VariableBase<C>)>,
    dim: &Name,
    w: &Weights,
    new_coord: &Variable,
) -> Result<IndexMap<Name, Variable>> {
    let mut out = IndexMap::new();
    for (k, c) in coords {
        if k == dim {
            out.insert(k.clone(), new_coord.clone());
        } else if !c.dims().contains(dim) {
            out.insert(k.clone(), c.to_owned());
        } else if numeric(c.dtype()) {
            out.insert(k.clone(), interp_var(c, dim, w)?);
        }
    }
    Ok(out)
}

impl<S: Repr> DataArrayBase<S> {
    /// Interpolate along `dim` at the coordinate values `at` (a 1-D variable of numbers or
    /// dates; in another unit of the same kind, converted first).
    ///
    /// # Errors
    /// [`CoreError::NoIndex`] if `dim` has no coordinate; [`CoreError::InvalidReduction`] for
    /// string coordinates, mismatched target types or non-1-D targets; unit errors;
    /// [`CoreError::UnsupportedDType`] for non-numeric data.
    pub fn interp<R: Repr>(
        &self,
        dim: &str,
        at: &VariableBase<R>,
        method: InterpMethod,
    ) -> Result<DataArray> {
        let dim = Name::from(dim);
        let coord = self
            .coords()
            .get(&dim)
            .ok_or_else(|| CoreError::NoIndex(dim.clone()))?;
        let (w, new_coord) = plan(coord, &dim, at, method)?;
        let var = interp_var(self.variable(), &dim, &w)?;
        let coords = coords_after(self.coords().iter(), &dim, &w, &new_coord)?;
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
    }

    /// Interpolate onto the dimension coordinates of `other`, for every dimension both have
    /// a coordinate for.
    ///
    /// # Errors
    /// As for [`interp`](Self::interp).
    pub fn interp_like<R: Repr>(
        &self,
        other: &DataArrayBase<R>,
        method: InterpMethod,
    ) -> Result<DataArray> {
        let mut out = self.to_owned();
        for (k, c) in other.coords() {
            let is_index = c.dims().len() == 1 && c.dims().first() == Some(k);
            if is_index && out.coords().contains_key(k) && out.dims().contains(k) {
                out = out.interp(k, c, method)?;
            }
        }
        Ok(out)
    }
}

impl<S: Repr> DatasetBase<S> {
    /// Interpolate every numeric data variable with dimension `dim` at `at`; non-numeric
    /// variables with it are dropped, others kept.
    ///
    /// # Errors
    /// As for [`DataArrayBase::interp`].
    pub fn interp<R: Repr>(
        &self,
        dim: &str,
        at: &VariableBase<R>,
        method: InterpMethod,
    ) -> Result<Dataset> {
        let dim = Name::from(dim);
        let coord = self
            .coords()
            .get(&dim)
            .ok_or_else(|| CoreError::NoIndex(dim.clone()))?;
        let (w, new_coord) = plan(coord, &dim, at, method)?;
        let mut out = Dataset::new();
        for (k, c) in coords_after(self.coords().iter(), &dim, &w, &new_coord)? {
            out.add_coord(k, c)?;
        }
        for (k, v) in self.data_vars() {
            if !v.dims().contains(&dim) {
                out.add_var(k.clone(), v.to_owned())?;
            } else if numeric(v.dtype()) {
                out.add_var(k.clone(), interp_var(v, &dim, &w)?)?;
            }
        }
        out.attrs_mut()
            .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        Ok(out)
    }
}
