//! Calculus along a coordinate (xarray's `differentiate`, `integrate`,
//! `cumulative_integrate`) and least-squares polynomial fits (`polyfit`, `polyval`), with
//! units.
//!
//! - Derivatives are numpy's `gradient`: second-order central differences inside (exact for
//!   uneven spacing), first- or second-order one-sided differences at the ends. Integrals are
//!   the trapezoidal rule (`cumulative_integrate` starts at zero). NaN propagates.
//! - Units: a derivative has the data's unit over the coordinate's (both as differences, so
//!   °C data along a °C axis gives K/K); an integral has the data's unit times the
//!   coordinate's step, and refuses data in an interval or logarithmic unit (°C, dB), whose
//!   integral has no meaning.
//! - Time coordinates (dates, CF dates, time deltas) are steps in seconds unless `time_unit`
//!   names another time unit; the result carries that unit. xarray defaults to the
//!   coordinate's own resolution (nanoseconds) and attaches nothing, so its numbers differ
//!   by the factor the unit states.
//! - `polyfit` is xarray's: a Vandermonde design scaled column by column, solved by SVD with
//!   numpy's `rcond` cut-off (minimum-norm for deficient ranks). NaN values (and NaN
//!   coordinates) are skipped per lane unless `skipna` is `false`, which gives NaN
//!   coefficients for such lanes. Coefficients carry no unit (each degree has another);
//!   their attributes `x_units` and `y_units` record the coordinate's and data's units, so
//!   the coefficient of degree `k` is in `y_units / x_units^k` and [`polyval`] restores the
//!   unit. Time coordinates are fitted in seconds since 1970-01-01.
//! - Deviations from xarray's `polyfit`: residuals and the covariance are computed per lane
//!   from the points actually used (xarray uses the full design for the covariance even
//!   when it skipped NaN values, and gives the unscaled covariance without the other
//!   dimensions); the singular values always run along `degree` (NaN-padded when there are
//!   fewer points than coefficients; xarray pads by the rank deficit and fails on the
//!   length); a scaled covariance from a lane with no more points than coefficients is NaN;
//!   the covariance of a lane whose design is rank-deficient is NaN (xarray's `inv` raises).
//! - Lazy data is read in slabs along another dimension (whole lanes are needed).

use annoray_units::{Unit, UnitError, UnitKind};
use indexmap::IndexMap;
use ndarray::{ArrayD, Axis, IxDyn, Slice};

use crate::lookup::{Keys, index_keys, unix_origin_ns};
use crate::slabs::by_slabs;
use crate::{
    AttrValue, CoreError, DType, DataArray, DataArrayBase, Dataset, DatasetBase, Name, Repr,
    Result, TypedArray, Variable, VariableBase,
};

fn invalid(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(reason.into())
}

/// A one-dimensional coordinate as numbers.
struct Positions {
    dim: Name,
    x: Vec<f64>,
    /// The unit of `x` (absolute: °C stays °C).
    unit: Option<Unit>,
}

/// How to place time coordinates on a number line.
#[derive(Clone, Copy)]
enum Origin {
    /// Relative to the first valid time (steps only matter).
    First,
    /// Since 1970-01-01 (`polyfit`, as xarray).
    Unix,
}

fn positions<C: Repr>(
    name: &str,
    coord: Option<&VariableBase<C>>,
    time_unit: Option<&Unit>,
    origin: Origin,
) -> Result<Positions> {
    let coord = coord.ok_or_else(|| CoreError::NoSuchVariable(Name::from(name)))?;
    let dim = match coord.dims() {
        [d] => d.clone(),
        _ => {
            return Err(invalid(format!(
                "{name:?} must be a one-dimensional coordinate"
            )));
        }
    };
    match index_keys(coord, &dim)? {
        Keys::Num(x) => {
            let unit = coord.unit().cloned();
            let epoch = matches!(unit.as_ref().map(Unit::kind), Some(UnitKind::Epoch(_)));
            match (time_unit, unit) {
                (None, unit) => Ok(Positions { dim, x, unit }),
                (Some(t), Some(u)) if epoch => {
                    let c = u.delta().converter_to(t)?;
                    Ok(Positions {
                        dim,
                        x: x.into_iter().map(|v| c.apply(v)).collect(),
                        unit: Some(t.clone()),
                    })
                }
                (Some(_), _) => Err(invalid(format!(
                    "time_unit applies to time coordinates only, not {name:?}"
                ))),
            }
        }
        Keys::Ns(ns, kind) => {
            let zero = match origin {
                Origin::Unix => {
                    unix_origin_ns(&kind).ok_or(CoreError::ArithmeticOverflow("time"))?
                }
                Origin::First => ns.iter().copied().find(|t| *t != i128::MIN).unwrap_or(0),
            };
            let target = match time_unit {
                Some(t) => t.clone(),
                None => "s".parse::<Unit>()?,
            };
            let c = "ns".parse::<Unit>()?.converter_to(&target)?;
            #[allow(clippy::cast_precision_loss)] // offsets from a nearby origin
            let x = ns
                .iter()
                .map(|t| {
                    if *t == i128::MIN {
                        f64::NAN
                    } else {
                        c.apply(t.saturating_sub(zero) as f64)
                    }
                })
                .collect();
            Ok(Positions {
                dim,
                x,
                unit: Some(target),
            })
        }
        Keys::Str(_) => Err(invalid(format!(
            "cannot do calculus along string labels ({name:?})"
        ))),
    }
}

/// The unit of a derivative: the data's step per coordinate step.
fn per(data: Option<&Unit>, x: Option<&Unit>) -> Result<Option<Unit>> {
    Ok(match (data.map(Unit::delta), x.map(Unit::delta)) {
        (None, None) => None,
        (Some(u), None) => Some(u),
        (None, Some(x)) => Some(x.recip()?),
        (Some(u), Some(x)) => Some(u.div(&x)?),
    })
}

/// The unit of an integral: the data's unit times the coordinate step. Ratio-scale data only.
fn times(data: Option<&Unit>, x: Option<&Unit>) -> Result<Option<Unit>> {
    if let Some(u) = data
        && !matches!(u.kind(), UnitKind::Linear)
    {
        return Err(UnitError::UnsupportedOp {
            op: "integration",
            unit: u.to_string(),
        }
        .into());
    }
    Ok(match (data, x.map(Unit::delta)) {
        (None, None) => None,
        (Some(u), None) => Some(u.clone()),
        (None, Some(x)) => Some(x),
        (Some(u), Some(x)) => Some(u.mul(&x)?),
    })
}

/// Consecutive steps `x[i+1] − x[i]`.
fn steps(x: &[f64]) -> Vec<f64> {
    x.iter().zip(x.iter().skip(1)).map(|(a, b)| b - a).collect()
}

/// numpy's `gradient` of `f` at positions `x` (`f.len() > edge_order`, checked by callers).
fn gradient(f: &[f64], x: &[f64], second_order_edges: bool) -> Vec<f64> {
    let h = steps(x);
    // Even spacing (all steps equal) takes numpy's simpler formulas.
    let uniform = h.first().filter(|h0| h.iter().all(|v| v == *h0));
    let interior = f
        .windows(3)
        .zip(h.windows(2))
        .map(|(w, s)| match (w, s, uniform) {
            ([a, _, c], _, Some(dx)) => (c - a) / (2.0 * dx),
            ([a, b, c], [d1, d2], None) => {
                let ca = -d2 / (d1 * (d1 + d2));
                let cb = (d2 - d1) / (d1 * d2);
                let cc = d1 / (d2 * (d1 + d2));
                ca * a + cb * b + cc * c
            }
            _ => f64::NAN,
        });
    let (first, last) = match (f, h.as_slice()) {
        ([f0, f1, f2, ..], [d1, d2, ..]) if second_order_edges => {
            let (ca, cb, cc) = match uniform {
                Some(dx) => (-1.5 / dx, 2.0 / dx, -0.5 / dx),
                None => (
                    -(2.0 * d1 + d2) / (d1 * (d1 + d2)),
                    (d1 + d2) / (d1 * d2),
                    -d1 / (d2 * (d1 + d2)),
                ),
            };
            let first = ca * f0 + cb * f1 + cc * f2;
            let last = match (f, h.as_slice()) {
                ([.., e0, e1, e2], [.., d1, d2]) => {
                    let (ca, cb, cc) = match uniform {
                        Some(dx) => (0.5 / dx, -2.0 / dx, 1.5 / dx),
                        None => (
                            d2 / (d1 * (d1 + d2)),
                            -(d2 + d1) / (d1 * d2),
                            (2.0 * d2 + d1) / (d2 * (d1 + d2)),
                        ),
                    };
                    ca * e0 + cb * e1 + cc * e2
                }
                _ => f64::NAN,
            };
            (first, last)
        }
        ([f0, f1, ..], [d0, ..]) => {
            let last = match (f, h.as_slice()) {
                ([.., e0, e1], [.., d]) => (e1 - e0) / d,
                _ => f64::NAN,
            };
            ((f1 - f0) / d0, last)
        }
        _ => (f64::NAN, f64::NAN),
    };
    std::iter::once(first)
        .chain(interior)
        .chain(std::iter::once(last))
        .collect()
}

/// The trapezoidal areas between consecutive points.
fn trapezoids<'a>(f: &'a [f64], x: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    f.iter()
        .zip(f.iter().skip(1))
        .zip(steps(x))
        .map(|((a, b), d)| d * (a + b) / 2.0)
}

/// `f` of every lane of `data` along `axis`, giving lanes of `len`.
fn along(
    data: &ArrayD<f64>,
    axis: usize,
    len: usize,
    f: impl Fn(&[f64]) -> Vec<f64>,
) -> Result<ArrayD<f64>> {
    let mut shape = data.shape().to_vec();
    *shape
        .get_mut(axis)
        .ok_or_else(|| invalid("axis out of range"))? = len;
    let mut out = ArrayD::from_elem(IxDyn(&shape), f64::NAN);
    for (src, mut dst) in data
        .lanes(Axis(axis))
        .into_iter()
        .zip(out.lanes_mut(Axis(axis)))
    {
        for (d, v) in dst.iter_mut().zip(f(&src.to_vec())) {
            *d = v;
        }
    }
    Ok(out)
}

/// Numeric values as `f64` (not booleans, strings or times), and whether they were `f32`.
fn numbers<S: Repr>(v: &VariableBase<S>) -> Result<(ArrayD<f64>, bool)> {
    let typed = v.typed()?;
    Ok((
        crate::ops::to_f64(&typed, None)?,
        typed.dtype() == DType::F32,
    ))
}

/// `f` of `da` lane by lane along `dim`: lazy data in slabs along another dimension.
fn lanes<S: Repr>(
    da: &DataArrayBase<S>,
    dim: &Name,
    f: impl Fn(&DataArray) -> Result<DataArray>,
) -> Result<DataArray> {
    match da.dims().iter().find(|d| *d != dim) {
        Some(other) => by_slabs(da, other, 1, f),
        None => f(&da.to_owned()),
    }
}

fn variable(dims: Vec<Name>, data: ArrayD<f64>, f32: bool, unit: Option<Unit>) -> Result<Variable> {
    #[allow(clippy::cast_possible_truncation)] // f32 data stays f32
    let typed = if f32 {
        TypedArray::F32(data.mapv(|x| x as f32))
    } else {
        TypedArray::F64(data)
    };
    let mut v = Variable::from_typed(dims, typed)?;
    v.set_unit(unit);
    Ok(v)
}

fn without_dim<C: Repr>(
    coords: &IndexMap<Name, VariableBase<C>>,
    dim: &Name,
) -> IndexMap<Name, Variable> {
    coords
        .iter()
        .filter(|(_, c)| !c.dims().contains(dim))
        .map(|(k, c)| (k.clone(), c.to_owned()))
        .collect()
}

fn owned_coords<C: Repr>(coords: &IndexMap<Name, VariableBase<C>>) -> IndexMap<Name, Variable> {
    coords
        .iter()
        .map(|(k, c)| (k.clone(), c.to_owned()))
        .collect()
}

impl<S: Repr> DataArrayBase<S> {
    /// The derivative along the one-dimensional coordinate `coord` (numpy's `gradient`):
    /// `edge_order` 1 or 2 sets the accuracy of the one-sided differences at the ends.
    /// `time_unit` (a time unit; seconds by default) sets the step of a time coordinate.
    /// The unit is the data's over the coordinate's; `f32` stays `f32`, integers become
    /// `f64`; coordinates are kept, attributes dropped.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] without such a coordinate;
    /// [`CoreError::InvalidReduction`] for a multi-dimensional or string coordinate, an
    /// `edge_order` other than 1 or 2, fewer than `edge_order + 1` points, or a
    /// `time_unit` along a non-time coordinate; unit errors;
    /// [`CoreError::UnsupportedDType`] for non-numeric data.
    pub fn differentiate(
        &self,
        coord: &str,
        edge_order: usize,
        time_unit: Option<&Unit>,
    ) -> Result<DataArray> {
        if !(1..=2).contains(&edge_order) {
            return Err(invalid(format!(
                "edge_order must be 1 or 2, not {edge_order}"
            )));
        }
        let p = positions(coord, self.coords().get(coord), time_unit, Origin::First)?;
        let n = self.variable().len_of(&p.dim)?;
        if n <= edge_order {
            return Err(invalid(format!(
                "a derivative with edge_order {edge_order} needs at least {} points along {:?}",
                edge_order.saturating_add(1),
                p.dim
            )));
        }
        let unit = per(self.unit(), p.unit.as_ref())?;
        lanes(self, &p.dim, |da| {
            let v = da.variable();
            let (data, f32) = numbers(v)?;
            let out = along(&data, v.axis(&p.dim)?, n, |f| {
                gradient(f, &p.x, edge_order == 2)
            })?;
            Ok(DataArrayBase::from_parts(
                da.name().cloned(),
                variable(v.dims().to_vec(), out, f32, unit.clone())?,
                da.coords().clone(),
            ))
        })
    }

    /// The integral by the trapezoidal rule along each of `coords` in turn (one-dimensional
    /// coordinates), removing their dimensions. Units multiply by the coordinate step
    /// (`time_unit`, seconds by default, for times). The result is `f64`; coordinates along
    /// the integrated dimensions and attributes are dropped.
    ///
    /// # Errors
    /// As for [`differentiate`](Self::differentiate) (no minimum length);
    /// [`UnitError::UnsupportedOp`] for data in an interval or logarithmic unit.
    pub fn integrate(&self, coords: &[&str], time_unit: Option<&Unit>) -> Result<DataArray> {
        let mut out = self.to_owned();
        for c in coords {
            out = integrate_once(&out, c, time_unit)?;
        }
        Ok(out)
    }

    /// The running integral by the trapezoidal rule along each of `coords` in turn,
    /// starting from zero (scipy's `cumulative_trapezoid` with `initial=0`). Dimensions and
    /// coordinates are kept; units as for [`integrate`](Self::integrate).
    ///
    /// # Errors
    /// As for [`integrate`](Self::integrate).
    pub fn cumulative_integrate(
        &self,
        coords: &[&str],
        time_unit: Option<&Unit>,
    ) -> Result<DataArray> {
        let mut out = self.to_owned();
        for c in coords {
            out = cumulative_once(&out, c, time_unit)?;
        }
        Ok(out)
    }

    /// A least-squares polynomial fit of degree `deg` along the one-dimensional coordinate
    /// `coord` (xarray's `polyfit`): a dataset with `polyfit_coefficients` along `degree`
    /// (highest first) and the other dimensions, plus what `opts` asks for (residuals, rank
    /// and singular values, covariance). See the module notes on units and NaN.
    ///
    /// # Errors
    /// As for [`differentiate`](Self::differentiate); [`CoreError::InvalidReduction`] for
    /// weights of the wrong length, or a scaled covariance with no more points than
    /// coefficients.
    pub fn polyfit(&self, coord: &str, deg: usize, opts: &PolyfitOptions) -> Result<Dataset> {
        let design = Design::new(coord, self.coords().get(coord), deg, opts, self.dims())?;
        let mut out = Dataset::new();
        design.add_full(&mut out)?;
        design.fit_into(&mut out, "", self)?;
        for d in self.dims() {
            if *d != design.dim
                && let Some(c) = self.coords().get(d)
                && c.dims() == [d.clone()]
            {
                out.add_coord(d.clone(), c.to_owned())?;
            }
        }
        Ok(out)
    }
}

fn integrate_once<S: Repr>(
    da: &DataArrayBase<S>,
    coord: &str,
    time_unit: Option<&Unit>,
) -> Result<DataArray> {
    let p = positions(coord, da.coords().get(coord), time_unit, Origin::First)?;
    let unit = times(da.unit(), p.unit.as_ref())?;
    lanes(da, &p.dim, |slab| {
        let v = slab.variable();
        let axis = v.axis(&p.dim)?;
        let (data, _) = numbers(v)?;
        let out = along(&data, axis, 1, |f| vec![trapezoids(f, &p.x).sum()])?;
        let dims = v.dims().iter().filter(|d| **d != p.dim).cloned().collect();
        Ok(DataArrayBase::from_parts(
            slab.name().cloned(),
            variable(
                dims,
                out.index_axis_move(Axis(axis), 0),
                false,
                unit.clone(),
            )?,
            without_dim(slab.coords(), &p.dim),
        ))
    })
}

fn cumulative_once<S: Repr>(
    da: &DataArrayBase<S>,
    coord: &str,
    time_unit: Option<&Unit>,
) -> Result<DataArray> {
    let p = positions(coord, da.coords().get(coord), time_unit, Origin::First)?;
    let unit = times(da.unit(), p.unit.as_ref())?;
    let n = da.variable().len_of(&p.dim)?;
    lanes(da, &p.dim, |slab| {
        let v = slab.variable();
        let (data, _) = numbers(v)?;
        let out = along(&data, v.axis(&p.dim)?, n, |f| {
            let mut acc = 0.0;
            std::iter::once(0.0)
                .chain(trapezoids(f, &p.x).map(|a| {
                    acc += a;
                    acc
                }))
                .collect()
        })?;
        Ok(DataArrayBase::from_parts(
            slab.name().cloned(),
            variable(v.dims().to_vec(), out, false, unit.clone())?,
            slab.coords().clone(),
        ))
    })
}

impl<S: Repr> DatasetBase<S> {
    /// [`DataArrayBase::differentiate`] of every data variable along the coordinate's
    /// dimension; the others are kept.
    ///
    /// # Errors
    /// As for [`DataArrayBase::differentiate`].
    pub fn differentiate(
        &self,
        coord: &str,
        edge_order: usize,
        time_unit: Option<&Unit>,
    ) -> Result<Dataset> {
        let p = positions(coord, self.coords().get(coord), time_unit, Origin::First)?;
        crate::reshape::per_variable(self, &p.dim, |da| {
            da.differentiate(coord, edge_order, time_unit)
        })
    }

    /// [`DataArrayBase::integrate`] of every data variable along the coordinates'
    /// dimensions; variables without them are kept, coordinates along them dropped.
    ///
    /// # Errors
    /// As for [`DataArrayBase::integrate`].
    pub fn integrate(&self, coords: &[&str], time_unit: Option<&Unit>) -> Result<Dataset> {
        let mut out = self.to_owned();
        for c in coords {
            let p = positions(c, out.coords().get(*c), time_unit, Origin::First)?;
            let mut next = Dataset::new();
            for (k, v) in out.coords() {
                if !v.dims().contains(&p.dim) {
                    next.add_coord(k.clone(), v.clone())?;
                }
            }
            for k in out.data_vars().keys() {
                let da = out.get(k)?;
                let v = if da.dims().contains(&p.dim) {
                    integrate_once(&da, c, time_unit)?.variable().to_owned()
                } else {
                    da.variable().to_owned()
                };
                next.add_var(k.clone(), v)?;
            }
            *next.attrs_mut() = out.attrs().clone();
            out = next;
        }
        Ok(out)
    }

    /// [`DataArrayBase::cumulative_integrate`] of every data variable along the
    /// coordinates' dimensions; the others are kept.
    ///
    /// # Errors
    /// As for [`DataArrayBase::cumulative_integrate`].
    pub fn cumulative_integrate(
        &self,
        coords: &[&str],
        time_unit: Option<&Unit>,
    ) -> Result<Dataset> {
        let mut out = self.to_owned();
        for c in coords {
            let p = positions(c, out.coords().get(*c), time_unit, Origin::First)?;
            out =
                crate::reshape::per_variable(&out, &p.dim, |da| cumulative_once(da, c, time_unit))?;
        }
        Ok(out)
    }

    /// [`DataArrayBase::polyfit`] of every data variable along `coord`'s dimension, as
    /// `<name>_polyfit_coefficients` (and `_residuals`, `_covariance`); variables without
    /// the dimension are left out. Attributes are kept.
    ///
    /// # Errors
    /// As for [`DataArrayBase::polyfit`].
    pub fn polyfit(&self, coord: &str, deg: usize, opts: &PolyfitOptions) -> Result<Dataset> {
        let dims = self.dims();
        let design = Design::new(coord, self.coords().get(coord), deg, opts, &dims)?;
        let mut out = Dataset::new();
        design.add_full(&mut out)?;
        let mut present: Vec<Name> = Vec::new();
        for k in self.data_vars().keys() {
            let da = self.get(k)?;
            if !da.dims().contains(&design.dim) {
                continue;
            }
            design.fit_into(&mut out, &format!("{k}_"), &da)?;
            for d in da.dims() {
                if *d != design.dim && !present.contains(d) {
                    present.push(d.clone());
                }
            }
        }
        for d in dims {
            if present.contains(&d)
                && let Some(c) = self.coords().get(&d)
                && c.dims() == [d.clone()]
            {
                out.add_coord(d.clone(), c.to_owned())?;
            }
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }
}

/// Which covariance matrix of the coefficients [`polyfit`](DataArrayBase::polyfit) returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Covariance {
    /// None (the default).
    #[default]
    None,
    /// Scaled by the residual variance (`cov=True`).
    Scaled,
    /// Unscaled (`cov="unscaled"`).
    Unscaled,
}

/// Options of [`polyfit`](DataArrayBase::polyfit), as xarray's keywords.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PolyfitOptions {
    skipna: Option<bool>,
    rcond: Option<f64>,
    weights: Option<Vec<f64>>,
    full: bool,
    cov: Covariance,
}

impl PolyfitOptions {
    /// The defaults: NaN skipped, numpy's `rcond`, no weights, coefficients only.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Skip NaN values per lane (the default; `false` gives NaN coefficients for lanes with
    /// NaN).
    #[must_use]
    pub fn skipna(mut self, skipna: bool) -> Self {
        self.skipna = Some(skipna);
        self
    }

    /// Singular values below `rcond` times the largest are treated as zero (default: the
    /// number of points times machine epsilon).
    #[must_use]
    pub fn rcond(mut self, rcond: f64) -> Self {
        self.rcond = Some(rcond);
        self
    }

    /// Weights of the points (multiplying the residuals, as numpy's `w`), one per position
    /// along the dimension.
    #[must_use]
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Also return the residuals (`<name>_polyfit_residuals`), the design's rank
    /// (`<coord>_matrix_rank`) and its singular values (`<coord>_singular_values`).
    #[must_use]
    pub fn full(mut self, full: bool) -> Self {
        self.full = full;
        self
    }

    /// Also return the covariance of the coefficients (`<name>_polyfit_covariance`, along
    /// `cov_i` and `cov_j`); the scaled one adds the residuals.
    #[must_use]
    pub fn cov(mut self, cov: Covariance) -> Self {
        self.cov = cov;
        self
    }
}

/// The scaled Vandermonde design of a fit.
struct Design {
    dim: Name,
    coord: Name,
    degree: Name,
    order: usize,
    /// Columns (degree `order − 1` first) of the weighted design divided by `scale`; rows
    /// with a NaN coordinate or weight are NaN.
    lhs: Vec<Vec<f64>>,
    scale: Vec<f64>,
    weights: Option<Vec<f64>>,
    rcond: f64,
    skipna: bool,
    full: bool,
    cov: Covariance,
    x_unit: Option<String>,
}

impl Design {
    fn new<C: Repr>(
        name: &str,
        coord: Option<&VariableBase<C>>,
        deg: usize,
        opts: &PolyfitOptions,
        dims: &[Name],
    ) -> Result<Self> {
        let p = positions(name, coord, None, Origin::Unix)?;
        let time = coord.is_some_and(|c| {
            matches!(
                c.dtype(),
                DType::DateTime | DType::CfDate | DType::TimeDelta
            )
        });
        let x_unit = match (time, coord.map(VariableBase::dtype)) {
            (true, Some(DType::TimeDelta)) => Some("s".to_owned()),
            (true, _) => Some("seconds since 1970-01-01 00:00:00".to_owned()),
            (false, _) => p.unit.as_ref().map(ToString::to_string),
        };
        let order = deg
            .checked_add(1)
            .ok_or(CoreError::ArithmeticOverflow("polyfit degree"))?;
        let m = p.x.len();
        if let Some(w) = &opts.weights
            && w.len() != m
        {
            return Err(invalid(format!(
                "{} weights for {m} points along {:?}",
                w.len(),
                p.dim
            )));
        }
        if opts.cov == Covariance::Scaled && m <= order {
            return Err(invalid(
                "the number of points must exceed the number of coefficients to scale the covariance",
            ));
        }
        let mut lhs = Vec::with_capacity(order);
        for k in (0..order).rev() {
            let k = i32::try_from(k).map_err(|_| CoreError::ArithmeticOverflow("polyfit"))?;
            let col: Vec<f64> =
                p.x.iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let w = opts
                            .weights
                            .as_ref()
                            .and_then(|w| w.get(i))
                            .copied()
                            .unwrap_or(1.0);
                        if x.is_finite() && w.is_finite() {
                            x.powi(k) * w
                        } else {
                            f64::NAN
                        }
                    })
                    .collect();
            lhs.push(col);
        }
        let scale: Vec<f64> = lhs
            .iter()
            .map(|c| {
                let s = c
                    .iter()
                    .filter(|v| !v.is_nan())
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
                if s > 0.0 && s.is_finite() { s } else { 1.0 }
            })
            .collect();
        for (c, s) in lhs.iter_mut().zip(&scale) {
            c.iter_mut().for_each(|v| *v /= s);
        }
        #[allow(clippy::cast_precision_loss)]
        let rcond = opts.rcond.unwrap_or(m as f64 * f64::EPSILON);
        let mut degree = String::from("degree");
        while dims.iter().any(|d| d == degree.as_str()) {
            degree.insert(0, '_');
        }
        Ok(Self {
            dim: p.dim,
            coord: Name::from(name),
            degree: Name::from(degree.as_str()),
            order,
            lhs,
            scale,
            weights: opts.weights.clone(),
            rcond,
            skipna: opts.skipna.unwrap_or(true),
            full: opts.full,
            cov: opts.cov,
            x_unit,
        })
    }

    /// The `degree` coordinate, highest first.
    fn degrees(&self) -> Result<Variable> {
        let d = (0..self.order)
            .rev()
            .map(|k| i64::try_from(k).map_err(|_| CoreError::ArithmeticOverflow("degree")))
            .collect::<Result<Vec<i64>>>()?;
        Variable::new(
            [self.degree.clone()],
            ArrayD::from_shape_vec(IxDyn(&[self.order]), d).map_err(|e| invalid(e.to_string()))?,
        )
    }

    /// The rank and singular values of the design (with `full`), and the `degree` coordinate.
    fn add_full(&self, out: &mut Dataset) -> Result<()> {
        out.add_coord(self.degree.clone(), self.degrees()?)?;
        if !self.full {
            return Ok(());
        }
        let rows: Vec<bool> = (0..self.lhs.first().map_or(0, Vec::len))
            .map(|i| {
                self.lhs
                    .iter()
                    .all(|c| c.get(i).is_some_and(|v| !v.is_nan()))
            })
            .collect();
        let a = self.rows(&rows);
        let m = rows.iter().filter(|r| **r).count();
        let (mut sigma, _, _) = svd(a);
        sigma.sort_by(|p, q| q.total_cmp(p));
        let smax = sigma.first().copied().unwrap_or(0.0);
        #[allow(clippy::cast_precision_loss)]
        let tol = smax * m.max(self.order) as f64 * f64::EPSILON;
        let rank = sigma.iter().filter(|s| **s > tol).count();
        // Fewer points than coefficients: the missing singular values are NaN.
        let sigma: Vec<f64> = (0..self.order)
            .map(|i| {
                if i < m {
                    sigma.get(i).copied().unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                }
            })
            .collect();
        let rank = i64::try_from(rank).map_err(|_| CoreError::ArithmeticOverflow("rank"))?;
        out.add_var(
            format!("{}_matrix_rank", self.coord),
            Variable::new(Vec::<Name>::new(), ArrayD::from_elem(IxDyn(&[]), rank))?,
        )?;
        out.add_var(
            format!("{}_singular_values", self.coord),
            Variable::new(
                [self.degree.clone()],
                ArrayD::from_shape_vec(IxDyn(&[self.order]), sigma)
                    .map_err(|e| invalid(e.to_string()))?,
            )?,
        )
    }

    /// The design's columns restricted to the rows marked `true`.
    fn rows(&self, keep: &[bool]) -> Vec<Vec<f64>> {
        self.lhs
            .iter()
            .map(|c| {
                c.iter()
                    .zip(keep)
                    .filter(|(_, k)| **k)
                    .map(|(v, _)| *v)
                    .collect()
            })
            .collect()
    }

    /// One lane's packed result: coefficients (highest degree first), the residual sum of
    /// squares, the covariance (row-major).
    fn fit(&self, y: &[f64]) -> Vec<f64> {
        let n = self.order;
        let packed = n.saturating_add(1).saturating_add(n.saturating_mul(n));
        let nan = vec![f64::NAN; packed];
        let keep: Vec<bool> = y
            .iter()
            .enumerate()
            .map(|(i, v)| {
                !v.is_nan()
                    && self
                        .lhs
                        .iter()
                        .all(|c| c.get(i).is_some_and(|x| !x.is_nan()))
            })
            .collect();
        let m = keep.iter().filter(|k| **k).count();
        if m == 0 || (!self.skipna && m < y.len()) {
            return nan;
        }
        let a = self.rows(&keep);
        let b: Vec<f64> = y
            .iter()
            .enumerate()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|((i, v), _)| {
                v * self
                    .weights
                    .as_ref()
                    .and_then(|w| w.get(i))
                    .copied()
                    .unwrap_or(1.0)
            })
            .collect();
        let (sigma, u, v) = svd(a.clone());
        let smax = sigma.iter().copied().fold(0.0, f64::max);
        let cut = self.rcond * smax;
        let used: Vec<bool> = sigma.iter().map(|s| *s > cut && *s > 0.0).collect();
        let rank = used.iter().filter(|k| **k).count();
        // c = V Σ⁺ Uᵀ b (in scaled columns).
        let mut c = vec![0.0; n];
        for ((s, uj), (vj, k)) in sigma.iter().zip(&u).zip(v.iter().zip(&used)) {
            if *k {
                let t = uj.iter().zip(&b).map(|(p, q)| p * q).sum::<f64>() / s;
                c.iter_mut().zip(vj).for_each(|(ci, vi)| *ci += t * vi);
            }
        }
        let resid = if rank == n && m > n {
            let fitted = (0..m).map(|r| {
                a.iter()
                    .zip(&c)
                    .map(|(col, ci)| col.get(r).copied().unwrap_or(f64::NAN) * ci)
                    .sum::<f64>()
            });
            fitted.zip(&b).map(|(f, y)| (y - f) * (y - f)).sum()
        } else {
            f64::NAN
        };
        let mut out: Vec<f64> = c.iter().zip(&self.scale).map(|(ci, s)| ci / s).collect();
        out.push(resid);
        if self.cov == Covariance::None {
            out.extend(std::iter::repeat_n(f64::NAN, n.saturating_mul(n)));
            return out;
        }
        #[allow(clippy::cast_precision_loss)]
        let fac = match self.cov {
            Covariance::Scaled if m > n => resid / (m.saturating_sub(n)) as f64,
            Covariance::Scaled => f64::NAN,
            _ => 1.0,
        };
        // (AᵀA)⁻¹ = V Σ⁻² Vᵀ, unscaled back to the coordinate's columns.
        for (i, si) in self.scale.iter().enumerate() {
            for (j, sj) in self.scale.iter().enumerate() {
                let x = if rank < n {
                    f64::NAN
                } else {
                    sigma
                        .iter()
                        .zip(&v)
                        .map(|(s, vk)| {
                            vk.get(i).copied().unwrap_or(f64::NAN)
                                * vk.get(j).copied().unwrap_or(f64::NAN)
                                / (s * s)
                        })
                        .sum::<f64>()
                };
                out.push(x / (si * sj) * fac);
            }
        }
        out
    }

    /// Fit every lane of `da` and add the results as `<prefix>polyfit_*`.
    fn fit_into<S: Repr>(
        &self,
        out: &mut Dataset,
        prefix: &str,
        da: &DataArrayBase<S>,
    ) -> Result<()> {
        let n = self.order;
        let rows = n.saturating_add(1).saturating_add(n.saturating_mul(n));
        let packed_dim = Name::from("__polyfit__");
        let others: Vec<Name> = da
            .dims()
            .iter()
            .filter(|d| **d != self.dim)
            .cloned()
            .collect();
        let packed = lanes(da, &self.dim, |slab| {
            let v = slab.variable();
            let (data, _) = numbers(v)?;
            let axis = v.axis(&self.dim)?;
            let mut shape = vec![rows];
            shape.extend(
                data.shape()
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != axis)
                    .map(|(_, s)| *s),
            );
            let mut out = ArrayD::from_elem(IxDyn(&shape), f64::NAN);
            for (src, mut dst) in data
                .lanes(Axis(axis))
                .into_iter()
                .zip(out.lanes_mut(Axis(0)))
            {
                for (d, x) in dst.iter_mut().zip(self.fit(&src.to_vec())) {
                    *d = x;
                }
            }
            let dims: Vec<Name> = std::iter::once(packed_dim.clone())
                .chain(v.dims().iter().filter(|d| **d != self.dim).cloned())
                .collect();
            Ok(DataArray::new(Variable::new(dims, out)?))
        })?;
        let all = packed.variable().to_f64()?;
        let rest: Vec<usize> = all.shape().iter().skip(1).copied().collect();
        let part =
            |from: usize, to: usize| all.slice_axis(Axis(0), Slice::from(from..to)).to_owned();
        let mut coeffs = Variable::new(
            std::iter::once(self.degree.clone())
                .chain(others.iter().cloned())
                .collect::<Vec<_>>(),
            part(0, n),
        )?;
        let attrs = coeffs.attrs_mut();
        if let Some(u) = &self.x_unit {
            attrs.insert(Name::from("x_units"), AttrValue::Str(u.clone()));
        }
        if let Some(u) = da.unit() {
            attrs.insert(Name::from("y_units"), AttrValue::Str(u.to_string()));
        }
        out.add_var(format!("{prefix}polyfit_coefficients"), coeffs)?;
        if self.full || self.cov == Covariance::Scaled {
            let mut r = Variable::new(others.clone(), all.index_axis(Axis(0), n).to_owned())?;
            r.set_unit(match da.unit() {
                Some(u) => Some(u.delta().powi(2)?),
                None => None,
            });
            out.add_var(format!("{prefix}polyfit_residuals"), r)?;
        }
        if self.cov != Covariance::None {
            let from = n.saturating_add(1);
            let flat = part(from, rows);
            let mut shape = vec![n, n];
            shape.extend(&rest);
            let cov = ArrayD::from_shape_vec(IxDyn(&shape), flat.iter().copied().collect())
                .map_err(|e| invalid(e.to_string()))?;
            let dims: Vec<Name> = [Name::from("cov_i"), Name::from("cov_j")]
                .into_iter()
                .chain(others.iter().cloned())
                .collect();
            out.add_var(
                format!("{prefix}polyfit_covariance"),
                Variable::new(dims, cov)?,
            )?;
        }
        Ok(())
    }
}

/// The singular value decomposition of the matrix with columns `a` (one-sided Jacobi):
/// singular values, left vectors (columns; zero for a zero singular value) and right
/// vectors (columns), unsorted and paired by position.
fn svd(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let dot = |p: &[f64], q: &[f64]| p.iter().zip(q).map(|(x, y)| x * y).sum::<f64>();
    let rotate = |p: &mut [f64], q: &mut [f64], c: f64, s: f64| {
        for (x, y) in p.iter_mut().zip(q.iter_mut()) {
            let (a, b) = (*x, *y);
            *x = c * a - s * b;
            *y = s * a + c * b;
        }
    };
    for _ in 0..100 {
        let mut rotated = false;
        for q in 1..n {
            for p in 0..q {
                let (left, right) = a.split_at_mut(q);
                let (Some(ap), Some(aq)) = (left.get_mut(p), right.first_mut()) else {
                    continue;
                };
                let (alpha, beta, gamma) = (dot(ap, ap), dot(aq, aq), dot(ap, aq));
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + zeta.hypot(1.0));
                let c = 1.0 / t.hypot(1.0);
                let s = c * t;
                rotate(ap, aq, c, s);
                let (left, right) = v.split_at_mut(q);
                if let (Some(vp), Some(vq)) = (left.get_mut(p), right.first_mut()) {
                    rotate(vp, vq, c, s);
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let sigma: Vec<f64> = a.iter().map(|c| dot(c, c).sqrt()).collect();
    let u = a
        .into_iter()
        .zip(&sigma)
        .map(|(c, s)| {
            if *s > 0.0 {
                c.into_iter().map(|x| x / s).collect()
            } else {
                vec![0.0; c.len()]
            }
        })
        .collect();
    (sigma, u, v)
}

/// Evaluate polynomials at `coord` (xarray's `polyval`): `coeffs` holds coefficients along
/// `degree_dim`, whose coordinate gives each one's degree (missing degrees are zero). With
/// the `x_units`/`y_units` attributes [`polyfit`](DataArrayBase::polyfit) records, `coord`
/// is converted to the fit's unit first and the result carries the data's unit; time
/// coordinates are taken in seconds since 1970-01-01 (time deltas in seconds), as fitted.
///
/// A coordinate without a unit is taken to be in `x_units` as it stands (no check is
/// possible); a coordinate with a unit must convert to it. Coefficients without the
/// attributes evaluate on the coordinate's numbers and give a result without a unit.
///
/// # Errors
/// [`CoreError::NoIndex`] without a `degree_dim` coordinate; [`CoreError::InvalidReduction`]
/// for negative, fractional or repeated degrees, or multi-dimensional times; unit errors.
pub fn polyval<C: Repr, R: Repr>(
    coord: &DataArrayBase<C>,
    coeffs: &DataArrayBase<R>,
    degree_dim: &str,
) -> Result<DataArray> {
    let labels = coeffs
        .coords()
        .get(degree_dim)
        .ok_or_else(|| CoreError::NoIndex(Name::from(degree_dim)))?
        .to_f64()?;
    let mut by_degree: IndexMap<usize, isize> = IndexMap::new();
    for (pos, d) in labels.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = if *d >= 0.0 && d.fract() == 0.0 && *d < 1e9 {
            *d as usize
        } else {
            return Err(invalid(format!("{d} is not a polynomial degree")));
        };
        let pos = isize::try_from(pos).map_err(|_| CoreError::ArithmeticOverflow("degree"))?;
        if by_degree.insert(k, pos).is_some() {
            return Err(invalid(format!("degree {k} appears twice")));
        }
    }
    let unit_attr = |key: &str| -> Result<Option<Unit>> {
        match coeffs.attrs().get(key) {
            Some(AttrValue::Str(s)) => Ok(Some(s.parse::<Unit>()?)),
            _ => Ok(None),
        }
    };
    let (x_unit, y_unit) = (unit_attr("x_units")?, unit_attr("y_units")?);
    let var = coord.variable();
    let x = if matches!(
        var.dtype(),
        DType::DateTime | DType::CfDate | DType::TimeDelta
    ) {
        let name = var
            .dims()
            .first()
            .cloned()
            .ok_or_else(|| invalid("polyval of a scalar time"))?;
        let p = positions("coordinate", Some(var), None, Origin::Unix)?;
        if var.ndim() != 1 {
            return Err(invalid("times for polyval must be one-dimensional"));
        }
        Variable::new(
            [name],
            ArrayD::from_shape_vec(IxDyn(&[p.x.len()]), p.x).map_err(|e| invalid(e.to_string()))?,
        )?
    } else {
        let mut values = var.to_f64()?;
        if let (Some(from), Some(to)) = (var.unit(), &x_unit)
            && from != to
        {
            let c = from.converter_to(to)?;
            values.mapv_inplace(|v| c.apply(v));
        }
        Variable::new(var.dims().to_vec(), values)?
    };
    let x = DataArrayBase::from_parts(None, x, owned_coords(coord.coords()));
    let term = |k: usize| -> Result<Option<DataArray>> {
        let Some(pos) = by_degree.get(&k) else {
            return Ok(None);
        };
        let c = coeffs.isel([(degree_dim, *pos)])?;
        let mut v = c.variable().to_owned();
        v.set_unit(None);
        v.attrs_mut().clear();
        let coords = c
            .coords()
            .iter()
            .filter(|(k, _)| k.as_str() != degree_dim)
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(Some(DataArrayBase::from_parts(None, v, coords)))
    };
    let top = by_degree.keys().copied().max().unwrap_or(0);
    let mut acc = match term(top)? {
        Some(t) => t,
        None => return Err(invalid("no coefficients")),
    };
    for k in (0..top).rev() {
        acc = acc.try_mul(&x)?;
        if let Some(t) = term(k)? {
            acc = acc.try_add(&t)?;
        }
    }
    acc.set_unit(y_unit);
    Ok(acc)
}

/// [`polyval`] of every variable of `coeffs` along `degree_dim` (a `polyfit` result, say),
/// under the same names; variables without `degree_dim` are left out.
///
/// # Errors
/// As for [`polyval`].
pub fn polyval_dataset<C: Repr, R: Repr>(
    coord: &DataArrayBase<C>,
    coeffs: &DatasetBase<R>,
    degree_dim: &str,
) -> Result<Dataset> {
    let mut out = Dataset::new();
    for k in coeffs.data_vars().keys() {
        let da = coeffs.get(k)?;
        if !da.dims().iter().any(|d| d == degree_dim) {
            continue;
        }
        let v = polyval(coord, &da, degree_dim)?;
        for (ck, c) in v.coords() {
            if !out.coords().contains_key(ck) {
                out.add_coord(ck.clone(), c.clone())?;
            }
        }
        out.add_var(k.clone(), v.variable().clone())?;
    }
    Ok(out)
}
