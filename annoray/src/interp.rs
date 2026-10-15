//! Interpolation along dimension coordinates (xarray's `interp`, as numpy's `interp` and
//! scipy's `interp1d` do it).
//!
//! - **Linear** and **nearest** (ties go to the lower coordinate), and the logarithmic
//!   variants: **log-coordinate** (linear in `ln x`), **log-value** (`ln y` linear in `x`:
//!   exponential profiles such as density against altitude) and **log-log** (`ln y` linear in
//!   `ln x`: power laws such as cross sections against energy).
//! - Outside the coordinate's range, by [`InterpOptions`] and separately below and above:
//!   NaN (the default, as xarray), the edge value, or the edge interval's law extended (a
//!   straight line, exponential or power law, by method). NaN targets give NaN.
//! - Log coordinates must be positive numbers (not times) in a linear unit. Log values need a
//!   linear unit (not °C or dates); between two values of which either is not positive the
//!   result is NaN, since the logarithm is undefined. A target exactly on a coordinate always
//!   takes that value. Coordinates other than the index along the dimension are interpolated
//!   linearly in value (with the method's weights).
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
    /// Linear in the logarithm of the coordinate (e.g. along pressure levels).
    LogCoord,
    /// The logarithm of the values linear in the coordinate: exponential between points
    /// (e.g. density against altitude).
    LogValue,
    /// The logarithm of the values linear in the logarithm of the coordinate: a power law
    /// between points (e.g. cross sections against energy).
    LogLog,
}

/// What interpolation gives outside the coordinate's range.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum Extrapolate {
    /// NaN (the default).
    #[default]
    Nan,
    /// The value at the nearest end of the coordinate.
    Edge,
    /// The edge interval's law continued: a straight line for linear interpolation, an
    /// exponential for `LogValue`, a power law for `LogLog`, linear in `ln x` for `LogCoord`.
    /// (For `Nearest`, and with a single point, the edge value.)
    Extend,
    /// A power law from the edge point: `y = y_edge · (x / x_edge)^k`, whatever the method
    /// (e.g. a tabulated high-energy exponent: `y_last · (x_last / x)^n` is
    /// `PowerLaw(-n)`). Needs positive, non-time coordinates and targets; values in a
    /// ratio-scale unit. Coordinates other than the index take the edge value.
    PowerLaw(f64),
}

/// How to interpolate: the method, and what to do below and above the coordinate's range.
/// An [`InterpMethod`] converts into options without extrapolation.
///
/// ```
/// use annoray::{Extrapolate, InterpMethod, InterpOptions};
///
/// // Power law within a table; hold the first value below it, extend the law above it.
/// let opts = InterpOptions::new(InterpMethod::LogLog)
///     .below(Extrapolate::Edge)
///     .above(Extrapolate::Extend);
/// assert_eq!(opts.method, InterpMethod::LogLog);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct InterpOptions {
    /// The interpolation method.
    pub method: InterpMethod,
    /// Below the smallest coordinate.
    pub below: Extrapolate,
    /// Above the largest coordinate.
    pub above: Extrapolate,
}

impl InterpOptions {
    /// `method` without extrapolation.
    #[must_use]
    pub fn new(method: InterpMethod) -> Self {
        Self {
            method,
            ..Self::default()
        }
    }

    /// The same extrapolation on both sides.
    #[must_use]
    pub fn extrapolate(self, both: Extrapolate) -> Self {
        Self {
            below: both,
            above: both,
            ..self
        }
    }

    /// Extrapolation below the smallest coordinate.
    #[must_use]
    pub fn below(self, below: Extrapolate) -> Self {
        Self { below, ..self }
    }

    /// Extrapolation above the largest coordinate.
    #[must_use]
    pub fn above(self, above: Extrapolate) -> Self {
        Self { above, ..self }
    }
}

impl From<InterpMethod> for InterpOptions {
    fn from(method: InterpMethod) -> Self {
        Self::new(method)
    }
}

impl InterpMethod {
    const fn log_coord(self) -> bool {
        matches!(self, Self::LogCoord | Self::LogLog)
    }

    const fn log_values(self) -> bool {
        matches!(self, Self::LogValue | Self::LogLog)
    }
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

/// For one target: the two source positions, the weight of the second, and a factor applied
/// to data values (a power-law extrapolation from one point; else 1).
type Pick = (usize, usize, f64, f64);

/// How to interpolate one dimension: a [`Pick`] per target (`None` for NaN); and whether
/// values are interpolated in log space.
struct Weights {
    at: Vec<Option<Pick>>,
    log_values: bool,
    power_law: bool,
}

fn weights(x: &[f64], targets: &[f64], opts: &InterpOptions) -> Vec<Option<Pick>> {
    let method = opts.method;
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
    // The weight of the upper point of the interval `[xl, xh]` at `t` (any `t`: outside the
    // interval it extrapolates). Log coordinates are positive (checked when planning).
    let fraction = |t: f64, xl: f64, xh: f64| -> Option<f64> {
        if method.log_coord() {
            (t > 0.0).then(|| (t / xl).ln() / (xh / xl).ln())
        } else {
            Some((t - xl) / (xh - xl))
        }
    };
    let last_pos = xs.len().saturating_sub(1);
    targets
        .iter()
        .map(|&t| {
            if t.is_nan() {
                return None;
            }
            if t < first || t > last {
                // Outside: the edge point, and the interval next to it.
                let (side, edge, inner) = if t < first {
                    (opts.below, 0, 1)
                } else {
                    (opts.above, last_pos, last_pos.saturating_sub(1))
                };
                let ie = order.get(edge).copied()?;
                return match side {
                    Extrapolate::Nan => None,
                    Extrapolate::Edge => Some((ie, ie, 0.0, 1.0)),
                    Extrapolate::PowerLaw(k) => {
                        let xe = xs.get(edge).copied()?;
                        (t > 0.0 && xe > 0.0).then(|| (ie, ie, 0.0, (t / xe).powf(k)))
                    }
                    Extrapolate::Extend => {
                        if xs.len() < 2 || method == InterpMethod::Nearest {
                            return Some((ie, ie, 0.0, 1.0));
                        }
                        let (lo, hi) = (edge.min(inner), edge.max(inner));
                        let (xl, xh) = (xs.get(lo).copied()?, xs.get(hi).copied()?);
                        let (il, ih) = (order.get(lo).copied()?, order.get(hi).copied()?);
                        Some((il, ih, fraction(t, xl, xh)?, 1.0))
                    }
                };
            }
            // `hi`: the first sorted position strictly above `t` (or the last one).
            let hi = xs.partition_point(|v| *v <= t).min(last_pos);
            let lo = hi.saturating_sub(1);
            let (xl, xh) = (xs.get(lo).copied()?, xs.get(hi).copied()?);
            let (il, ih) = (order.get(lo).copied()?, order.get(hi).copied()?);
            if t == xh {
                return Some((ih, ih, 0.0, 1.0));
            }
            if t == xl {
                return Some((il, il, 0.0, 1.0));
            }
            if method == InterpMethod::Nearest {
                let i = if xh - t < t - xl { ih } else { il };
                return Some((i, i, 0.0, 1.0));
            }
            Some((il, ih, fraction(t, xl, xh)?, 1.0))
        })
        .collect()
}

/// Interpolate one variable's values along `axis`.
fn apply<S: Repr>(
    v: &VariableBase<S>,
    axis: usize,
    w: &Weights,
    log_values: bool,
    scale_values: bool,
) -> Result<TypedArray<crate::Owned>> {
    let typed = v.typed()?;
    let f32_data = typed.dtype() == DType::F32;
    let data = to_f64(&typed, None)?;
    let mut shape = data.shape().to_vec();
    if let Some(s) = shape.get_mut(axis) {
        *s = w.at.len();
    }
    let mut out = ArrayD::from_elem(IxDyn(&shape), f64::NAN);
    for (k, wk) in w.at.iter().enumerate() {
        let Some((lo, hi, t, factor)) = *wk else {
            continue;
        };
        let mut dst = out.index_axis_mut(Axis(axis), k);
        let a = data.index_axis(Axis(axis), lo);
        if lo == hi {
            dst.assign(&a);
            if scale_values && factor != 1.0 {
                dst.mapv_inplace(|v| v * factor);
            }
        } else {
            let b = data.index_axis(Axis(axis), hi);
            let zip = ndarray::Zip::from(&mut dst).and(&a).and(&b);
            if log_values {
                // exp(ln p + (ln q - ln p) t); undefined unless both are positive.
                zip.for_each(|d, p, q| {
                    *d = if *p > 0.0 && *q > 0.0 {
                        (p.ln() + (q / p).ln() * t).exp()
                    } else {
                        f64::NAN
                    };
                });
            } else {
                zip.for_each(|d, p, q| *d = p + (q - p) * t);
            }
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
    opts: &InterpOptions,
) -> Result<(Weights, Variable)> {
    let method = opts.method;
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
    let keys = index_keys(coord, dim)?;
    if method.log_coord() {
        if matches!(keys, Keys::Ns(..)) {
            return Err(invalid("log interpolation along times is not defined"));
        }
        linear_unit(coord.unit(), "a log coordinate")?;
    }
    let power_law = [opts.below, opts.above]
        .iter()
        .any(|e| matches!(e, Extrapolate::PowerLaw(_)));
    if power_law {
        if matches!(keys, Keys::Ns(..)) {
            return Err(invalid(
                "power-law extrapolation along times is not defined",
            ));
        }
        linear_unit(coord.unit(), "power-law extrapolation of the coordinate")?;
    }
    let (x, origin) = positions(keys, None)?;
    if method.log_coord() && x.iter().any(|v| *v <= 0.0) {
        return Err(invalid(format!(
            "log interpolation along {dim:?} needs positive coordinates"
        )));
    }
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
    Ok((
        Weights {
            at: weights(&x, &t, opts),
            log_values: method.log_values(),
            power_law,
        },
        new_coord,
    ))
}

/// Log interpolation is defined for ratio-scale units only (not °C, dates or dB).
fn linear_unit(u: Option<&annoray_units::Unit>, what: &str) -> Result<()> {
    match u.map(annoray_units::Unit::kind) {
        None | Some(annoray_units::UnitKind::Linear) => Ok(()),
        Some(_) => Err(invalid(format!(
            "{what} needs a ratio-scale unit, not {}",
            u.map(ToString::to_string).unwrap_or_default()
        ))),
    }
}

/// `v` interpolated along `dim` with `w`, keeping its metadata (values in log space for a
/// data variable of a log-value method; coordinates always linear in value).
fn interp_var<S: Repr>(
    v: &VariableBase<S>,
    dim: &Name,
    w: &Weights,
    data_var: bool,
) -> Result<Variable> {
    let axis = v.axis(dim)?;
    let log_values = data_var && w.log_values;
    if log_values {
        linear_unit(v.unit(), "log interpolation of values")?;
    }
    if data_var && w.power_law {
        linear_unit(v.unit(), "power-law extrapolation of values")?;
    }
    let values = apply(v, axis, w, log_values, data_var)?;
    let mut out = Variable::from_typed(v.dims().iter().cloned(), values)?;
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
            out.insert(k.clone(), interp_var(c, dim, w, false)?);
        }
    }
    Ok(out)
}

impl<S: Repr> DataArrayBase<S> {
    /// Interpolate along `dim` at the coordinate values `at` (a 1-D variable of numbers or
    /// dates; in another unit of the same kind, converted first). `opts` is an
    /// [`InterpMethod`] or [`InterpOptions`] (with extrapolation).
    ///
    /// # Errors
    /// [`CoreError::NoIndex`] if `dim` has no coordinate; [`CoreError::InvalidReduction`] for
    /// string coordinates, mismatched target types or non-1-D targets; unit errors;
    /// [`CoreError::UnsupportedDType`] for non-numeric data.
    pub fn interp<R: Repr>(
        &self,
        dim: &str,
        at: &VariableBase<R>,
        opts: impl Into<InterpOptions>,
    ) -> Result<DataArray> {
        let dim = Name::from(dim);
        let coord = self
            .coords()
            .get(&dim)
            .ok_or_else(|| CoreError::NoIndex(dim.clone()))?;
        let (w, new_coord) = plan(coord, &dim, at, &opts.into())?;
        let var = interp_var(self.variable(), &dim, &w, true)?;
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
        opts: impl Into<InterpOptions>,
    ) -> Result<DataArray> {
        let opts = opts.into();
        let mut out = self.to_owned();
        for (k, c) in other.coords() {
            let is_index = c.dims().len() == 1 && c.dims().first() == Some(k);
            if is_index && out.coords().contains_key(k) && out.dims().contains(k) {
                out = out.interp(k, c, opts)?;
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
        opts: impl Into<InterpOptions>,
    ) -> Result<Dataset> {
        let dim = Name::from(dim);
        let coord = self
            .coords()
            .get(&dim)
            .ok_or_else(|| CoreError::NoIndex(dim.clone()))?;
        let (w, new_coord) = plan(coord, &dim, at, &opts.into())?;
        let mut out = Dataset::new();
        for (k, c) in coords_after(self.coords().iter(), &dim, &w, &new_coord)? {
            out.add_coord(k, c)?;
        }
        for (k, v) in self.data_vars() {
            if !v.dims().contains(&dim) {
                out.add_var(k.clone(), v.to_owned())?;
            } else if numeric(v.dtype()) {
                out.add_var(k.clone(), interp_var(v, &dim, &w, true)?)?;
            }
        }
        out.attrs_mut()
            .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        Ok(out)
    }
}
