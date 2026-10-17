//! Covariance and Pearson correlation of two data arrays (xarray's `cov` and `corr`).
//!
//! As in xarray: the arrays are aligned (inner join) and broadcast; a position counts only
//! where both are present; the means are taken over the valid positions; `cov` divides by
//! the number of valid positions less `ddof`. Units follow the arithmetic: `cov` of metres
//! and seconds is in m·s; `corr` has no unit. Lazy data is read.

use crate::{BinOp, DataArray, DataArrayBase, Join, Name, ReduceOptions, Reduction, Repr, Result};
use annoray_units::UnitPolicy;

fn without_name(a: DataArray) -> DataArray {
    let (_, v, coords) = a.into_parts();
    let mut out = DataArray::new(v);
    for (k, c) in coords {
        // Coordinates of the result came from valid arrays, so they fit.
        if let Ok(o) = out.clone().with_coord(k, c) {
            out = o;
        }
    }
    out
}

fn reduce<S: Repr>(
    a: &DataArrayBase<S>,
    op: Reduction,
    dims: Option<&[Name]>,
    opts: &ReduceOptions,
) -> Result<DataArray> {
    match dims {
        None => a.reduce(op, .., opts),
        Some(d) => a.reduce(op, d, opts),
    }
}

fn demeaned<S: Repr>(a: &DataArrayBase<S>, dims: Option<&[Name]>) -> Result<DataArray> {
    let mean = reduce(a, Reduction::Mean, dims, &ReduceOptions::default())?;
    a.binary(BinOp::Sub, &mean, UnitPolicy::Strict)
}

/// Whether `a` may hold a missing value (NaN; never for integers). Lazy data is not read
/// for this: it may (the masked computation streams).
fn has_nan<S: Repr>(a: &DataArrayBase<S>) -> Result<bool> {
    if a.variable().is_lazy() {
        return Ok(true);
    }
    Ok(match a.variable().typed()? {
        crate::TypedArray::F32(x) => x.iter().any(|v| v.is_nan()),
        crate::TypedArray::F64(x) => x.iter().any(|v| v.is_nan()),
        _ => false,
    })
}

/// The covariance of aligned `a` and `b`, with the standard deviations corr needs.
fn moments<S: Repr, R: Repr>(
    a: &DataArrayBase<S>,
    b: &DataArrayBase<R>,
    dims: Option<&[Name]>,
    ddof: usize,
    with_std: bool,
) -> Result<(DataArray, Option<(DataArray, DataArray)>)> {
    let p = crate::align::plan(
        &[
            crate::align::Obj::of_array(a),
            crate::align::Obj::of_array(b),
        ],
        Join::Inner,
    )?;
    if !p.is_noop() {
        let (a, b) = (
            crate::align::apply_array(a, &p, 0)?,
            crate::align::apply_array(b, &p, 1)?,
        );
        return moments(&a, &b, dims, ddof, with_std);
    }
    if has_nan(a)? || has_nan(b)? {
        // Only the positions both have count: the others NaN in both (in the data's float
        // type, as xarray's `where` keeps float32).
        let valid = (&a.notnull()? & &b.notnull()?)?;
        let nan = |d: crate::DType| -> Result<DataArray> {
            if d == crate::DType::F32 {
                DataArray::scalar(f32::NAN)
            } else {
                DataArray::scalar(f64::NAN)
            }
        };
        let a = crate::where_(&valid, a, &nan(a.dtype())?)?;
        let b = crate::where_(&valid, b, &nan(b.dtype())?)?;
        let count = reduce(&valid, Reduction::Sum, dims, &ReduceOptions::default())?;
        return finish(&a, &b, dims, ddof, with_std, &count);
    }
    // No gaps: every position counts.
    let mut sizes: indexmap::IndexMap<Name, usize> =
        a.sizes().map(|(d, n)| (d.clone(), n)).collect();
    for (d, n) in b.sizes() {
        sizes.entry(d.clone()).or_insert(n);
    }
    let reduced: usize = sizes
        .iter()
        .filter(|(d, _)| dims.is_none_or(|ds| ds.contains(*d)))
        .map(|(_, n)| *n)
        .product();
    #[allow(clippy::cast_precision_loss)] // element counts
    let count = DataArray::scalar(reduced as f64)?;
    finish(a, b, dims, ddof, with_std, &count)
}

fn finish<S: Repr, R: Repr>(
    a: &DataArrayBase<S>,
    b: &DataArrayBase<R>,
    dims: Option<&[Name]>,
    ddof: usize,
    with_std: bool,
    count: &DataArray,
) -> Result<(DataArray, Option<(DataArray, DataArray)>)> {
    let product = demeaned(a, dims)?.binary(BinOp::Mul, &demeaned(b, dims)?, UnitPolicy::Strict)?;
    let opts = ReduceOptions::default().min_count(1);
    let sum = reduce(&product, Reduction::Sum, dims, &opts)?;
    #[allow(clippy::cast_precision_loss)] // ddof is small
    let n = count.binary_scalar(BinOp::Sub, ddof as f64)?;
    let c = sum.binary(BinOp::Div, &n, UnitPolicy::Strict)?;
    let stds = if with_std {
        let o = ReduceOptions::default();
        Some((
            reduce(a, Reduction::Std, dims, &o)?,
            reduce(b, Reduction::Std, dims, &o)?,
        ))
    } else {
        None
    };
    Ok((c, stds))
}

/// The covariance of `a` and `b` over `dims` (all shared dimensions when `None`), divided by
/// the count of valid pairs less `ddof` (xarray's default `ddof` is 1).
///
/// # Errors
/// Alignment errors; unit errors (none arise for numbers); errors for non-numeric data.
pub fn cov<S: Repr, R: Repr>(
    a: &DataArrayBase<S>,
    b: &DataArrayBase<R>,
    dims: Option<&[Name]>,
    ddof: usize,
) -> Result<DataArray> {
    let (c, _) = moments(a, b, dims, ddof, false)?;
    Ok(without_name(c))
}

/// The Pearson correlation of `a` and `b` over `dims` (all when `None`); unitless.
///
/// # Errors
/// As for [`cov`].
pub fn corr<S: Repr, R: Repr>(
    a: &DataArrayBase<S>,
    b: &DataArrayBase<R>,
    dims: Option<&[Name]>,
) -> Result<DataArray> {
    let (c, stds) = moments(a, b, dims, 0, true)?;
    let Some((sa, sb)) = stds else {
        return Err(crate::CoreError::InvalidReduction("corr: no spread".into()));
    };
    let mut r = c.binary(
        BinOp::Div,
        &sa.binary(BinOp::Mul, &sb, UnitPolicy::Strict)?,
        UnitPolicy::AutoRescale,
    )?;
    // m·s / (m·s): a pure number.
    if r.unit().is_some_and(annoray_units::Unit::is_dimensionless) {
        r = r.to_unit(&annoray_units::Unit::one())?;
        r.set_unit(None);
    }
    Ok(without_name(r))
}
