//! Covariance and Pearson correlation of two data arrays (xarray's `cov` and `corr`).
//!
//! As in xarray: the arrays are aligned (inner join) and broadcast; a position counts only
//! where both are present; the means are taken over the valid positions; `cov` divides by
//! the number of valid positions less `ddof`. Units follow the arithmetic: `cov` of metres
//! and seconds is in m·s; `corr` has no unit. Lazy data is read.

use crate::{BinOp, DataArray, DataArrayBase, Join, Name, ReduceOptions, Reduction, Repr, Result};
use annoray_units::UnitPolicy;

fn reduce(
    a: &DataArray,
    op: Reduction,
    dims: Option<&[Name]>,
    opts: &ReduceOptions,
) -> Result<DataArray> {
    match dims {
        None => a.reduce(op, .., opts),
        Some(d) => a.reduce(op, d, opts),
    }
}

/// The aligned arrays, each NaN where either is missing, and the count of valid positions.
fn masked<S: Repr, R: Repr>(
    a: &DataArrayBase<S>,
    b: &DataArrayBase<R>,
    dims: Option<&[Name]>,
) -> Result<(DataArray, DataArray, DataArray)> {
    let (a, b) = (a.to_owned(), b.to_owned());
    let mut both = crate::align(&[&a, &b], Join::Inner)?.into_iter();
    let (Some(a), Some(b)) = (both.next(), both.next()) else {
        return Err(crate::CoreError::InvalidReduction("cov: alignment".into()));
    };
    let valid = (&a.notnull()? & &b.notnull()?)?;
    let nan = DataArray::scalar(f64::NAN)?;
    let a = crate::where_(&valid, &a, &nan)?;
    let b = crate::where_(&valid, &b, &nan)?;
    let count = reduce(&valid, Reduction::Sum, dims, &ReduceOptions::default())?;
    Ok((a, b, count))
}

fn demeaned(a: &DataArray, dims: Option<&[Name]>) -> Result<DataArray> {
    let mean = reduce(a, Reduction::Mean, dims, &ReduceOptions::default())?;
    a.binary(BinOp::Sub, &mean, UnitPolicy::Strict)
}

fn cov_impl<S: Repr, R: Repr>(
    a: &DataArrayBase<S>,
    b: &DataArrayBase<R>,
    dims: Option<&[Name]>,
    ddof: usize,
) -> Result<(DataArray, DataArray, DataArray)> {
    let (a, b, count) = masked(a, b, dims)?;
    let product =
        demeaned(&a, dims)?.binary(BinOp::Mul, &demeaned(&b, dims)?, UnitPolicy::Strict)?;
    let opts = ReduceOptions::default().min_count(1);
    let sum = reduce(&product, Reduction::Sum, dims, &opts)?;
    #[allow(clippy::cast_precision_loss)] // ddof is small
    let n = count.binary_scalar(BinOp::Sub, ddof as f64)?;
    Ok((sum.binary(BinOp::Div, &n, UnitPolicy::Strict)?, a, b))
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
    let (c, _, _) = cov_impl(a, b, dims, ddof)?;
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
    let (c, a, b) = cov_impl(a, b, dims, 0)?;
    let opts = ReduceOptions::default();
    let sa = reduce(&a, Reduction::Std, dims, &opts)?;
    let sb = reduce(&b, Reduction::Std, dims, &opts)?;
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
