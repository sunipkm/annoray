//! `where` with both branches given: `cond ? x : y`, elementwise (xarray's `xr.where` and
//! `DataArray.where(cond, other)`).
//!
//! - The operands are aligned exactly (their shared dimension coordinates must match, as in
//!   xarray) and broadcast against one another.
//! - Units: `y` is converted into `x`'s unit (any unit of the same kind). A unitless `y` is
//!   accepted against `x` with a unit only when it is the scalar NaN, or the scalar 0 for a
//!   ratio-scale or logarithmic unit (the comparison rule); likewise the other way round.
//! - Element type: that of `x` and `y` when they share one (any type: text, dates,
//!   booleans); `f64` for numbers of different types or after a unit conversion.
//! - The name is the one the named operands share, if any; attributes and coordinates are
//!   those of `x`, then `y`, then `cond`.
//! - Lazy operands are read.

use annoray_units::{Converter, Unit, UnitKind};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

use crate::align::{Join, Obj, apply_array, plan};
use crate::ops::{broadcast_to_dims, out_dims, to_f64};
use crate::{
    CoreError, DType, DataArray, DataArrayBase, Name, Owned, Repr, Result, TypedArray, Variable,
    VariableBase, View,
};

fn is_scalar<R: Repr>(v: &VariableBase<R>, f: impl Fn(f64) -> bool) -> bool {
    v.ndim() == 0 && !v.is_lazy() && v.to_f64().is_ok_and(|a| a.iter().all(|x| f(*x)))
}

/// Whether unitless `bare` can stand beside data in `unit`.
fn bare_ok<R: Repr>(unit: &Unit, bare: &VariableBase<R>) -> bool {
    let zero_ok = matches!(unit.kind(), UnitKind::Linear | UnitKind::Log(_));
    is_scalar(bare, f64::is_nan) || (zero_ok && is_scalar(bare, |x| x == 0.0))
}

/// The result's unit and the conversion of `y` into it.
fn plan_units<X: Repr, Y: Repr>(
    x: &DataArrayBase<X>,
    y: &DataArrayBase<Y>,
) -> Result<(Option<Unit>, Option<Converter>)> {
    let incompatible = |a: &Unit| {
        CoreError::Unit(annoray_units::UnitError::Incompatible {
            lhs: a.to_string(),
            rhs: Unit::one().to_string(),
        })
    };
    match (x.unit(), y.unit()) {
        (None, None) => Ok((None, None)),
        (Some(u), None) if bare_ok(u, y.variable()) => Ok((Some(u.clone()), None)),
        (None, Some(u)) if bare_ok(u, x.variable()) => Ok((Some(u.clone()), None)),
        (Some(u), None) | (None, Some(u)) => Err(incompatible(u)),
        (Some(a), Some(b)) if a == b => Ok((Some(a.clone()), None)),
        (Some(a), Some(b)) => Ok((Some(a.clone()), Some(b.converter_to(a)?))),
    }
}

fn is_number(d: DType) -> bool {
    !matches!(
        d,
        DType::Bool | DType::Str | DType::DateTime | DType::CfDate | DType::TimeDelta
    )
}

/// `c ? a : b`, each broadcast from its own dimensions to `od`/`shape`.
fn pick<T: Clone>(
    (c, cd): (ArrayViewD<'_, bool>, &[Name]),
    (a, ad): (ArrayViewD<'_, T>, &[Name]),
    (b, bd): (ArrayViewD<'_, T>, &[Name]),
    od: &[Name],
    shape: &[usize],
) -> Result<ArrayD<T>> {
    let spread = |what: &str| CoreError::InvalidReduction(format!("cannot broadcast {what}"));
    let sh = IxDyn(shape);
    let c = broadcast_to_dims(c, cd, od)?;
    let c = c
        .broadcast(sh.clone())
        .ok_or_else(|| spread("the condition"))?;
    let a = broadcast_to_dims(a, ad, od)?;
    let a = a.broadcast(sh.clone()).ok_or_else(|| spread("x"))?;
    let b = broadcast_to_dims(b, bd, od)?;
    let b = b.broadcast(sh).ok_or_else(|| spread("y"))?;
    Ok(ndarray::Zip::from(&c)
        .and(&a)
        .and(&b)
        .map_collect(|c, a, b| if *c { a.clone() } else { b.clone() }))
}

/// `cond ? x : y`; the result's dimensions come from `cond` first when `cond_first`, else
/// from `x` first.
fn choose<C: Repr, X: Repr, Y: Repr>(
    cond: &DataArrayBase<C>,
    x: &DataArrayBase<X>,
    y: &DataArrayBase<Y>,
    cond_first: bool,
) -> Result<DataArray> {
    let p = plan(
        &[Obj::of_array(cond), Obj::of_array(x), Obj::of_array(y)],
        Join::Exact,
    )?;
    let (cond, x, y) = (
        apply_array(cond, &p, 0)?,
        apply_array(x, &p, 1)?,
        apply_array(y, &p, 2)?,
    );
    if cond.dtype() != DType::Bool {
        return Err(CoreError::DTypeMismatch {
            expected: DType::Bool,
            found: cond.dtype(),
        });
    }
    let (unit, conv) = plan_units(&x, &y)?;
    let sizes =
        |a: &DataArray| -> Vec<(Name, usize)> { a.sizes().map(|(d, n)| (d.clone(), n)).collect() };
    let (first, second) = if cond_first { (&cond, &x) } else { (&x, &cond) };
    let (od, shape) = out_dims(&sizes(first), &sizes(second))?;
    let lead: Vec<(Name, usize)> = od.iter().cloned().zip(shape.iter().copied()).collect();
    let (od, shape) = out_dims(&lead, &sizes(&y))?;

    let ct = cond.variable().typed()?;
    let TypedArray::Bool(c) = &ct else {
        return Err(CoreError::InvalidReduction("where: condition".into()));
    };
    let c = (c.view(), cond.dims());
    let (xt, yt) = (x.variable().typed()?, y.variable().typed()?);
    let data = if xt.dtype() == yt.dtype() && conv.is_none() {
        same_type(c, (&xt, x.dims()), (&yt, y.dims()), &od, &shape)?
    } else if is_number(xt.dtype()) && is_number(yt.dtype()) {
        let (a, b) = (to_f64(&xt, None)?, to_f64(&yt, conv.as_ref())?);
        TypedArray::F64(pick(
            c,
            (a.view(), x.dims()),
            (b.view(), y.dims()),
            &od,
            &shape,
        )?)
    } else {
        return Err(CoreError::UnsupportedDType {
            op: "where (with branches of different kinds)",
            dtype: if is_number(xt.dtype()) {
                yt.dtype()
            } else {
                xt.dtype()
            },
        });
    };
    let mut var = Variable::from_typed(od, data)?;
    var.set_unit(unit);
    var.attrs_mut()
        .extend(x.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    let mut coords: indexmap::IndexMap<Name, Variable> = indexmap::IndexMap::new();
    for a in [&x, &y, &cond] {
        for (k, v) in a.coords() {
            coords.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    let mut names = [cond.name(), x.name(), y.name()].into_iter().flatten();
    let name = names.next().filter(|n| names.all(|m| m == *n)).cloned();
    Ok(DataArrayBase::from_parts(name, var, coords))
}

/// [`pick`] for branches of the same element type.
fn same_type(
    c: (ArrayViewD<'_, bool>, &[Name]),
    (a, ad): (&TypedArray<View<'_>>, &[Name]),
    (b, bd): (&TypedArray<View<'_>>, &[Name]),
    od: &[Name],
    shape: &[usize],
) -> Result<TypedArray<Owned>> {
    macro_rules! same {
        ($($v:ident),*) => {
            match (a, b) {
                $((TypedArray::$v(a), TypedArray::$v(b)) => {
                    TypedArray::$v(pick(c, (a.view(), ad), (b.view(), bd), od, shape)?)
                })*
                _ => return Err(CoreError::InvalidReduction("where: element types".into())),
            }
        };
    }
    Ok(same!(
        F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, Bool, Str, DateTime, CfDate, TimeDelta
    ))
}

/// `cond ? x : y`, elementwise (xarray's `xr.where`). The result's dimensions are those of
/// `cond`, then `x`, then `y`.
///
/// - The operands are aligned exactly (shared dimension coordinates must match) and
///   broadcast against one another.
/// - `y` is converted into `x`'s unit. A unitless operand stands beside one with a unit
///   only when it is the scalar NaN, or the scalar 0 for a ratio-scale or logarithmic unit.
/// - Branches of one element type keep it (text, dates, booleans included); numbers of
///   different types, or converted between units, give `f64`.
/// - The name is the one the named operands share; attributes are `x`'s; coordinates come
///   from `x`, then `y`, then `cond`. Lazy operands are read.
///
/// # Errors
/// [`CoreError::DTypeMismatch`] if `cond` is not boolean; unit errors; alignment errors
/// (coordinates of a shared dimension differ); [`CoreError::UnsupportedDType`] for branches
/// of different kinds (text against numbers).
pub fn where_<C: Repr, X: Repr, Y: Repr>(
    cond: &DataArrayBase<C>,
    x: &DataArrayBase<X>,
    y: &DataArrayBase<Y>,
) -> Result<DataArray> {
    choose(cond, x, y, true)
}

impl<S: Repr> DataArrayBase<S> {
    /// Values where `cond` is true, `other`'s elsewhere (xarray's `where(cond, other)` with an
    /// array or quantity); see [`where_`]. The result's dimensions are this array's, then
    /// `cond`'s, then `other`'s.
    ///
    /// # Errors
    /// As for [`where_`].
    pub fn where_else<C: Repr, O: Repr>(
        &self,
        cond: &DataArrayBase<C>,
        other: &DataArrayBase<O>,
    ) -> Result<DataArray> {
        choose(cond, self, other, false)
    }
}
