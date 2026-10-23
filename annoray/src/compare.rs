//! Comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`) and logical operations (`and`, `or`, `xor`,
//! `not`), giving boolean arrays.
//!
//! - Operands align and broadcast as in arithmetic (inner join on labels, by name).
//! - Numbers in different units of a kind are converted (`m` against `km`, °C against K as
//!   temperatures), by dividing by the scale (into the larger unit) when it is a whole number:
//!   `16100 m == 16.1 km` then holds exactly as the decimals read. Data with a unit compares
//!   with unitless data only when the unitless side is the scalar 0 and the unit is
//!   ratio-scale or logarithmic: zero is the same in every such unit (0 W m⁻² in any flux
//!   unit), but not in °C or `days since …`, where a bare number has no meaning (pint's rule).
//! - Text, dates, CF dates (same calendar) and durations compare with their own kind; booleans
//!   with booleans; integers exactly.
//! - Complex numbers compare for equality (`==`, `!=`) with complex and real numbers; they
//!   have no order, so `<`, `<=`, `>`, `>=` are errors.
//! - NaN and NaT compare false, except `!=`, which is true (numpy). A complex number is NaN
//!   when either part is.
//! - The result has no unit; attributes and coordinates as in arithmetic. Comparing lazy
//!   data with a scalar stays lazy; two lazy arrays are read.
//! - Logical operations take booleans without units.

use annoray_units::{Converter, Unit, UnitKind, UnitPolicy};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

use crate::align::{Join, Obj, apply_array, plan};
use crate::ops::{broadcast_to_dims, out_dims, to_c128, to_f64, variables_equal};
use crate::{
    CoreError, DType, DataArray, DataArrayBase, NaT, Name, Owned, Repr, Result, TypedArray,
    Variable, View,
};

/// A comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CmpOp {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `==`
    Eq,
    /// `!=`
    Ne,
}

impl CmpOp {
    /// The comparison with operands swapped (`a < b` is `b > a`).
    #[must_use]
    pub const fn swapped(self) -> Self {
        match self {
            Self::Lt => Self::Gt,
            Self::Le => Self::Ge,
            Self::Gt => Self::Lt,
            Self::Ge => Self::Le,
            other => other,
        }
    }

    fn holds<T: PartialOrd>(self, a: &T, b: &T) -> bool {
        match self {
            Self::Lt => a < b,
            Self::Le => a <= b,
            Self::Gt => a > b,
            Self::Ge => a >= b,
            Self::Eq => a == b,
            Self::Ne => a != b,
        }
    }
}

/// A logical operation on booleans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LogicOp {
    /// `&`
    And,
    /// `|`
    Or,
    /// `^`
    Xor,
}

fn bad_shape(e: ndarray::ShapeError) -> CoreError {
    CoreError::InvalidReduction(e.to_string())
}

/// `x op y` elementwise over the broadcast shape; `missing` values compare false (true for
/// `!=`).
#[allow(clippy::too_many_arguments)]
fn zip_cmp<T: PartialOrd + Sync>(
    x: ArrayViewD<'_, T>,
    xd: &[Name],
    y: ArrayViewD<'_, T>,
    yd: &[Name],
    od: &[Name],
    shape: &[usize],
    op: CmpOp,
    missing: impl Fn(&T) -> bool + Sync + Send,
) -> Result<ArrayD<bool>> {
    let x = broadcast_to_dims(x, xd, od)?;
    let y = broadcast_to_dims(y, yd, od)?;
    let no = || CoreError::InvalidReduction("operands do not broadcast".into());
    let (xb, yb) = (
        x.broadcast(IxDyn(shape)).ok_or_else(no)?,
        y.broadcast(IxDyn(shape)).ok_or_else(no)?,
    );
    let f = |p: &T, q: &T| {
        if missing(p) || missing(q) {
            op == CmpOp::Ne
        } else {
            op.holds(p, q)
        }
    };
    let zip = ndarray::Zip::from(&xb).and(&yb);
    #[cfg(feature = "rayon")]
    if xb.len() > PAR_ELEMENTS {
        return Ok(zip.par_map_collect(f));
    }
    Ok(zip.map_collect(f))
}

/// Elements above which comparisons are split across threads.
#[cfg(feature = "rayon")]
const PAR_ELEMENTS: usize = 1 << 16;

/// `a op s` for numeric `a` against the number `s` (both in `f64`, `a` converted by `conv`
/// element by element, nothing materialized); NaN compares false (true for `!=`). `None`
/// for non-numeric `a`.
fn cmp_scalar(
    a: &TypedArray<View<'_>>,
    conv: Option<&Converter>,
    s: f64,
    op: CmpOp,
) -> Option<ArrayD<bool>> {
    #[allow(clippy::cast_precision_loss)] // i64/u64 compared as f64, as to_f64 does
    fn go<T: Copy + Send + Sync>(
        x: &ArrayViewD<'_, T>,
        to: impl Fn(T) -> f64 + Sync + Send,
        conv: Option<&Converter>,
        s: f64,
        op: CmpOp,
    ) -> ArrayD<bool> {
        let s_nan = s.is_nan();
        let f = |v: &T| {
            let mut p = to(*v);
            if let Some(c) = conv {
                p = c.apply(p);
            }
            if p.is_nan() || s_nan {
                op == CmpOp::Ne
            } else {
                op.holds(&p, &s)
            }
        };
        let zip = ndarray::Zip::from(x);
        #[cfg(feature = "rayon")]
        if x.len() > PAR_ELEMENTS {
            return zip.par_map_collect(f);
        }
        zip.map_collect(f)
    }
    #[allow(clippy::cast_precision_loss)]
    Some(match a {
        TypedArray::F32(x) => go(x, f64::from, conv, s, op),
        TypedArray::F64(x) => go(x, |v| v, conv, s, op),
        TypedArray::I8(x) => go(x, f64::from, conv, s, op),
        TypedArray::I16(x) => go(x, f64::from, conv, s, op),
        TypedArray::I32(x) => go(x, f64::from, conv, s, op),
        TypedArray::I64(x) => go(x, |v| v as f64, conv, s, op),
        TypedArray::U8(x) => go(x, f64::from, conv, s, op),
        TypedArray::U16(x) => go(x, f64::from, conv, s, op),
        TypedArray::U32(x) => go(x, f64::from, conv, s, op),
        TypedArray::U64(x) => go(x, |v| v as f64, conv, s, op),
        _ => return None,
    })
}

fn is_int(d: DType) -> bool {
    matches!(
        d,
        DType::I8
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::U64
    )
}

fn ints(t: &TypedArray<View<'_>>) -> Option<ArrayD<i128>> {
    Some(match t {
        TypedArray::I8(a) => a.mapv(i128::from),
        TypedArray::I16(a) => a.mapv(i128::from),
        TypedArray::I32(a) => a.mapv(i128::from),
        TypedArray::I64(a) => a.mapv(i128::from),
        TypedArray::U8(a) => a.mapv(i128::from),
        TypedArray::U16(a) => a.mapv(i128::from),
        TypedArray::U32(a) => a.mapv(i128::from),
        TypedArray::U64(a) => a.mapv(i128::from),
        _ => return None,
    })
}

/// CF dates as nanoseconds since 1970-01-01 in their (common) calendar; NaT as `None`.
fn cf_keys(
    a: &ArrayViewD<'_, annoray_units::CfDate>,
    cal: annoray_units::Calendar,
) -> Result<ArrayD<Option<i128>>> {
    let epoch = annoray_units::CfDate::from_ymd(cal, 1970, 1, 1)?;
    let mut out = Vec::with_capacity(a.len());
    for d in a {
        if d.is_nat() {
            out.push(None);
        } else if d.calendar() != cal {
            return Err(CoreError::InvalidReduction(
                "comparing dates in different calendars".into(),
            ));
        } else {
            out.push(Some(d.nanos_since(&epoch)?));
        }
    }
    ArrayD::from_shape_vec(IxDyn(a.shape()), out).map_err(bad_shape)
}

fn first_calendar(a: &ArrayViewD<'_, annoray_units::CfDate>) -> Option<annoray_units::Calendar> {
    a.iter().find(|d| !d.is_nat()).map(|d| d.calendar())
}

/// `a op b` on raw data (units already planned into the converters).
#[allow(clippy::too_many_arguments)]
fn compute(
    op: CmpOp,
    (a, ad, ac): (&TypedArray<View<'_>>, &[Name], Option<&Converter>),
    (b, bd, bc): (&TypedArray<View<'_>>, &[Name], Option<&Converter>),
    od: &[Name],
    shape: &[usize],
) -> Result<ArrayD<bool>> {
    let (adt, bdt) = (a.dtype(), b.dtype());
    let numeric = |d: DType| is_int(d) || matches!(d, DType::F32 | DType::F64);
    let complex = |d: DType| matches!(d, DType::C64 | DType::C128);
    if complex(adt) || complex(bdt) {
        let dtype = if complex(adt) { adt } else { bdt };
        let comparable = |d: DType| numeric(d) || complex(d);
        if !matches!(op, CmpOp::Eq | CmpOp::Ne) || !comparable(adt) || !comparable(bdt) {
            return Err(CoreError::UnsupportedDType {
                op: "an ordering comparison or one with a non-number",
                dtype,
            });
        }
        let (x, y) = (to_c128(a, ac)?, to_c128(b, bc)?);
        let (x, y) = (
            broadcast_to_dims(x.view(), ad, od)?,
            broadcast_to_dims(y.view(), bd, od)?,
        );
        let no = || CoreError::InvalidReduction("operands do not broadcast".into());
        let (xb, yb) = (
            x.broadcast(IxDyn(shape)).ok_or_else(no)?,
            y.broadcast(IxDyn(shape)).ok_or_else(no)?,
        );
        let nan = |z: &num_complex::Complex<f64>| z.re.is_nan() || z.im.is_nan();
        return Ok(ndarray::Zip::from(&xb).and(&yb).map_collect(|p, q| {
            if nan(p) || nan(q) {
                op == CmpOp::Ne
            } else {
                (p == q) == (op == CmpOp::Eq)
            }
        }));
    }
    if is_int(adt) && is_int(bdt) && ac.is_none() && bc.is_none() {
        let (Some(x), Some(y)) = (ints(a), ints(b)) else {
            return Err(CoreError::InvalidReduction("integers".into()));
        };
        return zip_cmp(x.view(), ad, y.view(), bd, od, shape, op, |_| false);
    }
    if numeric(adt) && numeric(bdt) {
        // Against a single number (the common case): converted element by element, in
        // parallel, without an f64 copy of the array.
        if bd.is_empty() && b.len() == 1 && ad == od {
            let s = to_f64(b, bc)?.iter().next().copied().unwrap_or(f64::NAN);
            if let Some(r) = cmp_scalar(a, ac, s, op) {
                return Ok(r);
            }
        }
        if ad.is_empty() && a.len() == 1 && bd == od {
            let s = to_f64(a, ac)?.iter().next().copied().unwrap_or(f64::NAN);
            if let Some(r) = cmp_scalar(b, bc, s, op.swapped()) {
                return Ok(r);
            }
        }
        let (x, y) = (to_f64(a, ac)?, to_f64(b, bc)?);
        return zip_cmp(x.view(), ad, y.view(), bd, od, shape, op, |v: &f64| {
            v.is_nan()
        });
    }
    match (a, b) {
        (TypedArray::Bool(x), TypedArray::Bool(y)) => {
            zip_cmp(x.view(), ad, y.view(), bd, od, shape, op, |_| false)
        }
        (TypedArray::Str(x), TypedArray::Str(y)) => {
            zip_cmp(x.view(), ad, y.view(), bd, od, shape, op, |_| false)
        }
        (TypedArray::DateTime(x), TypedArray::DateTime(y)) => {
            zip_cmp(x.view(), ad, y.view(), bd, od, shape, op, NaT::is_nat)
        }
        (TypedArray::TimeDelta(x), TypedArray::TimeDelta(y)) => {
            zip_cmp(x.view(), ad, y.view(), bd, od, shape, op, NaT::is_nat)
        }
        (TypedArray::CfDate(x), TypedArray::CfDate(y)) => {
            let cal = first_calendar(&x.view())
                .or_else(|| first_calendar(&y.view()))
                .unwrap_or(annoray_units::Calendar::Standard);
            let (kx, ky) = (cf_keys(&x.view(), cal)?, cf_keys(&y.view(), cal)?);
            zip_cmp(kx.view(), ad, ky.view(), bd, od, shape, op, Option::is_none)
        }
        _ => Err(CoreError::UnsupportedDType {
            op: "comparison (with an operand of another kind)",
            dtype: if numeric(adt) { bdt } else { adt },
        }),
    }
}

/// Whether `v` is the scalar zero (any numeric type).
fn is_scalar_zero<R: Repr>(v: &crate::VariableBase<R>) -> bool {
    v.ndim() == 0 && !v.is_lazy() && v.to_f64().is_ok_and(|a| a.iter().all(|x| *x == 0.0))
}

/// Conversions for comparing `lhs` with `rhs`: `(lhs, rhs)` converters.
fn plan_units<S: Repr, R: Repr>(
    lhs: &DataArrayBase<S>,
    rhs: &DataArrayBase<R>,
) -> Result<(Option<Converter>, Option<Converter>)> {
    let time = |d: DType| matches!(d, DType::DateTime | DType::CfDate | DType::TimeDelta);
    if time(lhs.variable().dtype()) || time(rhs.variable().dtype()) {
        if lhs.unit().is_some() || rhs.unit().is_some() {
            return Err(CoreError::UnsupportedDType {
                op: "comparison of unit-carrying times",
                dtype: lhs.variable().dtype(),
            });
        }
        return Ok((None, None));
    }
    let zero_ok = |u: &Unit| matches!(u.kind(), UnitKind::Linear | UnitKind::Log(_));
    match (lhs.unit(), rhs.unit()) {
        (None, None) => Ok((None, None)),
        (Some(u), None) if zero_ok(u) && is_scalar_zero(rhs.variable()) => Ok((None, None)),
        (None, Some(u)) if zero_ok(u) && is_scalar_zero(lhs.variable()) => Ok((None, None)),
        (l, r) => {
            let one = Unit::one();
            let (lu, ru) = (l.unwrap_or(&one), r.unwrap_or(&one));
            let p = lu.plan_cmp(ru, UnitPolicy::AutoRescale)?;
            // The result is true/false, so either side may be converted: divide rather than
            // multiply. A whole number in the smaller unit then lands exactly on the decimal in
            // the larger one (16100 m / 1000 is the float 16.1; 16.1 km · 1000 is not 16100).
            let integer_factor = p
                .rhs
                .and_then(|c| c.exact_factor())
                .is_some_and(|f| f.denom() == 1 && f.numer() > 1);
            if integer_factor && p.lhs.is_none() {
                return Ok((Some(lu.converter_to(ru)?), None));
            }
            Ok((p.lhs, p.rhs))
        }
    }
}

/// Coordinates of shared dimensions must agree (after alignment).
fn check_coords<S: Repr, R: Repr>(a: &DataArrayBase<S>, b: &DataArrayBase<R>) -> Result<()> {
    for d in a.dims().iter().filter(|d| b.dims().contains(d)) {
        if let (Some(x), Some(y)) = (a.coords().get(d), b.coords().get(d))
            && !variables_equal(x, y)
        {
            return Err(CoreError::NotAligned(d.clone()));
        }
    }
    Ok(())
}

fn bool_var(dims: Vec<Name>, data: ArrayD<bool>) -> Result<Variable> {
    Variable::from_typed(dims, TypedArray::Bool(data))
}

impl<S: Repr> DataArrayBase<S> {
    /// `self op rhs`, elementwise, as booleans; see the module notes.
    ///
    /// # Errors
    /// Unit errors (incompatible units, a bare nonzero number against data with a unit);
    /// [`CoreError::UnsupportedDType`] for operands of different kinds; alignment errors.
    pub fn compare<R: Repr>(&self, op: CmpOp, rhs: &DataArrayBase<R>) -> Result<DataArray> {
        let p = plan(&[Obj::of_array(self), Obj::of_array(rhs)], Join::Inner)?;
        if !p.is_noop() {
            let a = apply_array(self, &p, 0)?;
            let b = apply_array(rhs, &p, 1)?;
            return a.compare(op, &b);
        }
        check_coords(self, rhs)?;
        let (lc, rc) = plan_units(self, rhs)?;
        let sizes = |d: &dyn Fn() -> Vec<(Name, usize)>| d();
        let (od, shape) = out_dims(
            &sizes(&|| self.sizes().map(|(d, n)| (d.clone(), n)).collect()),
            &sizes(&|| rhs.sizes().map(|(d, n)| (d.clone(), n)).collect()),
        )?;
        // Lazy data against a scalar: compared block by block when read.
        if self.variable().is_lazy() && rhs.variable().ndim() == 0 && !rhs.variable().is_lazy() {
            let scalar = rhs.variable().typed()?.to_owned();
            let dims = self.dims().to_vec();
            let mut var = self
                .variable()
                .lazy_map("comparison", move |t: TypedArray<Owned>| {
                    let r = compute(
                        op,
                        (&t.view(), &dims, lc.as_ref()),
                        (&scalar.view(), &[], rc.as_ref()),
                        &dims,
                        t.shape(),
                    )?;
                    Ok(TypedArray::Bool(r))
                })?;
            var.set_unit(None);
            return Ok(self.assemble(
                var,
                None,
                rhs.name(),
                rhs.attrs(),
                rhs.coords().iter().map(|(k, v)| (k, v.to_owned())),
            ));
        }
        if rhs.variable().is_lazy() && self.variable().ndim() == 0 && !self.variable().is_lazy() {
            return rhs.compare(op.swapped(), self).map(|mut r| {
                r.name = self.name().cloned();
                r
            });
        }
        let data = compute(
            op,
            (&self.variable().typed()?, self.dims(), lc.as_ref()),
            (&rhs.variable().typed()?, rhs.dims(), rc.as_ref()),
            &od,
            &shape,
        )?;
        let var = bool_var(od, data)?;
        Ok(self.assemble(
            var,
            None,
            rhs.name(),
            rhs.attrs(),
            rhs.coords().iter().map(|(k, v)| (k, v.to_owned())),
        ))
    }

    /// `self op rhs` on booleans, elementwise (aligned and broadcast as in arithmetic).
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for non-boolean operands; unit errors for operands with
    /// units; alignment errors.
    pub fn logical<R: Repr>(&self, op: LogicOp, rhs: &DataArrayBase<R>) -> Result<DataArray> {
        let p = plan(&[Obj::of_array(self), Obj::of_array(rhs)], Join::Inner)?;
        if !p.is_noop() {
            let a = apply_array(self, &p, 0)?;
            let b = apply_array(rhs, &p, 1)?;
            return a.logical(op, &b);
        }
        check_coords(self, rhs)?;
        for v in [self.dtype(), rhs.dtype()] {
            if v != DType::Bool {
                return Err(CoreError::UnsupportedDType {
                    op: "a logical operation (booleans only)",
                    dtype: v,
                });
            }
        }
        if let Some(u) = self.unit().or(rhs.unit()) {
            return Err(annoray_units::UnitError::UnsupportedOp {
                op: "a logical operation",
                unit: u.to_string(),
            }
            .into());
        }
        let sizes = |d: &dyn Fn() -> Vec<(Name, usize)>| d();
        let (od, shape) = out_dims(
            &sizes(&|| self.sizes().map(|(d, n)| (d.clone(), n)).collect()),
            &sizes(&|| rhs.sizes().map(|(d, n)| (d.clone(), n)).collect()),
        )?;
        let (TypedArray::Bool(x), TypedArray::Bool(y)) =
            (self.variable().typed()?, rhs.variable().typed()?)
        else {
            return Err(CoreError::InvalidReduction("booleans".into()));
        };
        let x = broadcast_to_dims(x.view(), self.dims(), &od)?;
        let y = broadcast_to_dims(y.view(), rhs.dims(), &od)?;
        let no = || CoreError::InvalidReduction("operands do not broadcast".into());
        let (xb, yb) = (
            x.broadcast(IxDyn(&shape)).ok_or_else(no)?,
            y.broadcast(IxDyn(&shape)).ok_or_else(no)?,
        );
        let v = xb
            .iter()
            .zip(yb.iter())
            .map(|(p, q)| match op {
                LogicOp::And => *p && *q,
                LogicOp::Or => *p || *q,
                LogicOp::Xor => p != q,
            })
            .collect();
        let var = bool_var(
            od,
            ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(bad_shape)?,
        )?;
        Ok(self.assemble(
            var,
            None,
            rhs.name(),
            rhs.attrs(),
            rhs.coords().iter().map(|(k, v)| (k, v.to_owned())),
        ))
    }

    /// Logical negation of booleans (`~`); lazy data stays lazy.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for non-boolean data.
    pub fn logical_not(&self) -> Result<DataArray> {
        if self.dtype() != DType::Bool {
            return Err(CoreError::UnsupportedDType {
                op: "a logical operation (booleans only)",
                dtype: self.dtype(),
            });
        }
        let not = |t: TypedArray<Owned>| match t {
            TypedArray::Bool(a) => Ok(TypedArray::Bool(a.mapv(|b| !b))),
            other => Err(CoreError::UnsupportedDType {
                op: "logical not",
                dtype: other.dtype(),
            }),
        };
        let mut var = if self.variable().is_lazy() {
            self.variable().lazy_map("logical not", not)?
        } else {
            Variable::from_typed(
                self.dims().iter().cloned(),
                not(self.variable().typed()?.to_owned())?,
            )?
        };
        var.set_unit(None);
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            var,
            self.coords()
                .iter()
                .map(|(k, c)| (k.clone(), c.to_owned()))
                .collect(),
        ))
    }
}

// ----- comparison methods and logical operators -----

/// The right operand of a comparison method ([`DataArrayBase::gt`] and friends): a data
/// array, a plain number (unitless: only 0 compares with data that has a ratio-scale unit) or
/// a quantity (converted into the data's unit).
pub trait Comparand {
    /// `lhs op self`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::compare`].
    fn compared_with<S: Repr>(self, lhs: &DataArrayBase<S>, op: CmpOp) -> Result<DataArray>;
}

impl<R: Repr> Comparand for &DataArrayBase<R> {
    fn compared_with<S: Repr>(self, lhs: &DataArrayBase<S>, op: CmpOp) -> Result<DataArray> {
        lhs.compare(op, self)
    }
}

impl Comparand for f64 {
    fn compared_with<S: Repr>(self, lhs: &DataArrayBase<S>, op: CmpOp) -> Result<DataArray> {
        lhs.compare(op, &crate::ops::scalar_array(self)?)
    }
}

impl<K: annoray_units::Kind> Comparand for &annoray_units::Quantity<K> {
    fn compared_with<S: Repr>(self, lhs: &DataArrayBase<S>, op: CmpOp) -> Result<DataArray> {
        let q = crate::ops::scalar_array(self.value())?.with_unit(self.unit().clone());
        lhs.compare(op, &q)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// `self < rhs`, elementwise ([`compare`](Self::compare) with [`CmpOp::Lt`]). Rust's `<`
    /// cannot return an array (`PartialOrd` gives a `bool`), hence the method.
    ///
    /// # Errors
    /// As for [`compare`](Self::compare).
    pub fn lt(&self, rhs: impl Comparand) -> Result<DataArray> {
        rhs.compared_with(self, CmpOp::Lt)
    }

    /// `self <= rhs`, elementwise.
    ///
    /// # Errors
    /// As for [`compare`](Self::compare).
    pub fn le(&self, rhs: impl Comparand) -> Result<DataArray> {
        rhs.compared_with(self, CmpOp::Le)
    }

    /// `self > rhs`, elementwise.
    ///
    /// # Errors
    /// As for [`compare`](Self::compare).
    pub fn gt(&self, rhs: impl Comparand) -> Result<DataArray> {
        rhs.compared_with(self, CmpOp::Gt)
    }

    /// `self >= rhs`, elementwise.
    ///
    /// # Errors
    /// As for [`compare`](Self::compare).
    pub fn ge(&self, rhs: impl Comparand) -> Result<DataArray> {
        rhs.compared_with(self, CmpOp::Ge)
    }

    /// `self == rhs`, elementwise (named apart from `PartialEq::eq`, which would compare whole
    /// arrays into one `bool`).
    ///
    /// # Errors
    /// As for [`compare`](Self::compare).
    pub fn eq_elem(&self, rhs: impl Comparand) -> Result<DataArray> {
        rhs.compared_with(self, CmpOp::Eq)
    }

    /// `self != rhs`, elementwise (NaN != NaN is true).
    ///
    /// # Errors
    /// As for [`compare`](Self::compare).
    pub fn ne_elem(&self, rhs: impl Comparand) -> Result<DataArray> {
        rhs.compared_with(self, CmpOp::Ne)
    }
}

/// `&`, `|`, `^` on boolean arrays, as for arithmetic: between references, and with a
/// `Result` on either side (Rust's orphan rule rules out `Result op Result`, so borrow one:
/// `(&a.gt(lo)? & a.lt(hi))?`).
macro_rules! impl_logic {
    ($($tr:ident $m:ident $op:ident),*) => {$(
        impl<S: Repr, R: Repr> std::ops::$tr<&DataArrayBase<R>> for &DataArrayBase<S> {
            type Output = Result<DataArray>;
            fn $m(self, rhs: &DataArrayBase<R>) -> Result<DataArray> {
                self.logical(LogicOp::$op, rhs)
            }
        }
        impl<R: Repr> std::ops::$tr<&DataArrayBase<R>> for Result<DataArray> {
            type Output = Result<DataArray>;
            fn $m(self, rhs: &DataArrayBase<R>) -> Result<DataArray> {
                self?.logical(LogicOp::$op, rhs)
            }
        }
        impl<S: Repr> std::ops::$tr<Result<DataArray>> for &DataArrayBase<S> {
            type Output = Result<DataArray>;
            fn $m(self, rhs: Result<DataArray>) -> Result<DataArray> {
                self.logical(LogicOp::$op, &rhs?)
            }
        }
    )*};
}

impl_logic!(BitAnd bitand And, BitOr bitor Or, BitXor bitxor Xor);

/// `!mask`: logical not of a boolean array.
impl<S: Repr> std::ops::Not for &DataArrayBase<S> {
    type Output = Result<DataArray>;
    fn not(self) -> Result<DataArray> {
        self.logical_not()
    }
}
