//! Unit-aware arithmetic on data arrays, broadcasting by dimension name.
//!
//! - **Broadcasting** is by name: the result has the left operand's dimensions followed by the
//!   right operand's extra ones. Shared dimensions must have equal lengths.
//! - **Coordinates** of shared dimensions must be equal (full alignment comes later); the
//!   result keeps the union of both operands' coordinates.
//! - **Units** follow `xr-units`: `+`/`-` need compatible units and, under the default strict
//!   policy, equal scales; `*`/`/` combine units. A missing unit counts as dimensionless.
//! - **Element types**: two `f32` operands stay `f32`; integers of the same type stay integers
//!   (checked, so overflow is an error) except for division; everything else is `f64`.
//!   Datetimes and durations support the usual differences and offsets.
//! - **Operators** return `Result<DataArray>` and accept a `Result` on either side of a data
//!   array, so `(&a + &b * &c)?` works with a single `?`. (Rust's orphan rules forbid
//!   `Result + Result` and `Result * f64`; use `?` first there.)

use std::ops::{Add, Div, Mul, Sub};

use chrono::{NaiveDateTime, TimeDelta};
use indexmap::IndexMap;
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use xr_units::{CfDate, Converter, Unit, UnitPolicy};

use crate::array::typed_dispatch;
use crate::dataarray::DataArrayBase;
use crate::{
    CoreError, DType, DataArray, Element, Name, Repr, Result, TypedArray, Variable, VariableBase,
    View,
};

/// An arithmetic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
}

impl BinOp {
    const fn name(self) -> &'static str {
        match self {
            Self::Add => "addition",
            Self::Sub => "subtraction",
            Self::Mul => "multiplication",
            Self::Div => "division",
        }
    }
}

/// Whether two variables have the same dimensions, unit, element type and values.
#[must_use]
pub fn variables_equal<S: Repr, R: Repr>(a: &VariableBase<S>, b: &VariableBase<R>) -> bool {
    fn same<T: Element, X: ndarray::Data<Elem = T>, R: Repr>(
        x: &ndarray::ArrayBase<X, IxDyn>,
        b: &TypedArray<R>,
    ) -> bool {
        T::peek(b).is_some_and(|y| x.view() == y.view())
    }
    a.dims() == b.dims()
        && a.unit() == b.unit()
        && typed_dispatch!(a.typed(), x => same(x, b.typed()))
}

// ----- broadcasting -----

/// Reorder `v`'s axes into `out_dims` order and insert length-1 axes for missing dimensions.
fn align<'a, T>(v: ArrayViewD<'a, T>, dims: &[Name], out_dims: &[Name]) -> ArrayViewD<'a, T> {
    let order: Vec<usize> = out_dims
        .iter()
        .filter_map(|d| dims.iter().position(|x| x == d))
        .collect();
    // `order` is a permutation of v's axes: every dimension of v appears in `out_dims` once.
    let mut v = v.permuted_axes(IxDyn(&order));
    for (i, d) in out_dims.iter().enumerate() {
        if !dims.contains(d) {
            // Inserting in increasing order keeps `i <= v.ndim()`.
            v = v.insert_axis(Axis(i));
        }
    }
    v
}

fn zip_map<A, B, C>(
    a: &ArrayViewD<'_, A>,
    b: &ArrayViewD<'_, B>,
    shape: &[usize],
    f: impl Fn(&A, &B) -> Option<C>,
    op: &'static str,
) -> Result<ArrayD<C>> {
    let bad = || CoreError::ArithmeticOverflow(op);
    let ab = a.broadcast(IxDyn(shape)).ok_or_else(bad)?;
    let bb = b.broadcast(IxDyn(shape)).ok_or_else(bad)?;
    let values = ab
        .iter()
        .zip(bb.iter())
        .map(|(x, y)| f(x, y))
        .collect::<Option<Vec<C>>>()
        .ok_or_else(bad)?;
    ArrayD::from_shape_vec(IxDyn(shape), values).map_err(|_| bad())
}

#[allow(clippy::cast_precision_loss)] // integer data is promoted to f64 deliberately
fn to_f64(a: &TypedArray<View<'_>>, conv: Option<&Converter>) -> Result<ArrayD<f64>> {
    let out = match a {
        TypedArray::F32(x) => x.mapv(f64::from),
        TypedArray::F64(x) => x.to_owned(),
        TypedArray::I8(x) => x.mapv(f64::from),
        TypedArray::I16(x) => x.mapv(f64::from),
        TypedArray::I32(x) => x.mapv(f64::from),
        TypedArray::I64(x) => x.mapv(|v| v as f64),
        TypedArray::U8(x) => x.mapv(f64::from),
        TypedArray::U16(x) => x.mapv(f64::from),
        TypedArray::U32(x) => x.mapv(f64::from),
        TypedArray::U64(x) => x.mapv(|v| v as f64),
        other => {
            return Err(CoreError::UnsupportedDType {
                op: "numeric arithmetic",
                dtype: other.dtype(),
            });
        }
    };
    Ok(match conv {
        Some(c) => out.mapv(|v| c.apply(v)),
        None => out,
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

fn is_time(d: DType) -> bool {
    matches!(d, DType::DateTime | DType::CfDate | DType::TimeDelta)
}

/// Dimensions and shape of the broadcast result.
fn out_dims(a: &[(Name, usize)], b: &[(Name, usize)]) -> Result<(Vec<Name>, Vec<usize>)> {
    let mut sizes: IndexMap<Name, usize> = IndexMap::new();
    for (d, n) in a.iter().chain(b) {
        match sizes.get(d) {
            Some(&m) if m != *n => {
                return Err(CoreError::DimLengthMismatch {
                    dim: d.clone(),
                    expected: m,
                    found: *n,
                });
            }
            Some(_) => {}
            None => {
                sizes.insert(d.clone(), *n);
            }
        }
    }
    Ok(sizes.into_iter().unzip())
}

fn td_from_ns(ns: i128) -> Option<TimeDelta> {
    let secs = i64::try_from(ns.div_euclid(1_000_000_000)).ok()?;
    let nanos = u32::try_from(ns.rem_euclid(1_000_000_000)).ok()?;
    TimeDelta::new(secs, nanos)
}

fn td_ns(td: &TimeDelta) -> Option<i128> {
    i128::from(td.num_seconds())
        .checked_mul(1_000_000_000)?
        .checked_add(i128::from(td.subsec_nanos()))
}

macro_rules! int_op {
    ($a:expr, $b:expr, $ad:expr, $bd:expr, $od:expr, $shape:expr, $op:expr, $($var:ident $t:ty),*) => {
        match ($a, $b) {
            $(
                (TypedArray::$var(x), TypedArray::$var(y)) => {
                    let (x, y) = (align(x.view(), $ad, $od), align(y.view(), $bd, $od));
                    let f = |p: &$t, q: &$t| -> Option<$t> {
                        match $op {
                            BinOp::Add => p.checked_add(*q),
                            BinOp::Sub => p.checked_sub(*q),
                            BinOp::Mul => p.checked_mul(*q),
                            BinOp::Div => None,
                        }
                    };
                    TypedArray::$var(zip_map(&x, &y, $shape, f, $op.name())?)
                }
            )*
            (x, _) => return Err(CoreError::UnsupportedDType { op: $op.name(), dtype: x.dtype() }),
        }
    };
}

/// Compute `a op b` on raw data, broadcasting by name. Unit conversions are already decided.
fn compute(
    op: BinOp,
    (a, ad, ac): (&TypedArray<View<'_>>, &[Name], Option<&Converter>),
    (b, bd, bc): (&TypedArray<View<'_>>, &[Name], Option<&Converter>),
    od: &[Name],
    shape: &[usize],
) -> Result<TypedArray<crate::Owned>> {
    let (adt, bdt) = (a.dtype(), b.dtype());
    let unsupported = |dtype| CoreError::UnsupportedDType {
        op: op.name(),
        dtype,
    };
    if is_time(adt) || is_time(bdt) {
        return time_op(op, (a, ad), (b, bd), od, shape);
    }
    if !matches!(adt, DType::F32 | DType::F64) && !is_int(adt) {
        return Err(unsupported(adt));
    }
    if !matches!(bdt, DType::F32 | DType::F64) && !is_int(bdt) {
        return Err(unsupported(bdt));
    }
    let converting = ac.is_some() || bc.is_some();
    if !converting && adt == bdt && is_int(adt) && op != BinOp::Div {
        return Ok(int_op!(a, b, ad, bd, od, shape, op,
            I8 i8, I16 i16, I32 i32, I64 i64, U8 u8, U16 u16, U32 u32, U64 u64));
    }
    if !converting && adt == DType::F32 && bdt == DType::F32 {
        if let (TypedArray::F32(x), TypedArray::F32(y)) = (a, b) {
            let (x, y) = (align(x.view(), ad, od), align(y.view(), bd, od));
            return Ok(TypedArray::F32(zip_map(
                &x,
                &y,
                shape,
                |p, q| Some(float_op(op, *p, *q)),
                op.name(),
            )?));
        }
    }
    let (x, y) = (to_f64(a, ac)?, to_f64(b, bc)?);
    let (x, y) = (align(x.view(), ad, od), align(y.view(), bd, od));
    Ok(TypedArray::F64(zip_map(
        &x,
        &y,
        shape,
        |p, q| Some(float_op(op, *p, *q)),
        op.name(),
    )?))
}

fn float_op<F: num_like::Float>(op: BinOp, p: F, q: F) -> F {
    match op {
        BinOp::Add => p.add(q),
        BinOp::Sub => p.sub(q),
        BinOp::Mul => p.mul(q),
        BinOp::Div => p.div(q),
    }
}

mod num_like {
    /// `f32` or `f64`. IEEE arithmetic never panics (it yields ±∞ or NaN instead).
    pub(crate) trait Float: Copy {
        fn add(self, q: Self) -> Self;
        fn sub(self, q: Self) -> Self;
        fn mul(self, q: Self) -> Self;
        fn div(self, q: Self) -> Self;
    }

    macro_rules! float {
        ($($t:ty),*) => {$(
            impl Float for $t {
                fn add(self, q: Self) -> Self { self + q }
                fn sub(self, q: Self) -> Self { self - q }
                fn mul(self, q: Self) -> Self { self * q }
                fn div(self, q: Self) -> Self { self / q }
            }
        )*};
    }
    float!(f32, f64);
}

fn time_op(
    op: BinOp,
    (a, ad): (&TypedArray<View<'_>>, &[Name]),
    (b, bd): (&TypedArray<View<'_>>, &[Name]),
    od: &[Name],
    shape: &[usize],
) -> Result<TypedArray<crate::Owned>> {
    let name = op.name();
    macro_rules! go {
        ($x:expr, $y:expr, $f:expr) => {
            zip_map(
                &align($x.view(), ad, od),
                &align($y.view(), bd, od),
                shape,
                $f,
                name,
            )?
        };
    }
    Ok(match (a, b, op) {
        (TypedArray::DateTime(x), TypedArray::DateTime(y), BinOp::Sub) => {
            TypedArray::TimeDelta(go!(x, y, |p: &NaiveDateTime, q: &NaiveDateTime| Some(
                p.signed_duration_since(*q)
            )))
        }
        (TypedArray::DateTime(x), TypedArray::TimeDelta(y), BinOp::Add) => {
            TypedArray::DateTime(go!(x, y, |p: &NaiveDateTime, q: &TimeDelta| p
                .checked_add_signed(*q)))
        }
        (TypedArray::DateTime(x), TypedArray::TimeDelta(y), BinOp::Sub) => {
            TypedArray::DateTime(go!(x, y, |p: &NaiveDateTime, q: &TimeDelta| p
                .checked_sub_signed(*q)))
        }
        (TypedArray::TimeDelta(x), TypedArray::DateTime(y), BinOp::Add) => {
            TypedArray::DateTime(go!(x, y, |p: &TimeDelta, q: &NaiveDateTime| q
                .checked_add_signed(*p)))
        }
        (TypedArray::TimeDelta(x), TypedArray::TimeDelta(y), BinOp::Add) => {
            TypedArray::TimeDelta(go!(x, y, |p: &TimeDelta, q: &TimeDelta| p.checked_add(q)))
        }
        (TypedArray::TimeDelta(x), TypedArray::TimeDelta(y), BinOp::Sub) => {
            TypedArray::TimeDelta(go!(x, y, |p: &TimeDelta, q: &TimeDelta| p.checked_sub(q)))
        }
        (TypedArray::CfDate(x), TypedArray::CfDate(y), BinOp::Sub) => {
            TypedArray::TimeDelta(go!(x, y, |p: &CfDate, q: &CfDate| p
                .nanos_since(q)
                .ok()
                .and_then(td_from_ns)))
        }
        (TypedArray::CfDate(x), TypedArray::TimeDelta(y), BinOp::Add | BinOp::Sub) => {
            let sub = op == BinOp::Sub;
            TypedArray::CfDate(go!(x, y, |p: &CfDate, q: &TimeDelta| {
                let ns = td_ns(q)?;
                p.checked_add_nanos(if sub { ns.checked_neg()? } else { ns })
                    .ok()
            }))
        }
        (x, y, _) => {
            let dtype = if is_time(x.dtype()) {
                x.dtype()
            } else {
                y.dtype()
            };
            return Err(CoreError::UnsupportedDType { op: name, dtype });
        }
    })
}

// ----- data-array arithmetic -----

impl<S: Repr> DataArrayBase<S> {
    /// `self op rhs` with an explicit unit policy.
    ///
    /// # Errors
    /// Unit errors ([`UnitError::ScaleMismatch`](xr_units::UnitError::ScaleMismatch) under the
    /// strict policy, incompatible units), [`CoreError::DimLengthMismatch`],
    /// [`CoreError::NotAligned`], [`CoreError::UnsupportedDType`], or
    /// [`CoreError::ArithmeticOverflow`] for integer overflow.
    pub fn binary<R: Repr>(
        &self,
        op: BinOp,
        rhs: &DataArrayBase<R>,
        policy: UnitPolicy,
    ) -> Result<DataArray> {
        // Coordinates of shared dimensions must agree.
        for d in self.dims().iter().filter(|d| rhs.dims().contains(d)) {
            if let (Some(a), Some(b)) = (self.coords.get(d), rhs.coords.get(d)) {
                if !variables_equal(a, b) {
                    return Err(CoreError::NotAligned(d.clone()));
                }
            }
        }

        // Units.
        let one = Unit::one();
        let (lu, ru) = (self.unit().unwrap_or(&one), rhs.unit().unwrap_or(&one));
        let time = is_time(self.dtype()) || is_time(rhs.dtype());
        let (unit, lconv, rconv) = if time {
            if self.unit().is_some() || rhs.unit().is_some() {
                return Err(CoreError::UnsupportedDType {
                    op: "unit-carrying time arithmetic",
                    dtype: self.dtype(),
                });
            }
            (None, None, None)
        } else {
            match op {
                BinOp::Add | BinOp::Sub => {
                    let plan = if op == BinOp::Add {
                        lu.plan_add(ru, policy)?
                    } else {
                        lu.plan_sub(ru, policy)?
                    };
                    (Some(plan.result), plan.lhs, plan.rhs)
                }
                BinOp::Mul => (Some(lu.mul(ru)?), None, None),
                BinOp::Div => (Some(lu.div(ru)?), None, None),
            }
        };
        let unit = if self.unit().is_none() && rhs.unit().is_none() {
            None
        } else {
            unit
        };

        // Data.
        let sizes = |a: &dyn Fn() -> Vec<(Name, usize)>| a();
        let (od, shape) = out_dims(
            &sizes(&|| self.sizes().map(|(d, n)| (d.clone(), n)).collect()),
            &sizes(&|| rhs.sizes().map(|(d, n)| (d.clone(), n)).collect()),
        )?;
        let data = compute(
            op,
            (&self.variable().typed().view(), self.dims(), lconv.as_ref()),
            (&rhs.variable().typed().view(), rhs.dims(), rconv.as_ref()),
            &od,
            &shape,
        )?;
        let mut var = Variable::from_typed(od, data)?;
        var.set_unit(
            unit.map(Unit::without_label)
                .map(|u| match (self.unit(), &u) {
                    // Keep the left operand's spelling when the unit is unchanged.
                    (Some(l), u) if l == u => l.clone(),
                    _ => u,
                }),
        );

        // Coordinates: union, left operand first.
        let mut coords: IndexMap<Name, Variable> = self
            .coords
            .iter()
            .map(|(k, v)| (k.clone(), v.to_owned()))
            .collect();
        for (k, v) in &rhs.coords {
            coords.entry(k.clone()).or_insert_with(|| v.to_owned());
        }
        let name = match (self.name(), rhs.name()) {
            (Some(a), Some(b)) if a == b => Some(a.clone()),
            _ => None,
        };
        Ok(DataArrayBase::from_parts(name, var, coords))
    }

    /// `self + rhs` (strict units).
    ///
    /// # Errors
    /// As for [`binary`](Self::binary).
    pub fn try_add<R: Repr>(&self, rhs: &DataArrayBase<R>) -> Result<DataArray> {
        self.binary(BinOp::Add, rhs, UnitPolicy::Strict)
    }

    /// `self - rhs` (strict units).
    ///
    /// # Errors
    /// As for [`binary`](Self::binary).
    pub fn try_sub<R: Repr>(&self, rhs: &DataArrayBase<R>) -> Result<DataArray> {
        self.binary(BinOp::Sub, rhs, UnitPolicy::Strict)
    }

    /// `self * rhs`.
    ///
    /// # Errors
    /// As for [`binary`](Self::binary).
    pub fn try_mul<R: Repr>(&self, rhs: &DataArrayBase<R>) -> Result<DataArray> {
        self.binary(BinOp::Mul, rhs, UnitPolicy::Strict)
    }

    /// `self / rhs` (always floating point).
    ///
    /// # Errors
    /// As for [`binary`](Self::binary).
    pub fn try_div<R: Repr>(&self, rhs: &DataArrayBase<R>) -> Result<DataArray> {
        self.binary(BinOp::Div, rhs, UnitPolicy::Strict)
    }

    /// `self op scalar`, where the scalar is dimensionless.
    ///
    /// # Errors
    /// As for [`binary`](Self::binary): adding a plain number to data with a dimensional unit
    /// is an error.
    pub fn binary_scalar(&self, op: BinOp, scalar: f64) -> Result<DataArray> {
        self.binary(op, &scalar_array(scalar)?, UnitPolicy::Strict)
    }

    /// Convert the data to another unit (as `f64`).
    ///
    /// # Errors
    /// Unit errors if the units are incompatible, or [`CoreError::UnsupportedDType`].
    pub fn to_unit(&self, unit: &Unit) -> Result<DataArray> {
        let from = self.unit().cloned().unwrap_or_else(Unit::one);
        let conv = from.converter_to(unit)?;
        let data = to_f64(&self.variable().typed().view(), Some(&conv))?;
        let var = Variable::from_typed(self.dims().iter().cloned(), TypedArray::F64(data))?
            .with_unit(unit.clone());
        let coords = self
            .coords
            .iter()
            .map(|(k, v)| (k.clone(), v.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
    }
}

fn scalar_array(v: f64) -> Result<DataArray> {
    Ok(DataArray::new(Variable::new(
        Vec::<Name>::new(),
        ArrayD::from_elem(IxDyn(&[]), v),
    )?))
}

fn scalar_lhs(op: BinOp, scalar: f64, rhs: &DataArrayBase<impl Repr>) -> Result<DataArray> {
    scalar_array(scalar)?
        .binary(op, rhs, UnitPolicy::Strict)
        .map(|mut r| {
            // Keep the array operand's dimension order.
            r.name = rhs.name().cloned();
            r
        })
}

macro_rules! impl_ops {
    ($($tr:ident $m:ident $op:ident),*) => {$(
        impl<S: Repr, R: Repr> $tr<&DataArrayBase<R>> for &DataArrayBase<S> {
            type Output = Result<DataArray>;
            fn $m(self, rhs: &DataArrayBase<R>) -> Result<DataArray> {
                self.binary(BinOp::$op, rhs, UnitPolicy::Strict)
            }
        }
        impl<R: Repr> $tr<&DataArrayBase<R>> for Result<DataArray> {
            type Output = Result<DataArray>;
            fn $m(self, rhs: &DataArrayBase<R>) -> Result<DataArray> {
                self?.binary(BinOp::$op, rhs, UnitPolicy::Strict)
            }
        }
        impl<S: Repr> $tr<Result<DataArray>> for &DataArrayBase<S> {
            type Output = Result<DataArray>;
            fn $m(self, rhs: Result<DataArray>) -> Result<DataArray> {
                self.binary(BinOp::$op, &rhs?, UnitPolicy::Strict)
            }
        }
        impl<S: Repr> $tr<f64> for &DataArrayBase<S> {
            type Output = Result<DataArray>;
            fn $m(self, rhs: f64) -> Result<DataArray> {
                self.binary_scalar(BinOp::$op, rhs)
            }
        }
        impl<S: Repr> $tr<&DataArrayBase<S>> for f64 {
            type Output = Result<DataArray>;
            fn $m(self, rhs: &DataArrayBase<S>) -> Result<DataArray> {
                scalar_lhs(BinOp::$op, self, rhs)
            }
        }
    )*};
}

impl_ops!(Add add Add, Sub sub Sub, Mul mul Mul, Div div Div);
