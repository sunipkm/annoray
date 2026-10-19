//! Unit-aware arithmetic on data arrays, broadcasting by dimension name.
//!
//! - **Broadcasting** is by name: the result has the left operand's dimensions followed by the
//!   right operand's extra ones. Shared dimensions must have equal lengths.
//! - **Coordinates**: operands are first aligned on shared dimensions with an inner join
//!   (only labels present in both are kept, see [`align`](crate::align)); the result keeps
//!   the union of both operands' coordinates.
//! - **Units** follow `annoray-units`: `+`/`-` need compatible units and, under the default strict
//!   policy, equal scales; `*`/`/` combine units. A missing unit counts as dimensionless.
//! - **Element types**: two `f32` operands stay `f32`; integers of the same type stay integers
//!   (checked, so overflow is an error) except for division; everything else is `f64`.
//!   Datetimes and durations support the usual differences and offsets.
//! - **Operators** return `Result<DataArray>` and accept a `Result` on either side of a data
//!   array, so `(&a + &b * &c)?` works with a single `?`. (Rust's orphan rules forbid
//!   `Result + Result` and `Result * f64`; use `?` first there.)

use std::ops::{Add, Div, Mul, Sub};

use annoray_units::{CfDate, Converter, Unit, UnitPolicy};
use chrono::{NaiveDateTime, TimeDelta};
use indexmap::IndexMap;
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};

use crate::array::typed_dispatch;
use crate::dataarray::DataArrayBase;
use crate::{
    CoreError, DType, DataArray, Element, NaT, Name, Repr, Result, TypedArray, Variable,
    VariableBase, View,
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
    /// The elementwise maximum (numpy's `maximum`: NaN wins). Units as for a comparison:
    /// the right operand is converted into the left's unit, which the result keeps.
    Max,
    /// The elementwise minimum (numpy's `minimum`: NaN wins); units as for [`Max`](Self::Max).
    Min,
}

impl BinOp {
    const fn name(self) -> &'static str {
        match self {
            Self::Add => "addition",
            Self::Sub => "subtraction",
            Self::Mul => "multiplication",
            Self::Div => "division",
            Self::Max => "maximum",
            Self::Min => "minimum",
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
        && match (a.typed(), b.typed()) {
            (Ok(x), Ok(y)) => typed_dispatch!(&x, v => same(v, &y)),
            _ => false,
        }
}

// ----- broadcasting -----

/// Reorder `v`'s axes into `out_dims` order and insert length-1 axes for missing dimensions.
///
/// # Errors
/// [`CoreError::DimCountMismatch`] unless `dims` names each axis of `v` once and all of them
/// are in `out_dims`.
pub(crate) fn broadcast_to_dims<'a, T>(
    v: ArrayViewD<'a, T>,
    dims: &[Name],
    out_dims: &[Name],
) -> Result<ArrayViewD<'a, T>> {
    let order: Vec<usize> = out_dims
        .iter()
        .filter_map(|d| dims.iter().position(|x| x == d))
        .collect();
    // `permuted_axes` needs a permutation of v's axes (it panics otherwise).
    let mut seen = vec![false; v.ndim()];
    let permutation = order.len() == v.ndim()
        && dims.len() == v.ndim()
        && order.iter().all(|i| {
            seen.get_mut(*i)
                .is_some_and(|s| !std::mem::replace(s, true))
        });
    if !permutation {
        return Err(CoreError::DimCountMismatch {
            ndim: v.ndim(),
            names: order.len(),
        });
    }
    let mut v = v.permuted_axes(IxDyn(&order));
    for (i, d) in out_dims.iter().enumerate() {
        if !dims.contains(d) {
            // Inserting in increasing order keeps `i <= v.ndim()`.
            v = v.insert_axis(Axis(i));
        }
    }
    Ok(v)
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
pub(crate) fn to_f64(a: &TypedArray<View<'_>>, conv: Option<&Converter>) -> Result<ArrayD<f64>> {
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
pub(crate) fn out_dims(
    a: &[(Name, usize)],
    b: &[(Name, usize)],
) -> Result<(Vec<Name>, Vec<usize>)> {
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

pub(crate) fn td_from_ns(ns: i128) -> Option<TimeDelta> {
    let secs = i64::try_from(ns.div_euclid(1_000_000_000)).ok()?;
    let nanos = u32::try_from(ns.rem_euclid(1_000_000_000)).ok()?;
    TimeDelta::new(secs, nanos)
}

pub(crate) fn td_ns(td: &TimeDelta) -> Option<i128> {
    i128::from(td.num_seconds())
        .checked_mul(1_000_000_000)?
        .checked_add(i128::from(td.subsec_nanos()))
}

macro_rules! int_op {
    ($a:expr, $b:expr, $ad:expr, $bd:expr, $od:expr, $shape:expr, $op:expr, $($var:ident $t:ty),*) => {
        match ($a, $b) {
            $(
                (TypedArray::$var(x), TypedArray::$var(y)) => {
                    let (x, y) = (broadcast_to_dims(x.view(), $ad, $od)?, broadcast_to_dims(y.view(), $bd, $od)?);
                    let f = |p: &$t, q: &$t| -> Option<$t> {
                        match $op {
                            BinOp::Add => p.checked_add(*q),
                            BinOp::Sub => p.checked_sub(*q),
                            BinOp::Mul => p.checked_mul(*q),
                            BinOp::Div => None,
                            BinOp::Max => Some(*p.max(q)),
                            BinOp::Min => Some(*p.min(q)),
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
    if !converting
        && adt == DType::F32
        && bdt == DType::F32
        && let (TypedArray::F32(x), TypedArray::F32(y)) = (a, b)
    {
        let (x, y) = (
            broadcast_to_dims(x.view(), ad, od)?,
            broadcast_to_dims(y.view(), bd, od)?,
        );
        return Ok(TypedArray::F32(float_zip(&x, &y, shape, op)?));
    }
    if !converting && let (TypedArray::F64(x), TypedArray::F64(y)) = (a, b) {
        let (x, y) = (
            broadcast_to_dims(x.view(), ad, od)?,
            broadcast_to_dims(y.view(), bd, od)?,
        );
        return Ok(TypedArray::F64(float_zip(&x, &y, shape, op)?));
    }
    let (x, y) = (to_f64(a, ac)?, to_f64(b, bc)?);
    let (x, y) = (
        broadcast_to_dims(x.view(), ad, od)?,
        broadcast_to_dims(y.view(), bd, od)?,
    );
    Ok(TypedArray::F64(float_zip(&x, &y, shape, op)?))
}

/// `y = x op y`, elementwise in place (same shapes).
fn in_place<F: num_like::Float + Send + Sync>(x: &ArrayViewD<'_, F>, y: &mut ArrayD<F>, op: BinOp) {
    fn go<F: Copy + Send + Sync>(
        x: &ArrayViewD<'_, F>,
        y: &mut ArrayD<F>,
        f: impl Fn(F, F) -> F + Sync + Send,
    ) {
        let zip = ndarray::Zip::from(y).and(x);
        let g = |q: &mut F, p: &F| *q = f(*p, *q);
        #[cfg(feature = "rayon")]
        if x.len() > PAR_ELEMENTS {
            zip.par_for_each(g);
            return;
        }
        zip.for_each(g);
    }
    match op {
        BinOp::Add => go(x, y, |p, q| p.add(q)),
        BinOp::Sub => go(x, y, |p, q| p.sub(q)),
        BinOp::Mul => go(x, y, |p, q| p.mul(q)),
        BinOp::Div => go(x, y, |p, q| p.div(q)),
        BinOp::Max => go(x, y, |p, q| p.max_nan(q)),
        BinOp::Min => go(x, y, |p, q| p.min_nan(q)),
    }
}

/// Elements above which elementwise float arithmetic is split across threads.
#[cfg(feature = "rayon")]
const PAR_ELEMENTS: usize = 1 << 16;

/// Elementwise float arithmetic, broadcasting both operands to `shape`. Each operator gets
/// its own loop (no per-element dispatch), and a single-element operand is applied as a
/// constant over the other one (no zero-stride broadcasting).
fn float_zip<F: num_like::Float + Send + Sync>(
    x: &ArrayViewD<'_, F>,
    y: &ArrayViewD<'_, F>,
    shape: &[usize],
    op: BinOp,
) -> Result<ArrayD<F>> {
    match op {
        BinOp::Add => zip_with(x, y, shape, op, |p, q| p.add(q)),
        BinOp::Sub => zip_with(x, y, shape, op, |p, q| p.sub(q)),
        BinOp::Mul => zip_with(x, y, shape, op, |p, q| p.mul(q)),
        BinOp::Div => zip_with(x, y, shape, op, |p, q| p.div(q)),
        BinOp::Max => zip_with(x, y, shape, op, |p, q| p.max_nan(q)),
        BinOp::Min => zip_with(x, y, shape, op, |p, q| p.min_nan(q)),
    }
}

fn zip_with<F: Copy + Send + Sync>(
    x: &ArrayViewD<'_, F>,
    y: &ArrayViewD<'_, F>,
    shape: &[usize],
    op: BinOp,
    f: impl Fn(F, F) -> F + Sync + Send + Copy,
) -> Result<ArrayD<F>> {
    let bad = || CoreError::ArithmeticOverflow(op.name());
    let single = |a: &ArrayViewD<'_, F>| (a.len() == 1).then(|| a.iter().next().copied()).flatten();
    // One operand a constant: map the other (when it already has the result's shape).
    if let Some(q) = single(y)
        && x.shape() == shape
    {
        return Ok(map_one(x, move |p| f(p, q)));
    }
    if let Some(p) = single(x)
        && y.shape() == shape
    {
        return Ok(map_one(y, move |q| f(p, q)));
    }
    let xb = x.broadcast(IxDyn(shape)).ok_or_else(bad)?;
    let yb = y.broadcast(IxDyn(shape)).ok_or_else(bad)?;
    let zip = ndarray::Zip::from(&xb).and(&yb);
    let g = |p: &F, q: &F| f(*p, *q);
    #[cfg(feature = "rayon")]
    if xb.len() > PAR_ELEMENTS {
        return Ok(zip.par_map_collect(g));
    }
    Ok(zip.map_collect(g))
}

fn map_one<F: Copy + Send + Sync>(
    x: &ArrayViewD<'_, F>,
    f: impl Fn(F) -> F + Sync + Send,
) -> ArrayD<F> {
    let zip = ndarray::Zip::from(x);
    let g = |p: &F| f(*p);
    #[cfg(feature = "rayon")]
    if x.len() > PAR_ELEMENTS {
        return zip.par_map_collect(g);
    }
    zip.map_collect(g)
}

mod num_like {
    /// `f32` or `f64`. IEEE arithmetic never panics (it yields ±∞ or NaN instead).
    pub(crate) trait Float: Copy {
        fn add(self, q: Self) -> Self;
        fn sub(self, q: Self) -> Self;
        fn mul(self, q: Self) -> Self;
        fn div(self, q: Self) -> Self;
        /// numpy's `maximum`: NaN if either is NaN.
        fn max_nan(self, q: Self) -> Self;
        /// numpy's `minimum`: NaN if either is NaN.
        fn min_nan(self, q: Self) -> Self;
    }

    macro_rules! float {
        ($($t:ty),*) => {$(
            impl Float for $t {
                fn add(self, q: Self) -> Self { self + q }
                fn sub(self, q: Self) -> Self { self - q }
                fn mul(self, q: Self) -> Self { self * q }
                fn div(self, q: Self) -> Self { self / q }
                fn max_nan(self, q: Self) -> Self {
                    if self.is_nan() || q.is_nan() { <$t>::NAN } else { self.max(q) }
                }
                fn min_nan(self, q: Self) -> Self {
                    if self.is_nan() || q.is_nan() { <$t>::NAN } else { self.min(q) }
                }
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
                &broadcast_to_dims($x.view(), ad, od)?,
                &broadcast_to_dims($y.view(), bd, od)?,
                shape,
                $f,
                name,
            )?
        };
    }
    // NaT in either operand gives NaT.
    macro_rules! nat {
        ($out:ty, |$p:ident: $pt:ty, $q:ident: $qt:ty| $body:expr) => {
            |$p: $pt, $q: $qt| -> Option<$out> {
                if $p.is_nat() || $q.is_nat() {
                    Some(<$out as NaT>::NAT)
                } else {
                    $body
                }
            }
        };
    }
    Ok(match (a, b, op) {
        (TypedArray::DateTime(x), TypedArray::DateTime(y), BinOp::Sub) => {
            TypedArray::TimeDelta(go!(
                x,
                y,
                nat!(TimeDelta, |p: &NaiveDateTime, q: &NaiveDateTime| Some(
                    p.signed_duration_since(*q)
                ))
            ))
        }
        (TypedArray::DateTime(x), TypedArray::TimeDelta(y), BinOp::Add) => {
            TypedArray::DateTime(go!(
                x,
                y,
                nat!(NaiveDateTime, |p: &NaiveDateTime, q: &TimeDelta| p
                    .checked_add_signed(*q))
            ))
        }
        (TypedArray::DateTime(x), TypedArray::TimeDelta(y), BinOp::Sub) => {
            TypedArray::DateTime(go!(
                x,
                y,
                nat!(NaiveDateTime, |p: &NaiveDateTime, q: &TimeDelta| p
                    .checked_sub_signed(*q))
            ))
        }
        (TypedArray::TimeDelta(x), TypedArray::DateTime(y), BinOp::Add) => {
            TypedArray::DateTime(go!(
                x,
                y,
                nat!(NaiveDateTime, |p: &TimeDelta, q: &NaiveDateTime| q
                    .checked_add_signed(*p))
            ))
        }
        (TypedArray::TimeDelta(x), TypedArray::TimeDelta(y), BinOp::Add) => {
            TypedArray::TimeDelta(go!(
                x,
                y,
                nat!(TimeDelta, |p: &TimeDelta, q: &TimeDelta| p.checked_add(q))
            ))
        }
        (TypedArray::TimeDelta(x), TypedArray::TimeDelta(y), BinOp::Sub) => {
            TypedArray::TimeDelta(go!(
                x,
                y,
                nat!(TimeDelta, |p: &TimeDelta, q: &TimeDelta| p.checked_sub(q))
            ))
        }
        (TypedArray::CfDate(x), TypedArray::CfDate(y), BinOp::Sub) => TypedArray::TimeDelta(go!(
            x,
            y,
            nat!(TimeDelta, |p: &CfDate, q: &CfDate| p
                .nanos_since(q)
                .ok()
                .and_then(td_from_ns))
        )),
        (TypedArray::CfDate(x), TypedArray::TimeDelta(y), BinOp::Add | BinOp::Sub) => {
            let sub = op == BinOp::Sub;
            TypedArray::CfDate(go!(
                x,
                y,
                nat!(CfDate, |p: &CfDate, q: &TimeDelta| {
                    let ns = td_ns(q)?;
                    p.checked_add_nanos(if sub { ns.checked_neg()? } else { ns })
                        .ok()
                })
            ))
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
    /// Unit errors ([`UnitError::ScaleMismatch`](annoray_units::UnitError::ScaleMismatch) under the
    /// strict policy, incompatible units), [`CoreError::DimLengthMismatch`],
    /// [`CoreError::NotAligned`], [`CoreError::UnsupportedDType`], or
    /// [`CoreError::ArithmeticOverflow`] for integer overflow.
    pub fn binary<R: Repr>(
        &self,
        op: BinOp,
        rhs: &DataArrayBase<R>,
        policy: UnitPolicy,
    ) -> Result<DataArray> {
        // Align shared dimensions on their labels (an inner join, as xarray does).
        let plan = crate::align::plan(
            &[
                crate::align::Obj::of_array(self),
                crate::align::Obj::of_array(rhs),
            ],
            crate::Join::Inner,
        )?;
        if !plan.is_noop() {
            let a = crate::align::apply_array(self, &plan, 0)?;
            let b = crate::align::apply_array(rhs, &plan, 1)?;
            // `b` is a fresh copy: the result can go into its buffer.
            return a.binary_reusing_rhs(op, b, policy);
        }
        // Coordinates of shared dimensions must agree.
        for d in self.dims().iter().filter(|d| rhs.dims().contains(d)) {
            if let (Some(a), Some(b)) = (self.coords.get(d), rhs.coords.get(d))
                && !variables_equal(a, b)
            {
                return Err(CoreError::NotAligned(d.clone()));
            }
        }

        let (unit, lconv, rconv) = self.plan_units(op, rhs, policy)?;

        // Data.
        let sizes = |a: &dyn Fn() -> Vec<(Name, usize)>| a();
        let (od, shape) = out_dims(
            &sizes(&|| self.sizes().map(|(d, n)| (d.clone(), n)).collect()),
            &sizes(&|| rhs.sizes().map(|(d, n)| (d.clone(), n)).collect()),
        )?;
        // Lazy operands give a lazy result: computed box by box when read.
        if self.variable().is_lazy() || rhs.variable().is_lazy() {
            let src = BinarySource::new(
                op,
                (self.variable(), lconv),
                (rhs.variable(), rconv),
                od.clone(),
                shape,
            )?;
            let var = Variable::from_lazy(od, crate::LazyArray::new(std::sync::Arc::new(src)))?;
            return Ok(self.assemble(
                var,
                unit,
                rhs.name(),
                rhs.attrs(),
                rhs.coords.iter().map(|(k, v)| (k, v.to_owned())),
            ));
        }
        let data = compute(
            op,
            (&self.variable().typed()?, self.dims(), lconv.as_ref()),
            (&rhs.variable().typed()?, rhs.dims(), rconv.as_ref()),
            &od,
            &shape,
        )?;
        let var = Variable::from_typed(od, data)?;
        Ok(self.assemble(
            var,
            unit,
            rhs.name(),
            rhs.attrs(),
            rhs.coords.iter().map(|(k, v)| (k, v.to_owned())),
        ))
    }

    /// The units of `self op rhs`, and how to convert each operand first.
    fn plan_units<R: Repr>(
        &self,
        op: BinOp,
        rhs: &DataArrayBase<R>,
        policy: UnitPolicy,
    ) -> Result<(Option<Unit>, Option<Converter>, Option<Converter>)> {
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
                BinOp::Max | BinOp::Min => {
                    let plan = lu.plan_cmp(ru, policy)?;
                    (Some(plan.result), plan.lhs, plan.rhs)
                }
            }
        };
        let unit = if self.unit().is_none() && rhs.unit().is_none() {
            None
        } else {
            unit
        };
        Ok((unit, lconv, rconv))
    }

    /// The result of a binary operation from its data: the unit (keeping the left operand's
    /// spelling when unchanged), the union of the coordinates (left first), and the name if
    /// both operands share it.
    pub(crate) fn assemble<'a>(
        &self,
        mut var: Variable,
        unit: Option<Unit>,
        rhs_name: Option<&Name>,
        rhs_attrs: &crate::Attrs,
        rhs_coords: impl Iterator<Item = (&'a Name, Variable)>,
    ) -> DataArray {
        // Attributes of both operands, conflicting keys dropped (xarray's default).
        *var.attrs_mut() = crate::attrs::drop_conflicts(self.attrs(), rhs_attrs);
        // Composed spellings (`km s-1`) are verified when made; an unchanged unit keeps the
        // left operand's spelling.
        var.set_unit(unit.map(|u| match (self.unit(), &u) {
            (Some(l), u) if l == u => l.clone(),
            _ => u,
        }));
        let mut coords: IndexMap<Name, Variable> = self
            .coords
            .iter()
            .map(|(k, v)| (k.clone(), v.to_owned()))
            .collect();
        for (k, v) in rhs_coords {
            coords.entry(k.clone()).or_insert(v);
        }
        let name = match (self.name(), rhs_name) {
            (Some(a), Some(b)) if a == b => Some(a.clone()),
            _ => None,
        };
        DataArrayBase::from_parts(name, var, coords)
    }

    /// [`binary`](Self::binary) taking ownership of `rhs`: when nothing needs aligning,
    /// broadcasting or converting and both operands are floats of one type, the result is
    /// written into `rhs`'s buffer instead of a new one. Otherwise as `binary`.
    pub(crate) fn binary_reusing_rhs(
        &self,
        op: BinOp,
        rhs: DataArray,
        policy: UnitPolicy,
    ) -> Result<DataArray> {
        let plan = crate::align::plan(
            &[
                crate::align::Obj::of_array(self),
                crate::align::Obj::of_array(&rhs),
            ],
            crate::Join::Inner,
        )?;
        let coords_agree = self
            .dims()
            .iter()
            .filter(|d| rhs.dims().contains(d))
            .all(|d| match (self.coords.get(d), rhs.coords.get(d)) {
                (Some(a), Some(b)) => variables_equal(a, b),
                _ => true,
            });
        let fast = plan.is_noop()
            && coords_agree
            && self.dims() == rhs.dims()
            && self.shape() == rhs.shape()
            && self.dtype() == rhs.dtype()
            && matches!(self.dtype(), DType::F32 | DType::F64)
            && !rhs.variable().is_lazy();
        if !fast {
            return self.binary(op, &rhs, policy);
        }
        let (unit, lconv, rconv) = self.plan_units(op, &rhs, policy)?;
        if lconv.is_some() || rconv.is_some() {
            return self.binary(op, &rhs, policy);
        }
        let lhs = self.variable().typed()?;
        let (rname, rvar, rcoords) = rhs.into_parts();
        let rattrs = rvar.attrs().clone();
        let dims: Vec<Name> = rvar.dims().to_vec();
        let mut data = rvar.into_typed()?;
        match (&lhs, &mut data) {
            (TypedArray::F64(x), TypedArray::F64(y)) => in_place(x, y, op),
            (TypedArray::F32(x), TypedArray::F32(y)) => in_place(x, y, op),
            _ => {
                return Err(CoreError::UnsupportedDType {
                    op: op.name(),
                    dtype: self.dtype(),
                });
            }
        }
        let var = Variable::from_typed(dims, data)?;
        Ok(self.assemble(
            var,
            unit,
            rname.as_ref(),
            &rattrs,
            rcoords.iter().map(|(k, v)| (k, v.clone())),
        ))
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

    /// `self op scalar`, where the scalar is dimensionless; for `f32` data the scalar is
    /// taken as `f32` (numpy's weak scalars), so the result stays `f32`.
    ///
    /// # Errors
    /// As for [`binary`](Self::binary): adding a plain number to data with a dimensional unit
    /// is an error.
    pub fn binary_scalar(&self, op: BinOp, scalar: f64) -> Result<DataArray> {
        // A plain number takes the data's float type, as numpy's weak scalars: `f32` data
        // stays `f32`.
        let s = if self.dtype() == DType::F32 {
            #[allow(clippy::cast_possible_truncation)] // the scalar in the data's type
            DataArray::new(Variable::new(
                Vec::<Name>::new(),
                ArrayD::from_elem(IxDyn(&[]), scalar as f32),
            )?)
        } else {
            scalar_array(scalar)?
        };
        self.binary(op, &s, UnitPolicy::Strict)
    }

    /// Convert the data to another unit: `f32` data stays `f32`, other numbers become `f64`.
    ///
    /// # Errors
    /// Unit errors if the units are incompatible, or [`CoreError::UnsupportedDType`].
    pub fn to_unit(&self, unit: &Unit) -> Result<DataArray> {
        let from = self.unit().cloned().unwrap_or_else(Unit::one);
        let conv = from.converter_to(unit)?;
        let var = if self.variable().is_lazy() {
            let src = ConvertSource::new(self.variable().as_lazy_source()?, conv)?;
            Variable::from_lazy(
                self.dims().iter().cloned(),
                crate::LazyArray::new(std::sync::Arc::new(src)),
            )?
        } else {
            Variable::from_typed(
                self.dims().iter().cloned(),
                converted(&self.variable().typed()?, &conv)?,
            )?
        }
        .with_unit(unit.clone());
        let coords = self
            .coords
            .iter()
            .map(|(k, v)| (k.clone(), v.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(self.name().cloned(), var, coords))
    }
}

// ----- lazy elementwise operations -----

/// One operand of a lazy operation: its source, dimensions and unit conversion.
#[derive(Debug)]
struct Operand {
    src: std::sync::Arc<dyn crate::LazySource>,
    dims: Vec<Name>,
    conv: Option<Converter>,
}

impl Operand {
    /// The operand's part of the result box `[start, start + count)` (dimensions it lacks
    /// are broadcast).
    fn read(
        &self,
        od: &[Name],
        start: &[usize],
        count: &[usize],
    ) -> Result<TypedArray<crate::Owned>> {
        let mut s = Vec::with_capacity(self.dims.len());
        let mut c = Vec::with_capacity(self.dims.len());
        for d in &self.dims {
            let i = od
                .iter()
                .position(|x| x == d)
                .ok_or_else(|| CoreError::NoSuchDim(d.clone()))?;
            s.push(start.get(i).copied().unwrap_or(0));
            c.push(count.get(i).copied().unwrap_or(0));
        }
        self.src.read_slice(&s, &c)
    }
}

/// `lhs op rhs` computed for whichever box is read (nothing is read when it is made).
#[derive(Debug)]
struct BinarySource {
    op: BinOp,
    lhs: Operand,
    rhs: Operand,
    od: Vec<Name>,
    shape: Vec<usize>,
    dtype: DType,
}

impl BinarySource {
    fn new<S: Repr, R: Repr>(
        op: BinOp,
        (l, lconv): (&VariableBase<S>, Option<Converter>),
        (r, rconv): (&VariableBase<R>, Option<Converter>),
        od: Vec<Name>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        let lhs = Operand {
            src: l.as_lazy_source()?,
            dims: l.dims().to_vec(),
            conv: lconv,
        };
        let rhs = Operand {
            src: r.as_lazy_source()?,
            dims: r.dims().to_vec(),
            conv: rconv,
        };
        // The result's element type, from the same computation on empty operands (a 0-d
        // operand has its one value; nothing else is read).
        let empty = |o: &Operand| {
            if o.dims.is_empty() {
                return o.src.read_slice(&[], &[]);
            }
            TypedArray::empty(o.src.dtype(), &vec![0; o.dims.len()]).ok_or_else(|| {
                CoreError::UnsupportedDType {
                    op: op.name(),
                    dtype: o.src.dtype(),
                }
            })
        };
        let (le, re) = (empty(&lhs)?, empty(&rhs)?);
        let dtype = compute(
            op,
            (&le.view(), &lhs.dims, lhs.conv.as_ref()),
            (&re.view(), &rhs.dims, rhs.conv.as_ref()),
            &od,
            &vec![0; od.len()],
        )?
        .dtype();
        Ok(Self {
            op,
            lhs,
            rhs,
            od,
            shape,
            dtype,
        })
    }
}

impl crate::LazySource for BinarySource {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        let d = self.od.get(axis)?;
        [&self.lhs, &self.rhs].into_iter().find_map(|o| {
            let a = o.dims.iter().position(|x| x == d)?;
            o.src.chunking(a)
        })
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<crate::Owned>> {
        let a = self.lhs.read(&self.od, start, count)?;
        let b = self.rhs.read(&self.od, start, count)?;
        compute(
            self.op,
            (&a.view(), &self.lhs.dims, self.lhs.conv.as_ref()),
            (&b.view(), &self.rhs.dims, self.rhs.conv.as_ref()),
            &self.od,
            count,
        )
    }
}

/// A unit conversion computed for whichever box is read.
#[derive(Debug)]
struct ConvertSource {
    src: std::sync::Arc<dyn crate::LazySource>,
    conv: Converter,
    dtype: DType,
}

impl ConvertSource {
    fn new(src: std::sync::Arc<dyn crate::LazySource>, conv: Converter) -> Result<Self> {
        let empty = TypedArray::empty(src.dtype(), &vec![0; src.shape().len()]).ok_or(
            CoreError::UnsupportedDType {
                op: "unit conversion",
                dtype: src.dtype(),
            },
        )?;
        let dtype = converted(&empty.view(), &conv)?.dtype();
        Ok(Self { src, conv, dtype })
    }
}

impl crate::LazySource for ConvertSource {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        self.src.shape()
    }
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        self.src.chunking(axis)
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<crate::Owned>> {
        converted(&self.src.read_slice(start, count)?.view(), &self.conv)
    }
}

/// Unit conversion in one pass (on several threads for large arrays).
#[allow(clippy::cast_possible_truncation)] // f32 data stays f32
/// `v`'s values converted to `unit`, as an owned variable with `v`'s attributes and encoding
/// (`f32` stays `f32`, other numbers become `f64`).
///
/// # Errors
/// Unit errors for incompatible units; [`CoreError::UnsupportedDType`].
pub(crate) fn variable_in_unit<R: crate::Repr>(
    v: &crate::VariableBase<R>,
    unit: &Unit,
) -> Result<Variable> {
    let from = v.unit().cloned().unwrap_or_else(Unit::one);
    let conv = from.converter_to(unit)?;
    let mut out = Variable::from_typed(v.dims().iter().cloned(), converted(&v.typed()?, &conv)?)?
        .with_unit(unit.clone());
    *out.attrs_mut() = v.attrs().clone();
    *out.encoding_mut() = v.encoding().clone();
    Ok(out)
}

fn converted(a: &TypedArray<View<'_>>, conv: &Converter) -> Result<TypedArray<crate::Owned>> {
    fn go<T: Copy + Send + Sync, O: Copy + Send + Sync>(
        x: &ArrayViewD<'_, T>,
        f: impl Fn(T) -> O + Sync + Send,
    ) -> ArrayD<O> {
        let zip = ndarray::Zip::from(x);
        let g = |v: &T| f(*v);
        #[cfg(feature = "rayon")]
        if x.len() > PAR_ELEMENTS {
            return zip.par_map_collect(g);
        }
        zip.map_collect(g)
    }
    if let TypedArray::F32(x) = a {
        return Ok(TypedArray::F32(go(x, |v: f32| {
            conv.apply(f64::from(v)) as f32
        })));
    }
    if let TypedArray::F64(x) = a {
        return Ok(TypedArray::F64(go(x, |v: f64| conv.apply(v))));
    }
    Ok(TypedArray::F64(to_f64(a, Some(conv))?))
}

pub(crate) fn scalar_array(v: f64) -> Result<DataArray> {
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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    fn arr(dims: &[&str], shape: &[usize], unit: &str, f: impl Fn(usize) -> f64) -> DataArray {
        let n: usize = shape.iter().product();
        let data = ArrayD::from_shape_vec(IxDyn(shape), (0..n).map(f).collect()).unwrap();
        DataArray::new(
            Variable::new(
                dims.iter().map(|d| Name::from(*d)).collect::<Vec<_>>(),
                data,
            )
            .unwrap()
            .with_unit(unit.parse().unwrap()),
        )
    }

    /// Reusing the right operand's buffer gives exactly what `binary` gives, for every
    /// operator (order matters for `-` and `/`), and falls back where it cannot apply.
    #[test]
    fn reusing_rhs_matches_binary() {
        let n = 1 << 17; // above the parallel threshold
        let a = arr(&["x"], &[n], "m", |i| i as f64 * 0.5 + 1.0);
        let b = arr(&["x"], &[n], "m", |i| (i % 97) as f64 + 0.25);
        for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div] {
            let want = a.binary(op, &b, UnitPolicy::Strict).unwrap();
            let got = a
                .binary_reusing_rhs(op, b.clone(), UnitPolicy::Strict)
                .unwrap();
            assert!(variables_equal(want.variable(), got.variable()), "{op:?}");
        }
        // Fallbacks: broadcasting, unit conversion, mixed types, errors.
        let c = arr(&["y"], &[3], "m", |i| i as f64);
        let km = arr(&["x"], &[n], "km", |i| i as f64);
        let s = arr(&["x"], &[n], "s", |i| i as f64);
        for rhs in [c, km] {
            let want = a.binary(BinOp::Add, &rhs, UnitPolicy::AutoRescale).unwrap();
            let got = a
                .binary_reusing_rhs(BinOp::Add, rhs, UnitPolicy::AutoRescale)
                .unwrap();
            assert!(variables_equal(want.variable(), got.variable()));
        }
        assert!(
            a.binary_reusing_rhs(BinOp::Add, s, UnitPolicy::Strict)
                .is_err()
        );
    }
}
