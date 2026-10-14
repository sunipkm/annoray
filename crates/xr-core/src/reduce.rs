//! Reductions over named dimensions (`sum`, `mean`, `std`, `min`, `argmax`, …), following
//! numpy and xarray.
//!
//! - **Missing values**: with `skipna` (the default) NaNs are ignored, as numpy's `nan*`
//!   functions do: an all-NaN sum is 0, an all-NaN mean is NaN. Without it any NaN gives NaN.
//!   Only floats can be missing.
//! - **Element types**: integer and boolean sums and products are exact (checked) and give
//!   `i64` (`u64` for unsigned input); means, medians and spreads of integers are `f64`;
//!   `f32` stays `f32`. `count`, `argmin` and `argmax` give `i64`; `any` and `all` give `bool`.
//!   Dates support `min`, `max`, `mean`, `median`; durations also `sum`.
//! - **Units**: kept by `sum`, `mean`, `median`, `min`, `max`; `std` of an affine unit (°C)
//!   becomes its difference unit; `var` squares the unit; `prod` raises it to the number of
//!   elements; `count`, `argmin`, `argmax`, `any`, `all` drop it. Summing °C is an error.
//! - **Coordinates** along reduced dimensions are dropped; **attributes** too unless
//!   [`ReduceOptions::keep_attrs`].

use std::cmp::Ordering;
use std::ops::RangeFull;

use chrono::{NaiveDateTime, TimeDelta};
use ndarray::{ArrayD, ArrayView1, ArrayViewD, IxDyn};
use xr_units::{CfDate, Unit, UnitPolicy};

use crate::dataarray::DataArrayBase;
use crate::ops::{td_from_ns, td_ns};
use crate::{
    CoreError, DType, DataArray, Dataset, DatasetBase, Element, Name, Owned, Repr, Result,
    TypedArray, Variable, VariableBase, View,
};

/// A reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Reduction {
    /// Sum.
    Sum,
    /// Product.
    Prod,
    /// Arithmetic mean.
    Mean,
    /// Median.
    Median,
    /// Standard deviation (with [`ReduceOptions::ddof`]).
    Std,
    /// Variance (with [`ReduceOptions::ddof`]).
    Var,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Number of non-missing values.
    Count,
    /// Position of the minimum along one dimension.
    ArgMin,
    /// Position of the maximum along one dimension.
    ArgMax,
    /// Whether any value is true (non-zero).
    Any,
    /// Whether all values are true (non-zero).
    All,
}

impl Reduction {
    /// The operation's name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Mean => "mean",
            Self::Median => "median",
            Self::Std => "std",
            Self::Var => "var",
            Self::Min => "min",
            Self::Max => "max",
            Self::Count => "count",
            Self::ArgMin => "argmin",
            Self::ArgMax => "argmax",
            Self::Any => "any",
            Self::All => "all",
        }
    }

    /// Whether a dataset reduction skips non-numeric variables (xarray's `numeric_only`).
    const fn numeric_only(self) -> bool {
        matches!(
            self,
            Self::Sum | Self::Prod | Self::Mean | Self::Median | Self::Std | Self::Var
        )
    }
}

/// Options for [`reduce`](DataArrayBase::reduce).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ReduceOptions {
    /// Ignore NaNs (default: yes).
    pub skipna: Option<bool>,
    /// Delta degrees of freedom for `std` and `var`: the divisor is `n - ddof`.
    pub ddof: usize,
    /// For `sum` and `prod`: fewer valid values than this gives NaN (and a float result).
    pub min_count: Option<usize>,
    /// Keep the attributes of reduced variables (and of a dataset).
    pub keep_attrs: bool,
}

impl ReduceOptions {
    /// Whether to ignore NaNs.
    #[must_use]
    pub const fn skipna(mut self, skipna: bool) -> Self {
        self.skipna = Some(skipna);
        self
    }

    /// Delta degrees of freedom.
    #[must_use]
    pub const fn ddof(mut self, ddof: usize) -> Self {
        self.ddof = ddof;
        self
    }

    /// Minimum number of valid values for `sum` and `prod`.
    #[must_use]
    pub const fn min_count(mut self, min_count: usize) -> Self {
        self.min_count = Some(min_count);
        self
    }

    /// Keep attributes.
    #[must_use]
    pub const fn keep_attrs(mut self, keep: bool) -> Self {
        self.keep_attrs = keep;
        self
    }
}

/// The dimensions to reduce: `..` for all, a name, or a list of names.
pub trait ReduceDims {
    /// The names, or `None` for all dimensions.
    fn reduce_dims(self) -> Option<Vec<Name>>;
}

impl ReduceDims for RangeFull {
    fn reduce_dims(self) -> Option<Vec<Name>> {
        None
    }
}

impl ReduceDims for &str {
    fn reduce_dims(self) -> Option<Vec<Name>> {
        Some(vec![Name::from(self)])
    }
}

impl ReduceDims for &Name {
    fn reduce_dims(self) -> Option<Vec<Name>> {
        Some(vec![self.clone()])
    }
}

impl<const N: usize> ReduceDims for [&str; N] {
    fn reduce_dims(self) -> Option<Vec<Name>> {
        Some(self.into_iter().map(Name::from).collect())
    }
}

impl ReduceDims for &[&str] {
    fn reduce_dims(self) -> Option<Vec<Name>> {
        Some(self.iter().copied().map(Name::from).collect())
    }
}

impl ReduceDims for &[Name] {
    fn reduce_dims(self) -> Option<Vec<Name>> {
        Some(self.to_vec())
    }
}

impl ReduceDims for Vec<Name> {
    fn reduce_dims(self) -> Option<Vec<Name>> {
        Some(self)
    }
}

/// Check the requested dimensions against `have` (all of them for `None`), without repeats.
fn resolve(have: &[Name], dims: Option<Vec<Name>>) -> Result<Vec<Name>> {
    let Some(dims) = dims else {
        return Ok(have.to_vec());
    };
    let mut out: Vec<Name> = Vec::with_capacity(dims.len());
    for d in dims {
        if !have.contains(&d) {
            return Err(CoreError::NoSuchDim(d));
        }
        if !out.contains(&d) {
            out.push(d);
        }
    }
    Ok(out)
}

// ----- elements -----

/// What reductions need to know about an element type.
trait RedElem: Element {
    /// Whether the value is missing (NaN).
    fn missing(&self) -> bool {
        false
    }
    /// Ordering for `min`/`max` (never called on missing values).
    fn order(&self, other: &Self) -> Ordering;
    /// The value as a float, for numeric types.
    fn as_f64(&self) -> Option<f64> {
        None
    }
    /// The value as an exact integer, for integer and boolean types.
    fn as_i128(&self) -> Option<i128> {
        None
    }
}

macro_rules! red_int {
    ($($t:ty),*) => {$(
        impl RedElem for $t {
            fn order(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
            #[allow(clippy::cast_precision_loss)] // integers are promoted to f64 deliberately
            fn as_f64(&self) -> Option<f64> {
                Some(*self as f64)
            }
            fn as_i128(&self) -> Option<i128> {
                Some(i128::from(*self))
            }
        }
    )*};
}
red_int!(i8, i16, i32, i64, u8, u16, u32, u64);

macro_rules! red_float {
    ($($t:ty),*) => {$(
        impl RedElem for $t {
            fn missing(&self) -> bool {
                self.is_nan()
            }
            fn order(&self, other: &Self) -> Ordering {
                self.total_cmp(other)
            }
            fn as_f64(&self) -> Option<f64> {
                Some(f64::from(*self))
            }
        }
    )*};
}
red_float!(f32, f64);

impl RedElem for bool {
    fn order(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
    fn as_f64(&self) -> Option<f64> {
        Some(if *self { 1.0 } else { 0.0 })
    }
    fn as_i128(&self) -> Option<i128> {
        Some(i128::from(*self))
    }
}

macro_rules! red_ord {
    ($($t:ty),*) => {$(
        impl RedElem for $t {
            fn order(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        }
    )*};
}
red_ord!(String, NaiveDateTime, TimeDelta);

impl RedElem for CfDate {
    fn order(&self, other: &Self) -> Ordering {
        self.nanos_since(other)
            .map_or(Ordering::Equal, |d| d.cmp(&0))
    }
}

// ----- lanes -----

fn product(v: impl IntoIterator<Item = usize>) -> Result<usize> {
    v.into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or(CoreError::ArithmeticOverflow("reduction size"))
}

/// Apply `f` to every lane along `axes` (the lanes are the reduced axes, flattened in order).
fn map_lanes<T: Clone, O>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    mut f: impl FnMut(ArrayView1<'_, T>) -> Result<O>,
) -> Result<ArrayD<O>> {
    let shape = a.shape();
    let len = |i: &usize| shape.get(*i).copied().unwrap_or(0);
    let keep: Vec<usize> = (0..a.ndim()).filter(|i| !axes.contains(i)).collect();
    let out_shape: Vec<usize> = keep.iter().map(len).collect();
    let outer = product(out_shape.iter().copied())?;
    let inner = product(axes.iter().map(len))?;
    let order: Vec<usize> = keep.iter().chain(axes).copied().collect();
    let p = a.view().permuted_axes(IxDyn(&order));
    let m = p
        .to_shape((outer, inner))
        .map_err(|_| CoreError::ArithmeticOverflow("reduction shape"))?;
    let values = m
        .rows()
        .into_iter()
        .map(&mut f)
        .collect::<Result<Vec<O>>>()?;
    ArrayD::from_shape_vec(IxDyn(&out_shape), values)
        .map_err(|_| CoreError::ArithmeticOverflow("reduction shape"))
}

/// Position of the first extreme (`Less` for the minimum), or of the first missing value
/// when not skipping them. `None` if there is no valid value.
fn extreme<T: RedElem>(lane: &ArrayView1<'_, T>, want: Ordering, skipna: bool) -> Option<usize> {
    let mut best: Option<(usize, &T)> = None;
    for (i, v) in lane.iter().enumerate() {
        if v.missing() {
            if !skipna {
                return Some(i);
            }
            continue;
        }
        match best {
            Some((_, b)) if v.order(b) != want => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i)
}

fn min_max<T: RedElem>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    op: Reduction,
    skipna: bool,
) -> Result<TypedArray<Owned>> {
    let want = if op == Reduction::Min {
        Ordering::Less
    } else {
        Ordering::Greater
    };
    let out = map_lanes(a, axes, |lane| {
        extreme(&lane, want, skipna)
            .and_then(|i| lane.get(i))
            // All missing: the result is missing too.
            .or_else(|| lane.iter().next().filter(|v| v.missing()))
            .cloned()
            .ok_or(CoreError::EmptyReduction(op.name()))
    })?;
    Ok(T::wrap(out))
}

fn arg<T: RedElem>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    op: Reduction,
    skipna: bool,
) -> Result<TypedArray<Owned>> {
    let want = if op == Reduction::ArgMin {
        Ordering::Less
    } else {
        Ordering::Greater
    };
    let out = map_lanes(a, axes, |lane| {
        let i = extreme(&lane, want, skipna).ok_or(CoreError::EmptyReduction(op.name()))?;
        i64::try_from(i).map_err(|_| CoreError::ArithmeticOverflow(op.name()))
    })?;
    Ok(TypedArray::I64(out))
}

fn count<T: RedElem>(a: &ArrayViewD<'_, T>, axes: &[usize]) -> Result<TypedArray<Owned>> {
    let out = map_lanes(a, axes, |lane| {
        i64::try_from(lane.iter().filter(|v| !v.missing()).count())
            .map_err(|_| CoreError::ArithmeticOverflow("count"))
    })?;
    Ok(TypedArray::I64(out))
}

fn truth<T: RedElem>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    op: Reduction,
) -> Result<TypedArray<Owned>> {
    let unsupported = || CoreError::UnsupportedDType {
        op: op.name(),
        dtype: T::DTYPE,
    };
    let out = map_lanes(a, axes, |lane| {
        let mut values = lane.iter().map(|v| v.as_f64().map(|x| x != 0.0));
        if op == Reduction::All {
            values.try_fold(true, |acc, v| v.map(|v| acc && v))
        } else {
            values.try_fold(false, |acc, v| v.map(|v| acc || v))
        }
        .ok_or_else(unsupported)
    })?;
    Ok(TypedArray::Bool(out))
}

// ----- numeric statistics -----

#[allow(clippy::cast_precision_loss)] // counts far below 2^53
fn as_count(n: usize) -> f64 {
    n as f64
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / as_count(v.len())
}

/// One lane's statistic, numpy-style (`nan*` functions when skipping NaNs).
fn float_stat(op: Reduction, buf: &mut Vec<f64>, skipna: bool, opts: &ReduceOptions) -> f64 {
    if buf.iter().any(|v| v.is_nan()) {
        if !skipna {
            return f64::NAN;
        }
        buf.retain(|v| !v.is_nan());
    }
    let n = buf.len();
    let sum_like = matches!(op, Reduction::Sum | Reduction::Prod);
    if sum_like && opts.min_count.is_some_and(|m| n < m) {
        return f64::NAN;
    }
    match op {
        Reduction::Sum => buf.iter().sum(),
        Reduction::Prod => buf.iter().product(),
        Reduction::Mean if n > 0 => mean(buf),
        Reduction::Median if n > 0 => {
            buf.sort_by(f64::total_cmp);
            let mid = n / 2;
            let hi = buf.get(mid).copied().unwrap_or(f64::NAN);
            if n % 2 == 1 {
                hi
            } else {
                let lo = mid
                    .checked_sub(1)
                    .and_then(|i| buf.get(i))
                    .copied()
                    .unwrap_or(f64::NAN);
                (lo + hi) / 2.0
            }
        }
        Reduction::Var | Reduction::Std => {
            let Some(dof) = n.checked_sub(opts.ddof).filter(|d| *d > 0) else {
                return f64::NAN;
            };
            let m = mean(buf);
            let var = buf.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / as_count(dof);
            if op == Reduction::Std {
                var.sqrt()
            } else {
                var
            }
        }
        _ => f64::NAN,
    }
}

fn float_stats<T: RedElem>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    op: Reduction,
    skipna: bool,
    opts: &ReduceOptions,
) -> Result<ArrayD<f64>> {
    let unsupported = || CoreError::UnsupportedDType {
        op: op.name(),
        dtype: T::DTYPE,
    };
    let mut buf = Vec::new();
    map_lanes(a, axes, |lane| {
        buf.clear();
        for v in &lane {
            buf.push(v.as_f64().ok_or_else(unsupported)?);
        }
        Ok(float_stat(op, &mut buf, skipna, opts))
    })
}

/// Exact integer sums and products.
fn int_stats<T: RedElem>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    op: Reduction,
) -> Result<ArrayD<i128>> {
    let overflow = || CoreError::ArithmeticOverflow(op.name());
    map_lanes(a, axes, |lane| {
        let mut acc: i128 = if op == Reduction::Prod { 1 } else { 0 };
        for v in &lane {
            let x = v.as_i128().ok_or_else(overflow)?;
            acc = if op == Reduction::Prod {
                acc.checked_mul(x)
            } else {
                acc.checked_add(x)
            }
            .ok_or_else(overflow)?;
        }
        Ok(acc)
    })
}

/// Statistics of dates and durations, on nanosecond offsets (`key`/`back` convert).
fn time_stats<T: RedElem>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    op: Reduction,
    key: impl Fn(&T) -> Option<i128>,
    back: impl Fn(i128) -> Option<T>,
) -> Result<ArrayD<T>> {
    let overflow = || CoreError::ArithmeticOverflow(op.name());
    let mut keys: Vec<i128> = Vec::new();
    map_lanes(a, axes, |lane| {
        keys.clear();
        for v in &lane {
            keys.push(key(v).ok_or_else(overflow)?);
        }
        let n = i128::try_from(keys.len()).map_err(|_| overflow())?;
        if n == 0 {
            return Err(CoreError::EmptyReduction(op.name()));
        }
        let ns = match op {
            Reduction::Sum => keys
                .iter()
                .try_fold(0i128, |acc, k| acc.checked_add(*k))
                .ok_or_else(overflow)?,
            Reduction::Mean => {
                // Rounded to the nearest nanosecond (halves up), as xarray does.
                let s = keys
                    .iter()
                    .try_fold(0i128, |acc, k| acc.checked_add(*k))
                    .ok_or_else(overflow)?;
                let (q, r) = (s.div_euclid(n), s.rem_euclid(n));
                let up = r.checked_mul(2).is_some_and(|r2| r2 >= n);
                if up {
                    q.checked_add(1).ok_or_else(overflow)?
                } else {
                    q
                }
            }
            _ => {
                keys.sort_unstable();
                let mid = keys.len() / 2;
                let hi = keys.get(mid).copied().ok_or_else(overflow)?;
                if keys.len() % 2 == 1 {
                    hi
                } else {
                    let lo = mid
                        .checked_sub(1)
                        .and_then(|i| keys.get(i))
                        .copied()
                        .ok_or_else(overflow)?;
                    hi.checked_sub(lo)
                        .map(|d| d.div_euclid(2))
                        .and_then(|d| lo.checked_add(d))
                        .ok_or_else(overflow)?
                }
            }
        };
        back(ns).ok_or_else(overflow)
    })
}

fn epoch() -> NaiveDateTime {
    chrono::DateTime::UNIX_EPOCH.naive_utc()
}

fn stats(
    a: &TypedArray<View<'_>>,
    op: Reduction,
    axes: &[usize],
    skipna: bool,
    opts: &ReduceOptions,
) -> Result<TypedArray<Owned>> {
    let dtype = a.dtype();
    let unsupported = || CoreError::UnsupportedDType {
        op: op.name(),
        dtype,
    };
    let exact = matches!(op, Reduction::Sum | Reduction::Prod) && opts.min_count.is_none();
    let time_op = matches!(op, Reduction::Mean | Reduction::Median)
        || (op == Reduction::Sum && dtype == DType::TimeDelta);
    Ok(match a {
        #[allow(clippy::cast_possible_truncation)] // f32 data stays f32, as in numpy
        TypedArray::F32(x) => {
            TypedArray::F32(float_stats(x, axes, op, skipna, opts)?.mapv(|v| v as f32))
        }
        TypedArray::F64(x) => TypedArray::F64(float_stats(x, axes, op, skipna, opts)?),
        TypedArray::Str(_) => return Err(unsupported()),
        TypedArray::DateTime(_) | TypedArray::TimeDelta(_) | TypedArray::CfDate(_) if !time_op => {
            return Err(unsupported());
        }
        TypedArray::DateTime(x) => TypedArray::DateTime(time_stats(
            x,
            axes,
            op,
            |v| td_ns(&v.signed_duration_since(epoch())),
            |ns| epoch().checked_add_signed(td_from_ns(ns)?),
        )?),
        TypedArray::TimeDelta(x) => {
            TypedArray::TimeDelta(time_stats(x, axes, op, td_ns, td_from_ns)?)
        }
        TypedArray::CfDate(x) => {
            let Some(base) = x.iter().next().copied() else {
                return Err(CoreError::EmptyReduction(op.name()));
            };
            TypedArray::CfDate(time_stats(
                x,
                axes,
                op,
                |v| v.nanos_since(&base).ok(),
                |ns| base.checked_add_nanos(ns).ok(),
            )?)
        }
        _ if exact => {
            let sums = crate::array::typed_dispatch!(a, x => int_stats(x, axes, op)?);
            let overflow = || CoreError::ArithmeticOverflow(op.name());
            if matches!(dtype, DType::U8 | DType::U16 | DType::U32 | DType::U64) {
                let v = sums
                    .iter()
                    .map(|s| u64::try_from(*s).map_err(|_| overflow()));
                TypedArray::U64(
                    ArrayD::from_shape_vec(sums.raw_dim(), v.collect::<Result<_>>()?)
                        .map_err(|_| overflow())?,
                )
            } else {
                let v = sums
                    .iter()
                    .map(|s| i64::try_from(*s).map_err(|_| overflow()));
                TypedArray::I64(
                    ArrayD::from_shape_vec(sums.raw_dim(), v.collect::<Result<_>>()?)
                        .map_err(|_| overflow())?,
                )
            }
        }
        _ => TypedArray::F64(
            crate::array::typed_dispatch!(a, x => float_stats(x, axes, op, skipna, opts)?),
        ),
    })
}

/// Reduce raw data over `axes`.
fn reduce_typed(
    a: &TypedArray<View<'_>>,
    op: Reduction,
    axes: &[usize],
    opts: &ReduceOptions,
) -> Result<TypedArray<Owned>> {
    let skipna = opts.skipna.unwrap_or(true);
    match op {
        Reduction::Min | Reduction::Max => {
            crate::array::typed_dispatch!(a, x => min_max(x, axes, op, skipna))
        }
        Reduction::ArgMin | Reduction::ArgMax => {
            crate::array::typed_dispatch!(a, x => arg(x, axes, op, skipna))
        }
        Reduction::Count => crate::array::typed_dispatch!(a, x => count(x, axes)),
        Reduction::Any | Reduction::All => {
            crate::array::typed_dispatch!(a, x => truth(x, axes, op))
        }
        _ => stats(a, op, axes, skipna, opts),
    }
}

/// The unit of a reduction's result.
fn reduced_unit(unit: Option<&Unit>, op: Reduction, n: usize) -> Result<Option<Unit>> {
    let Some(u) = unit else { return Ok(None) };
    Ok(match op {
        Reduction::Count
        | Reduction::ArgMin
        | Reduction::ArgMax
        | Reduction::Any
        | Reduction::All => None,
        Reduction::Min | Reduction::Max | Reduction::Mean | Reduction::Median => Some(u.clone()),
        Reduction::Sum => {
            // Fails for units that cannot be added (°C).
            u.plan_add(u, UnitPolicy::Strict)?;
            Some(u.clone())
        }
        Reduction::Std => Some(u.delta()),
        Reduction::Var => Some(u.delta().powi(2)?),
        Reduction::Prod => {
            let n = i32::try_from(n).map_err(|_| CoreError::ArithmeticOverflow("prod unit"))?;
            Some(u.powi(n)?)
        }
    })
}

/// Reduce a variable over the named dimensions (already checked).
fn reduce_var<S: Repr>(
    v: &VariableBase<S>,
    op: Reduction,
    dims: &[Name],
    opts: &ReduceOptions,
) -> Result<Variable> {
    let axes = dims
        .iter()
        .map(|d| v.axis(d))
        .collect::<Result<Vec<usize>>>()?;
    if matches!(op, Reduction::ArgMin | Reduction::ArgMax) && axes.len() != 1 {
        return Err(CoreError::InvalidReduction(format!(
            "{} needs exactly one dimension, got {}",
            op.name(),
            axes.len()
        )));
    }
    let n = product(axes.iter().filter_map(|a| v.shape().get(*a).copied()))?;
    let data = reduce_typed(&v.typed()?, op, &axes, opts)?;
    let out_dims: Vec<Name> = v
        .dims()
        .iter()
        .filter(|d| !dims.contains(d))
        .cloned()
        .collect();
    let mut out = Variable::from_typed(out_dims, data)?;
    out.set_unit(reduced_unit(v.unit(), op, n)?);
    if opts.keep_attrs {
        out.attrs_mut()
            .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    }
    Ok(out)
}

fn is_numeric(d: DType) -> bool {
    !matches!(
        d,
        DType::Str | DType::DateTime | DType::CfDate | DType::TimeDelta
    )
}

// ----- public API -----

macro_rules! shortcuts {
    ($out:ty; $($m:ident $op:ident $doc:literal),* $(,)?) => {$(
        #[doc = $doc]
        ///
        /// `dims` is `..` (all dimensions), a name, or a list of names. Default options; see
        /// [`reduce`](Self::reduce).
        ///
        /// # Errors
        /// As for [`reduce`](Self::reduce).
        pub fn $m(&self, dims: impl ReduceDims) -> Result<$out> {
            self.reduce(Reduction::$op, dims, &ReduceOptions::default())
        }
    )*};
}

macro_rules! all_shortcuts {
    ($out:ty) => {
        shortcuts!($out;
            sum Sum "Sum, skipping NaNs.",
            prod Prod "Product, skipping NaNs.",
            mean Mean "Mean, skipping NaNs.",
            median Median "Median, skipping NaNs.",
            std Std "Population standard deviation (`ddof = 0`), skipping NaNs.",
            var Var "Population variance (`ddof = 0`), skipping NaNs.",
            min Min "Minimum, skipping NaNs.",
            max Max "Maximum, skipping NaNs.",
            count Count "Number of non-NaN values.",
            argmin ArgMin "Position of the minimum along one dimension, skipping NaNs.",
            argmax ArgMax "Position of the maximum along one dimension, skipping NaNs.",
            any Any "Whether any value is non-zero.",
            all All "Whether every value is non-zero.",
        );
    };
}

impl<S: Repr> VariableBase<S> {
    /// Reduce over `dims` (`..` for all).
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::UnsupportedDType`] (e.g. the mean of strings);
    /// [`CoreError::EmptyReduction`] (e.g. the minimum of nothing, or `argmin` of all-NaN);
    /// [`CoreError::ArithmeticOverflow`] for integer sums; unit errors (summing °C);
    /// [`CoreError::InvalidReduction`] for `argmin`/`argmax` over other than one dimension;
    /// [`CoreError::Read`] if lazy data cannot be read.
    pub fn reduce(
        &self,
        op: Reduction,
        dims: impl ReduceDims,
        opts: &ReduceOptions,
    ) -> Result<Variable> {
        let dims = resolve(self.dims(), dims.reduce_dims())?;
        reduce_var(self, op, &dims, opts)
    }

    all_shortcuts!(Variable);
}

impl<S: Repr> DataArrayBase<S> {
    /// Reduce over `dims` (`..` for all). Coordinates along them are dropped.
    ///
    /// # Errors
    /// As for [`VariableBase::reduce`].
    pub fn reduce(
        &self,
        op: Reduction,
        dims: impl ReduceDims,
        opts: &ReduceOptions,
    ) -> Result<DataArray> {
        let dims = resolve(self.dims(), dims.reduce_dims())?;
        let var = reduce_var(&self.variable, op, &dims, opts)?;
        let coords = self
            .coords
            .iter()
            .filter(|(_, c)| !c.dims().iter().any(|d| dims.contains(d)))
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(self.name.clone(), var, coords))
    }

    all_shortcuts!(DataArray);
}

impl<S: Repr> DatasetBase<S> {
    /// Reduce every data variable over those of `dims` it has (`..` for all). Variables with
    /// none of them are kept as they are; for numeric reductions (`sum`, `mean`, …)
    /// non-numeric variables are dropped, as xarray's `numeric_only` does.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] if the dataset lacks a dimension; otherwise as for
    /// [`VariableBase::reduce`].
    pub fn reduce(
        &self,
        op: Reduction,
        dims: impl ReduceDims,
        opts: &ReduceOptions,
    ) -> Result<Dataset> {
        let dims = resolve(&self.dims(), dims.reduce_dims())?;
        let mut out = Dataset::new();
        for (k, c) in self.coords() {
            if !c.dims().iter().any(|d| dims.contains(d)) {
                out.add_coord(k.clone(), c.to_owned())?;
            }
        }
        for (k, v) in self.data_vars() {
            if op.numeric_only() && !is_numeric(v.dtype()) {
                continue;
            }
            let here: Vec<Name> = v
                .dims()
                .iter()
                .filter(|d| dims.contains(d))
                .cloned()
                .collect();
            let r = if here.is_empty() {
                v.to_owned()
            } else {
                reduce_var(v, op, &here, opts)?
            };
            out.add_var(k.clone(), r)?;
        }
        if opts.keep_attrs {
            out.attrs_mut()
                .extend(self.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        }
        Ok(out)
    }

    all_shortcuts!(Dataset);
}
