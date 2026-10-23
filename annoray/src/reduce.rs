//! Reductions over named dimensions (`sum`, `mean`, `std`, `min`, `argmax`, …), following
//! numpy and xarray.
//!
//! - **Missing values**: with `skipna` (the default) NaNs are ignored, as numpy's `nan*`
//!   functions do: an all-NaN sum is 0, an all-NaN mean is NaN. Without it any NaN gives NaN.
//!   Floats can be missing (NaN), and so can times (NaT, see [`NaT`](crate::NaT)), which are
//!   skipped the same way. (Unlike xarray, whose `min`/`max` of dates do not skip NaT by
//!   default while its `mean` does.)
//! - **Element types**: integer and boolean sums and products are exact (checked) and give
//!   `i64` (`u64` for unsigned input); means, medians and spreads of integers are `f64`;
//!   `f32` stays `f32`. `count`, `argmin` and `argmax` give `i64`; `any` and `all` give `bool`.
//!   Dates support `min`, `max`, `mean`, `median`; durations also `sum`. Complex numbers
//!   support `sum`, `prod` and `mean` (complex, accumulated in `f64`, `complex64` staying
//!   `complex64`), `var` and `std` (real: the mean squared modulus of the deviations, as
//!   numpy's) and `count`; they have no order, so `min`, `max`, `median`, `argmin`, `argmax`,
//!   `any` and `all` are errors. A complex value is missing when either part is NaN.
//! - **Units**: kept by `sum`, `mean`, `median`, `min`, `max`; `std` of an affine unit (°C)
//!   becomes its difference unit; `var` squares the unit; `prod` raises it to the number of
//!   elements; `count`, `argmin`, `argmax`, `any`, `all` drop it. Summing °C is an error.
//! - **Coordinates** along reduced dimensions are dropped; **attributes** are kept, as
//!   xarray's default `keep_attrs` (units are not attributes, so they never go stale), unless
//!   [`ReduceOptions::keep_attrs`] is false.

use std::cmp::Ordering;
use std::ops::RangeFull;

use annoray_units::{CfDate, Unit, UnitPolicy};
use annoray_units::{Duration, Instant};
use ndarray::{ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Axis, Dimension, IxDyn};
use num_complex::Complex;

use crate::dataarray::DataArrayBase;
use crate::ops::{td_from_ns, td_ns};
use crate::{
    CoreError, DType, DataArray, Dataset, DatasetBase, Element, NaT, Name, Owned, Repr, Result,
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
    pub(crate) const fn numeric_only(self) -> bool {
        matches!(
            self,
            Self::Sum | Self::Prod | Self::Mean | Self::Median | Self::Std | Self::Var
        )
    }
}

/// Options for [`reduce`](DataArrayBase::reduce).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReduceOptions {
    /// Ignore NaNs (default: yes).
    pub skipna: Option<bool>,
    /// Delta degrees of freedom for `std` and `var`: the divisor is `n - ddof`.
    pub ddof: usize,
    /// For `sum` and `prod`: fewer valid values than this gives NaN (and a float result).
    pub min_count: Option<usize>,
    /// Keep the attributes of reduced variables (and of a dataset); the default, as xarray's
    /// `keep_attrs` (units are not attributes, so they never go stale).
    pub keep_attrs: bool,
}

impl Default for ReduceOptions {
    fn default() -> Self {
        Self {
            skipna: None,
            ddof: 0,
            min_count: None,
            keep_attrs: true,
        }
    }
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
red_ord!(String);

macro_rules! red_time {
    ($($t:ty),*) => {$(
        impl RedElem for $t {
            fn missing(&self) -> bool {
                self.is_nat()
            }
            fn order(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        }
    )*};
}
red_time!(Instant, Duration);

impl RedElem for CfDate {
    fn missing(&self) -> bool {
        self.is_nat()
    }
    fn order(&self, other: &Self) -> Ordering {
        self.nanos_since(other)
            .map_or(Ordering::Equal, |d| d.cmp(&0))
    }
}

/// Complex numbers have no order: reductions that need one are refused before they get here
/// (see `complex_stats`), so `order` is never called on them.
macro_rules! red_complex {
    ($($t:ty),*) => {$(
        impl RedElem for Complex<$t> {
            fn missing(&self) -> bool {
                self.re.is_nan() || self.im.is_nan()
            }
            fn order(&self, _: &Self) -> Ordering {
                Ordering::Equal
            }
        }
    )*};
}
red_complex!(f32, f64);

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

/// Lanes above which [`map_lanes_with`] splits the work across threads.
#[cfg(feature = "rayon")]
const PAR_ELEMENTS: usize = 1 << 16;

/// [`map_lanes`] with per-thread scratch state from `init`; lanes run in parallel with the
/// `rayon` feature (for large inputs). Results are in lane order either way.
fn map_lanes_with<T: Clone + Sync, O: Send, W>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    init: impl Fn() -> W + Sync + Send,
    f: impl Fn(&mut W, ArrayView1<'_, T>) -> Result<O> + Sync + Send,
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
    #[cfg(feature = "rayon")]
    let values = if a.len() > PAR_ELEMENTS && outer > 1 {
        use rayon::prelude::*;
        m.axis_iter(ndarray::Axis(0))
            .into_par_iter()
            .map_init(&init, |w, lane| f(w, lane))
            .collect::<Result<Vec<O>>>()?
    } else {
        let mut w = init();
        m.rows()
            .into_iter()
            .map(|lane| f(&mut w, lane))
            .collect::<Result<Vec<O>>>()?
    };
    #[cfg(not(feature = "rayon"))]
    let values = {
        let mut w = init();
        m.rows()
            .into_iter()
            .map(|lane| f(&mut w, lane))
            .collect::<Result<Vec<O>>>()?
    };
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
    if (T::DTYPE == DType::F32 || T::DTYPE == DType::F64)
        && let Some(r) = row_stats(a, axes, op, skipna, &ReduceOptions::default())
    {
        // Float values round-trip through f64 exactly.
        #[allow(clippy::cast_possible_truncation)]
        return Ok(match r? {
            v if T::DTYPE == DType::F32 => TypedArray::F32(v.mapv(|x| x as f32)),
            v => TypedArray::F64(v),
        });
    }
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
            .ok_or_else(|| CoreError::EmptyReduction(op.name()))
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
pub(crate) fn float_stat(
    op: Reduction,
    buf: &mut Vec<f64>,
    skipna: bool,
    opts: &ReduceOptions,
) -> f64 {
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
        // Folding from +0.0: `Sum` for floats starts at -0.0, but an empty sum is 0.0.
        Reduction::Sum => buf.iter().fold(0.0, |a, b| a + b),
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
    if let Some(out) = row_stats(a, axes, op, skipna, opts) {
        return out;
    }
    if matches!(
        op,
        Reduction::Sum | Reduction::Mean | Reduction::Var | Reduction::Std
    ) {
        // Streamed over each lane: no copy of the values.
        return map_lanes_with(
            a,
            axes,
            || (),
            |(), lane| lane_stat(op, &lane, skipna, opts).ok_or_else(unsupported),
        );
    }
    map_lanes_with(a, axes, Vec::new, |buf: &mut Vec<f64>, lane| {
        buf.clear();
        for v in &lane {
            buf.push(v.as_f64().ok_or_else(unsupported)?);
        }
        Ok(float_stat(op, buf, skipna, opts))
    })
}

/// [`float_stat`] for sums, means, variances and standard deviations, streamed over a lane
/// (twice for variances) instead of copied: the same values added in the same order, so the
/// same results. `None` for non-numeric values.
fn lane_stat<T: RedElem>(
    op: Reduction,
    lane: &ArrayView1<'_, T>,
    skipna: bool,
    opts: &ReduceOptions,
) -> Option<f64> {
    let mut n = 0usize;
    // `float_stat` folds sums from +0.0 and means through `Iterator::sum` (from -0.0).
    let mut sum = if op == Reduction::Sum { 0.0 } else { -0.0 };
    for v in lane {
        let x = v.as_f64()?;
        if x.is_nan() {
            if !skipna {
                return Some(f64::NAN);
            }
            continue;
        }
        sum += x;
        n = n.saturating_add(1);
    }
    if op == Reduction::Sum {
        return Some(if opts.min_count.is_some_and(|m| n < m) {
            f64::NAN
        } else {
            sum
        });
    }
    if n == 0 {
        return Some(f64::NAN);
    }
    let mean = sum / as_count(n);
    if op == Reduction::Mean {
        return Some(mean);
    }
    let Some(dof) = n.checked_sub(opts.ddof).filter(|d| *d > 0) else {
        return Some(f64::NAN);
    };
    let mut dev = -0.0;
    for v in lane {
        let x = v.as_f64()?;
        if !x.is_nan() {
            dev += (x - mean) * (x - mean);
        }
    }
    let var = dev / as_count(dof);
    Some(if op == Reduction::Std {
        var.sqrt()
    } else {
        var
    })
}

// ----- row accumulation -----

/// Values left after reducing below which lanes are used instead of rows.
const ROW_MIN_KEPT: usize = 256;

/// Values left per thread when accumulating rows in parallel.
#[cfg(feature = "rayon")]
const ROW_PER_TASK: usize = 4096;

/// The operations that rows (and streaming) handle.
fn row_op(op: Reduction) -> bool {
    matches!(
        op,
        Reduction::Sum
            | Reduction::Mean
            | Reduction::Var
            | Reduction::Std
            | Reduction::Min
            | Reduction::Max
    )
}

/// `a` with the kept axes first (in their own order, so rows stay contiguous for standard
/// layouts) and the reduced ones after; and the kept shape.
fn kept_first<'a, T>(a: &ArrayViewD<'a, T>, axes: &[usize]) -> (ArrayViewD<'a, T>, Vec<usize>) {
    let keep: Vec<usize> = (0..a.ndim()).filter(|i| !axes.contains(i)).collect();
    let order: Vec<usize> = keep.iter().chain(axes).copied().collect();
    let kshape = keep
        .iter()
        .filter_map(|k| a.shape().get(*k).copied())
        .collect();
    (a.clone().permuted_axes(IxDyn(&order)), kshape)
}

/// Reductions over leading or middle dimensions, by accumulating whole rows (sub-arrays of
/// the kept shape) into the result, as numpy does: no transposed copy, simple loops over
/// memory, large results split across threads. Values are added in the same order as along
/// a lane, so results equal those of [`float_stat`] exactly. `None` where lanes suit better
/// (reducing trailing dimensions, few values left) or for other operations.
fn row_stats<T: RedElem + Sync>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    op: Reduction,
    skipna: bool,
    opts: &ReduceOptions,
) -> Option<Result<ArrayD<f64>>> {
    let shape = a.shape();
    let keep: Vec<usize> = (0..a.ndim()).filter(|i| !axes.contains(i)).collect();
    let trailing = keep.iter().enumerate().all(|(i, k)| i == *k);
    let kept: usize = keep.iter().filter_map(|k| shape.get(*k)).product();
    let reduced: usize = axes.iter().filter_map(|k| shape.get(*k)).product();
    if !row_op(op) || keep.is_empty() || trailing || kept < ROW_MIN_KEPT || reduced == 0 {
        return None;
    }
    let (p, kshape) = kept_first(a, axes);
    let mut state = RowState::new(op, skipna, opts, &kshape);
    state.feed(&p);
    if state.two_pass() {
        state.start_second_pass();
        state.feed(&p);
    }
    Some(Ok(state.finish()))
}

/// Accumulators of a row reduction, one per kept position: fed rows (sub-arrays of the kept
/// shape) in order, possibly over several calls (slabs of a larger array), then finished.
struct RowState {
    op: Reduction,
    skipna: bool,
    ddof: usize,
    min_count: Option<usize>,
    /// (sum, count) for sums, means and the first pass of variances.
    sums: ArrayD<(f64, usize)>,
    /// (best so far, NaN until a value is seen; whether a NaN was seen) for min and max.
    best: ArrayD<(f64, bool)>,
    /// (sum of squared deviations, mean) for the second pass of variances.
    dev: ArrayD<(f64, f64)>,
    second: bool,
    /// Variances streamed in one pass: (count, mean, sum of squared deviations) merged slab
    /// by slab (see [`feed_slab`](Self::feed_slab)).
    moments: ArrayD<(usize, f64, f64)>,
    merged: bool,
    kshape: Vec<usize>,
}

impl RowState {
    fn new(op: Reduction, skipna: bool, opts: &ReduceOptions, kshape: &[usize]) -> Self {
        let sh = IxDyn(kshape);
        let minmax = matches!(op, Reduction::Min | Reduction::Max);
        // Sums fold from +0.0 (an empty sum is 0); means from -0.0 like `Iterator::sum`.
        let zero = if op == Reduction::Sum { 0.0 } else { -0.0 };
        let empty = IxDyn(&vec![0; kshape.len()]);
        Self {
            op,
            skipna,
            ddof: opts.ddof,
            min_count: opts.min_count,
            sums: if minmax {
                ArrayD::from_elem(empty.clone(), (0.0, 0))
            } else {
                ArrayD::from_elem(sh.clone(), (zero, 0))
            },
            best: if minmax {
                ArrayD::from_elem(sh, (f64::NAN, false))
            } else {
                ArrayD::from_elem(empty.clone(), (f64::NAN, false))
            },
            dev: ArrayD::from_elem(empty.clone(), (0.0, 0.0)),
            second: false,
            moments: ArrayD::from_elem(empty, (0, 0.0, 0.0)),
            merged: false,
            kshape: kshape.to_vec(),
        }
    }

    /// Feed one slab of a larger array (kept axes first). Sums, means, minima and maxima
    /// accumulate exactly as [`feed`](Self::feed) does. Variances get the slab's exact
    /// two-pass moments, merged into the running ones with the pairwise update of Chan,
    /// Golub and LeVeque: one read of the data instead of two, at the cost of rounding a
    /// little differently from the in-memory (two-pass) result.
    fn feed_slab<T: RedElem + Sync>(&mut self, p: &ArrayViewD<'_, T>) {
        if !self.two_pass() {
            self.feed(p);
            return;
        }
        let opts = ReduceOptions::default().ddof(self.ddof);
        let mut local = RowState::new(self.op, self.skipna, &opts, &self.kshape);
        local.feed(p);
        local.start_second_pass();
        local.feed(p);
        if !self.merged {
            self.moments = ArrayD::from_elem(IxDyn(&self.kshape), (0, 0.0, 0.0));
            self.merged = true;
        }
        ndarray::Zip::from(&mut self.moments)
            .and(&local.sums)
            .and(&local.dev)
            .for_each(|(n, m, m2), (_, nb), (d, mb)| {
                if *nb == 0 {
                    return;
                }
                if *n == 0 {
                    (*n, *m, *m2) = (*nb, *mb, *d);
                    return;
                }
                let total = n.saturating_add(*nb);
                let (na, nbf, nt) = (as_count(*n), as_count(*nb), as_count(total));
                let delta = mb - *m;
                *m += delta * nbf / nt;
                *m2 += d + delta * delta * na * nbf / nt;
                *n = total;
            });
    }

    fn two_pass(&self) -> bool {
        matches!(self.op, Reduction::Var | Reduction::Std)
    }

    fn start_second_pass(&mut self) {
        self.dev = self.sums.mapv(|(s, c)| (-0.0, mean_of(s, c)));
        self.second = true;
    }

    /// Feed every row of `p` (kept axes first, then reduced ones), in row-major order of
    /// the reduced positions.
    fn feed<T: RedElem + Sync>(&mut self, p: &ArrayViewD<'_, T>) {
        let val = |x: &T| x.as_f64().unwrap_or(f64::NAN);
        let skipna = self.skipna;
        if self.second {
            accumulate(p, &mut self.dev, |(d, m): &mut (f64, f64), x| {
                let x = val(x);
                if !(skipna && x.is_nan()) {
                    *d += (x - *m) * (x - *m);
                }
            });
        } else if matches!(self.op, Reduction::Min | Reduction::Max) {
            let want = if self.op == Reduction::Min {
                Ordering::Less
            } else {
                Ordering::Greater
            };
            accumulate(p, &mut self.best, |(b, m): &mut (f64, bool), x| {
                let x = val(x);
                if x.is_nan() {
                    *m = true;
                } else if b.is_nan() || x.total_cmp(b) == want {
                    *b = x;
                }
            });
        } else {
            accumulate(p, &mut self.sums, |(s, c): &mut (f64, usize), x| {
                let x = val(x);
                if !(skipna && x.is_nan()) {
                    *s += x;
                    *c = c.saturating_add(1);
                }
            });
        }
    }

    fn finish(self) -> ArrayD<f64> {
        let skipna = self.skipna;
        if self.merged {
            let (ddof, std) = (self.ddof, self.op == Reduction::Std);
            return self
                .moments
                .mapv(|(n, _, m2)| match n.checked_sub(ddof).filter(|x| *x > 0) {
                    Some(dof) => {
                        let var = m2 / as_count(dof);
                        if std { var.sqrt() } else { var }
                    }
                    None => f64::NAN,
                });
        }
        match self.op {
            Reduction::Min | Reduction::Max => self
                .best
                .mapv(|(b, m)| if m && !skipna { f64::NAN } else { b }),
            Reduction::Sum => {
                let min = self.min_count.unwrap_or(0);
                self.sums.mapv(|(s, c)| if c < min { f64::NAN } else { s })
            }
            Reduction::Mean => self.sums.mapv(|(s, c)| mean_of(s, c)),
            op => {
                let std = op == Reduction::Std;
                let ddof = self.ddof;
                ndarray::Zip::from(&self.dev)
                    .and(&self.sums)
                    .map_collect(
                        |(v, _), (_, c)| match c.checked_sub(ddof).filter(|x| *x > 0) {
                            Some(dof) => {
                                let var = v / as_count(dof);
                                if std { var.sqrt() } else { var }
                            }
                            None => f64::NAN,
                        },
                    )
            }
        }
    }
}

fn mean_of(s: f64, c: usize) -> f64 {
    if c > 0 { s / as_count(c) } else { f64::NAN }
}

/// `f(acc, value)` for every row of `p` and every kept position, rows in row-major order of
/// the reduced positions; split across threads along a kept axis for large results.
fn accumulate<T: RedElem + Sync, A: Send + Sync>(
    p: &ArrayViewD<'_, T>,
    acc: &mut ArrayD<A>,
    f: impl Fn(&mut A, &T) + Sync + Send + Copy,
) {
    let nk = acc.ndim();
    #[cfg(feature = "rayon")]
    {
        let kept = acc.len();
        let tasks = (kept / ROW_PER_TASK).clamp(1, rayon::current_num_threads().max(1));
        if tasks > 1 {
            use rayon::prelude::*;
            // Split the first kept axis if it is long enough, else the longest one.
            let kshape = acc.shape().to_vec();
            let first = kshape.first().copied().unwrap_or(0);
            let ax = if first >= tasks {
                0
            } else {
                (0..nk)
                    .max_by_key(|i| (kshape.get(*i).copied().unwrap_or(0), std::cmp::Reverse(*i)))
                    .unwrap_or(0)
            };
            let per = kshape.get(ax).copied().unwrap_or(1).div_ceil(tasks).max(1);
            p.axis_chunks_iter(Axis(ax), per)
                .into_par_iter()
                .zip(acc.axis_chunks_iter_mut(Axis(ax), per).into_par_iter())
                .for_each(|(inp, mut a)| rows_into(&inp, &mut a, nk, f));
            return;
        }
    }
    rows_into(p, &mut acc.view_mut(), nk, f);
}

/// The sequential part of [`accumulate`]: contiguous rows and accumulators are walked as
/// slices, others element by element in place.
fn rows_into<T, A>(
    inp: &ArrayViewD<'_, T>,
    acc: &mut ArrayViewMutD<'_, A>,
    nk: usize,
    mut f: impl FnMut(&mut A, &T),
) {
    if nk == 0 {
        // Everything reduced: one accumulator, fed the values in row-major order.
        if let Some(a) = acc.iter_mut().next() {
            for x in inp {
                f(a, x);
            }
        }
        return;
    }
    let rshape: Vec<usize> = inp.shape().iter().skip(nk).copied().collect();
    let mut go = |v: ArrayViewD<'_, T>| {
        if let (Some(s), Some(a)) = (v.as_slice(), acc.as_slice_mut()) {
            for (a, x) in a.iter_mut().zip(s) {
                f(a, x);
            }
        } else {
            ndarray::Zip::from(&mut *acc).and(&v).for_each(&mut f);
        }
    };
    if let [_] = rshape.as_slice() {
        for v in inp.axis_iter(Axis(nk)) {
            go(v);
        }
        return;
    }
    for idx in ndarray::indices(IxDyn(&rshape)) {
        let mut v = inp.view();
        for (k, i) in idx.slice().iter().enumerate().rev() {
            v = v.index_axis_move(Axis(nk.saturating_add(k)), *i);
        }
        go(v);
    }
}

// ----- streaming lazy data -----

/// Bytes of lazy data a streamed reduction reads at a time.
static STREAM_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(64 << 20);

/// Set the streaming budget (default 64 MiB): how many bytes of lazy (on-disk) data a
/// reduction reads at a time, and how many bytes of chunks the HDF5 reader decodes and the
/// writer compresses at once. Lower it to use less memory; raise it for more parallelism on
/// small chunks.
pub fn set_stream_bytes(bytes: usize) {
    STREAM_BYTES.store(bytes.max(1), std::sync::atomic::Ordering::Relaxed);
    annoray_hdf5::set_batch_bytes(bytes);
}

/// The current streaming budget, in bytes.
#[must_use]
pub fn stream_bytes() -> usize {
    STREAM_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Reduce lazy float data slab by slab along its first axis, within the streaming budget,
/// reading each slab once. Reducing that axis feeds the slabs to row accumulators; keeping
/// it reduces each slab on its own. Sums, means, minima and maxima add values in the
/// in-memory order, so they equal the in-memory results; variances merge per-slab moments
/// (see [`RowState::feed_slab`]) and agree to rounding. `None` for in-memory data and for
/// reductions this does not handle.
fn streamed<S: Repr>(
    v: &VariableBase<S>,
    op: Reduction,
    axes: &[usize],
    opts: &ReduceOptions,
) -> Option<Result<TypedArray<Owned>>> {
    let dtype = v.dtype();
    let shape = v.shape().to_vec();
    if !v.is_lazy()
        || !row_op(op)
        || !matches!(dtype, DType::F32 | DType::F64)
        || shape.is_empty()
        || axes.is_empty()
    {
        return None;
    }
    let esize = if dtype == DType::F32 { 4 } else { 8 };
    let row_bytes = shape
        .iter()
        .skip(1)
        .try_fold(esize, |a: usize, n| a.checked_mul(*n))?;
    let n0 = shape.first().copied().unwrap_or(0);
    let per = stream_bytes()
        .checked_div(row_bytes)
        .unwrap_or(1)
        .clamp(1, n0.max(1));
    Some(stream_slabs(v, op, axes, opts, &shape, n0, per))
}

fn stream_slabs<S: Repr>(
    v: &VariableBase<S>,
    op: Reduction,
    axes: &[usize],
    opts: &ReduceOptions,
    shape: &[usize],
    n0: usize,
    per: usize,
) -> Result<TypedArray<Owned>> {
    let src = v.as_lazy_source()?;
    let slab = |(s0, len): (usize, usize)| -> Result<TypedArray<Owned>> {
        let mut start = vec![0; shape.len()];
        let mut count = shape.to_vec();
        if let (Some(s), Some(c)) = (start.first_mut(), count.first_mut()) {
            *s = s0;
            *c = len;
        }
        src.read_slice(&start, &count)
    };
    let slabs = plan_slabs(n0, per, src.chunking(0));
    let f32_out = v.dtype() == DType::F32;
    #[allow(clippy::cast_possible_truncation)] // f32 data stays f32, as in memory
    let wrap = |a: ArrayD<f64>| {
        if f32_out {
            TypedArray::F32(a.mapv(|x| x as f32))
        } else {
            TypedArray::F64(a)
        }
    };
    if !axes.contains(&0) {
        // The first axis is kept: each slab gives its part of the result.
        let mut parts = Vec::with_capacity(slabs.len());
        for s in &slabs {
            parts.push(reduce_typed(&slab(*s)?.view(), op, axes, opts)?);
        }
        let views: Vec<TypedArray<View<'_>>> = parts.iter().map(TypedArray::view).collect();
        return crate::array::typed_dispatch!(views.first().ok_or_else(|| CoreError::Read("no slabs".into()))?,
            x => concat_axis0(&views, crate::array::marker(x)))
        .ok_or_else(|| CoreError::Read("joining slabs".into()));
    }
    let skipna = opts.skipna.unwrap_or(true);
    let mut state: Option<RowState> = None;
    let feed = |state: &mut Option<RowState>, data: &TypedArray<Owned>| -> Result<()> {
        macro_rules! go {
            ($x:expr) => {{
                let (p, kshape) = kept_first(&$x.view(), axes);
                state
                    .get_or_insert_with(|| RowState::new(op, skipna, opts, &kshape))
                    .feed_slab(&p);
            }};
        }
        match data {
            TypedArray::F32(x) => go!(x),
            TypedArray::F64(x) => go!(x),
            other => {
                return Err(CoreError::UnsupportedDType {
                    op: op.name(),
                    dtype: other.dtype(),
                });
            }
        }
        Ok(())
    };
    for s in &slabs {
        feed(&mut state, &slab(*s)?)?;
    }
    let state = state.ok_or_else(|| CoreError::EmptyReduction(op.name()))?;
    Ok(wrap(state.finish()))
}

/// Group reductions of lazy float data along its first axis in one streamed pass: each slab
/// is read once and its rows go to their group's accumulators. (Reducing group by group
/// would read every chunk once per group.) Rows reach each group in increasing position,
/// as when a group is reduced in memory, so results are the same (variances agree to
/// rounding: see [`RowState::feed_slab`]). One entry per group, in
/// order: an error for an empty group where its reduction is undefined, as in memory.
/// `None` where this does not apply.
#[allow(clippy::type_complexity)]
pub(crate) fn stream_groups<S: Repr>(
    v: &VariableBase<S>,
    dim: &Name,
    groups: &[Vec<usize>],
    op: Reduction,
    opts: &ReduceOptions,
) -> Option<Result<Vec<Result<Variable>>>> {
    let dtype = v.dtype();
    let shape = v.shape().to_vec();
    let sorted = groups
        .iter()
        .all(|g| g.windows(2).all(|w| w.first() < w.get(1)));
    if !v.is_lazy()
        || !row_op(op)
        || !matches!(dtype, DType::F32 | DType::F64)
        || v.dims().first() != Some(dim)
        || !sorted
    {
        return None;
    }
    Some(stream_groups_impl(v, groups, op, opts, &shape))
}

fn stream_groups_impl<S: Repr>(
    v: &VariableBase<S>,
    groups: &[Vec<usize>],
    op: Reduction,
    opts: &ReduceOptions,
    shape: &[usize],
) -> Result<Vec<Result<Variable>>> {
    let src = v.as_lazy_source()?;
    let esize = if v.dtype() == DType::F32 { 4 } else { 8 };
    let row_bytes = shape
        .iter()
        .skip(1)
        .try_fold(esize, |a: usize, n| a.checked_mul(*n))
        .ok_or(CoreError::ArithmeticOverflow("reduction size"))?;
    let n0 = shape.first().copied().unwrap_or(0);
    let per = stream_bytes()
        .checked_div(row_bytes)
        .unwrap_or(1)
        .clamp(1, n0.max(1));
    let slabs = plan_slabs(n0, per, src.chunking(0));
    let kshape: Vec<usize> = shape.iter().skip(1).copied().collect();
    let skipna = opts.skipna.unwrap_or(true);
    let mut states: Vec<RowState> = groups
        .iter()
        .map(|_| RowState::new(op, skipna, opts, &kshape))
        .collect();
    let pass = |states: &mut Vec<RowState>| -> Result<()> {
        // Per group: how far into its positions the slabs have got.
        let mut cursor = vec![0usize; groups.len()];
        for (s0, len) in &slabs {
            let end = s0.saturating_add(*len);
            let data = {
                let mut start = vec![0; shape.len()];
                let mut count = shape.to_vec();
                if let (Some(s), Some(c)) = (start.first_mut(), count.first_mut()) {
                    *s = *s0;
                    *c = *len;
                }
                src.read_slice(&start, &count)?
            };
            for ((g, cur), state) in groups.iter().zip(&mut cursor).zip(states.iter_mut()) {
                let from = *cur;
                while g.get(*cur).is_some_and(|i| *i < end) {
                    *cur = cur.saturating_add(1);
                }
                let rows: Vec<usize> = g
                    .get(from..*cur)
                    .unwrap_or_default()
                    .iter()
                    .map(|i| i.saturating_sub(*s0))
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                let part = data.view().select_axis(0, &rows);
                match &part {
                    TypedArray::F32(x) => state.feed_slab(&kept_first(&x.view(), &[0]).0),
                    TypedArray::F64(x) => state.feed_slab(&kept_first(&x.view(), &[0]).0),
                    other => {
                        return Err(CoreError::UnsupportedDType {
                            op: op.name(),
                            dtype: other.dtype(),
                        });
                    }
                }
            }
        }
        Ok(())
    };
    pass(&mut states)?;
    let dims: Vec<Name> = v.dims().iter().skip(1).cloned().collect();
    let f32_out = v.dtype() == DType::F32;
    groups
        .iter()
        .zip(states)
        .map(|(g, st)| {
            if g.is_empty() && matches!(op, Reduction::Min | Reduction::Max) {
                return Ok(Err(CoreError::EmptyReduction(op.name())));
            }
            let r = st.finish();
            #[allow(clippy::cast_possible_truncation)] // f32 data stays f32, as in memory
            let data = if f32_out {
                TypedArray::F32(r.mapv(|x| x as f32))
            } else {
                TypedArray::F64(r)
            };
            let mut out = Variable::from_typed(dims.clone(), data)?;
            out.set_unit(reduced_unit(v.unit(), op, g.len())?);
            if opts.keep_attrs {
                out.attrs_mut()
                    .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
            }
            Ok(Ok(out))
        })
        .collect()
}

/// Slabs `(start, len)` covering `0..n` about `per` long. With the data's chunking along the
/// axis, slabs end on chunk boundaries (a slab grows to the next boundary rather than split
/// a chunk), so each chunk is read and decoded once.
fn plan_slabs(n: usize, per: usize, chunking: Option<(usize, usize)>) -> Vec<(usize, usize)> {
    let per = per.max(1);
    let mut out = Vec::new();
    let mut s = 0usize;
    while s < n {
        let mut end = s.saturating_add(per).min(n);
        if let Some((size, offset)) = chunking.filter(|(size, _)| *size > 0)
            && end < n
        {
            // The last boundary at or before `end` (after `s`), else the next one after `s`.
            let back = end
                .checked_add(size)
                .and_then(|e| e.checked_sub(offset))
                .and_then(|e| e.checked_rem(size))
                .unwrap_or(0);
            let down = end.saturating_sub(back);
            end = if down > s {
                down
            } else {
                down.saturating_add(size).min(n)
            };
        }
        out.push((s, end.saturating_sub(s)));
        s = end;
    }
    out
}

fn concat_axis0<T: crate::Element>(
    parts: &[TypedArray<View<'_>>],
    _: crate::array::Marker<T>,
) -> Option<TypedArray<Owned>> {
    let views = parts
        .iter()
        .map(|p| T::peek(p).map(|a| a.view()))
        .collect::<Option<Vec<_>>>()?;
    ndarray::concatenate(Axis(0), &views).ok().map(T::wrap)
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
fn time_stats<T: RedElem + NaT>(
    a: &ArrayViewD<'_, T>,
    axes: &[usize],
    op: Reduction,
    skipna: bool,
    key: impl Fn(&T) -> Option<i128>,
    back: impl Fn(i128) -> Option<T>,
) -> Result<ArrayD<T>> {
    let overflow = || CoreError::ArithmeticOverflow(op.name());
    let mut keys: Vec<i128> = Vec::new();
    map_lanes(a, axes, |lane| {
        keys.clear();
        for v in &lane {
            if v.is_nat() {
                if !skipna {
                    return Ok(T::NAT);
                }
                continue;
            }
            keys.push(key(v).ok_or_else(overflow)?);
        }
        let n = i128::try_from(keys.len()).map_err(|_| overflow())?;
        if n == 0 {
            // Like NaN for floats: an empty sum is zero, any other statistic is missing.
            return match op {
                Reduction::Sum => back(0).ok_or_else(overflow),
                _ => Ok(T::NAT),
            };
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
            skipna,
            |v| (!v.is_nat()).then(|| i128::from(v.unix_nanos())),
            |ns| {
                i64::try_from(ns)
                    .ok()
                    .map(Instant::from_unix_nanos)
                    .filter(|t| !t.is_nat())
            },
        )?),
        TypedArray::TimeDelta(x) => {
            TypedArray::TimeDelta(time_stats(x, axes, op, skipna, td_ns, td_from_ns)?)
        }
        TypedArray::CfDate(x) => {
            // Offsets from the first real date (with none, every lane is NaT).
            let base = x
                .iter()
                .find(|d| !d.is_nat())
                .copied()
                .unwrap_or(CfDate::NAT);
            TypedArray::CfDate(time_stats(
                x,
                axes,
                op,
                skipna,
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

/// A complex part type: `f32` or `f64`, accumulated in `f64`.
trait ComplexPart: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}

impl ComplexPart for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[allow(clippy::cast_possible_truncation)] // complex64 data stays complex64, as in numpy
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl ComplexPart for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(x: f64) -> Self {
        x
    }
}

/// Reductions of complex numbers (see the module documentation).
fn complex_stats<P: ComplexPart>(
    a: &ArrayViewD<'_, Complex<P>>,
    op: Reduction,
    axes: &[usize],
    skipna: bool,
    opts: &ReduceOptions,
) -> Result<TypedArray<Owned>>
where
    Complex<P>: RedElem,
{
    let nan = Complex::new(f64::NAN, f64::NAN);
    let lane_values = |lane: &ArrayView1<'_, Complex<P>>| -> Option<Vec<Complex<f64>>> {
        let mut values = Vec::with_capacity(lane.len());
        for v in lane {
            if v.missing() {
                if !skipna {
                    return None;
                }
                continue;
            }
            values.push(Complex::new(v.re.to_f64(), v.im.to_f64()));
        }
        Some(values)
    };
    let narrow = |out: ArrayD<Complex<f64>>| -> TypedArray<Owned> {
        Complex::<P>::wrap(out.mapv(|z| Complex::new(P::from_f64(z.re), P::from_f64(z.im))))
    };
    match op {
        Reduction::Count => count(a, axes),
        Reduction::Sum | Reduction::Prod | Reduction::Mean => {
            let out = map_lanes(a, axes, |lane| {
                let Some(values) = lane_values(&lane) else {
                    return Ok(nan);
                };
                let n = values.len();
                if op != Reduction::Mean && opts.min_count.is_some_and(|m| n < m) {
                    return Ok(nan);
                }
                Ok(match op {
                    Reduction::Prod => values.iter().fold(Complex::new(1.0, 0.0), |a, b| a * b),
                    Reduction::Mean if n == 0 => nan,
                    Reduction::Mean => {
                        values.iter().fold(Complex::new(0.0, 0.0), |a, b| a + b) / as_count(n)
                    }
                    _ => values.iter().fold(Complex::new(0.0, 0.0), |a, b| a + b),
                })
            })?;
            Ok(narrow(out))
        }
        Reduction::Var | Reduction::Std => {
            let out = map_lanes(a, axes, |lane| {
                let Some(values) = lane_values(&lane) else {
                    return Ok(f64::NAN);
                };
                let n = values.len();
                let Some(dof) = n.checked_sub(opts.ddof).filter(|d| *d > 0) else {
                    return Ok(f64::NAN);
                };
                let m = values.iter().fold(Complex::new(0.0, 0.0), |a, b| a + b) / as_count(n);
                let var = values.iter().map(|v| (v - m).norm_sqr()).sum::<f64>() / as_count(dof);
                Ok(if op == Reduction::Std {
                    var.sqrt()
                } else {
                    var
                })
            })?;
            Ok(if Complex::<P>::DTYPE == DType::C64 {
                TypedArray::F32(out.mapv(f32_of))
            } else {
                TypedArray::F64(out)
            })
        }
        _ => Err(CoreError::UnsupportedDType {
            op: op.name(),
            dtype: Complex::<P>::DTYPE,
        }),
    }
}

#[allow(clippy::cast_possible_truncation)] // the variance of complex64 data is float32
fn f32_of(v: f64) -> f32 {
    v as f32
}

/// Reduce raw data over `axes`.
fn reduce_typed(
    a: &TypedArray<View<'_>>,
    op: Reduction,
    axes: &[usize],
    opts: &ReduceOptions,
) -> Result<TypedArray<Owned>> {
    let skipna = opts.skipna.unwrap_or(true);
    match a {
        TypedArray::C64(x) => return complex_stats(x, op, axes, skipna, opts),
        TypedArray::C128(x) => return complex_stats(x, op, axes, skipna, opts),
        _ => {}
    }
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
pub(crate) fn reduced_unit(unit: Option<&Unit>, op: Reduction, n: usize) -> Result<Option<Unit>> {
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
    let data = match streamed(v, op, &axes, opts) {
        Some(d) => d?,
        None => reduce_typed(&v.typed()?, op, &axes, opts)?,
    };
    let out_dims: Vec<Name> = v
        .dims()
        .iter()
        .filter(|d| !dims.contains(d))
        .cloned()
        .collect();
    let mut out = Variable::from_typed(out_dims, data)?;
    out.set_unit(reduced_unit(v.unit(), op, n)?);
    out.take_zone_from(v);
    if opts.keep_attrs {
        out.attrs_mut()
            .extend(v.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    }
    Ok(out)
}

pub(crate) fn is_numeric(d: DType) -> bool {
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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing
)]
mod slab_tests {
    use super::plan_slabs;

    fn covers(slabs: &[(usize, usize)], n: usize) -> bool {
        let mut next = 0;
        for (s, len) in slabs {
            if *s != next || *len == 0 {
                return false;
            }
            next = s + len;
        }
        next == n
    }

    #[test]
    fn slabs_end_on_chunk_boundaries() {
        // No chunking: plain steps.
        assert_eq!(plan_slabs(10, 4, None), vec![(0, 4), (4, 4), (8, 2)]);
        // Chunks of 10 from 0: a 32-row budget gives 30-row slabs.
        let s = plan_slabs(100, 32, Some((10, 0)));
        assert_eq!(s, vec![(0, 30), (30, 30), (60, 30), (90, 10)]);
        // Smaller budget than a chunk: one chunk per slab.
        assert_eq!(
            plan_slabs(25, 3, Some((10, 0))),
            vec![(0, 10), (10, 10), (20, 5)]
        );
        // Boundaries at 3, 13, 23, … (a selection starting 7 rows into a chunk).
        let s = plan_slabs(40, 12, Some((10, 3)));
        assert_eq!(s, vec![(0, 3), (3, 10), (13, 10), (23, 10), (33, 7)]);
        for (n, per, c) in [
            (1000, 37, Some((16, 5))),
            (7, 100, Some((3, 1))),
            (50, 1, None),
        ] {
            let s = plan_slabs(n, per, c);
            assert!(covers(&s, n), "{n} {per} {c:?}: {s:?}");
            if let Some((size, off)) = c {
                for (st, len) in &s[..s.len() - 1] {
                    assert_eq!((st + len + size - off) % size, 0, "{s:?}");
                }
            }
        }
    }
}
