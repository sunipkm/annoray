//! Moving windows (xarray's `rolling`).
//!
//! `da.rolling("time", 7, &RollingOptions::default())?.mean()?` gives, at every position, the
//! mean over the window ending there (or centred on it, with `center`; an even window of `w`
//! then covers `i - w/2 ..= i + w/2 - 1`, as in xarray). Positions whose window has fewer than
//! `min_periods` non-NaN values (by default, the whole window) are NaN. The result keeps the
//! input's shape and coordinates; integers become `f64`, `f32` stays `f32`.
//!
//! Reductions work one window at a time, so memory stays that of the input and output: sums,
//! means, counts and the spread (`var`, `std`) slide (one value in, one out per position,
//! Welford's updates for the spread), in parallel over blocks of lanes and in the data's
//! own type; the others reduce each window.
//! [`construct`](DataArrayRolling::construct) exposes the windows themselves as a new
//! trailing dimension, NaN-padded at the edges (which does take window-size times the memory).

use ndarray::{ArrayD, ArrayView1, ArrayViewD, ArrayViewMut1, Axis, IxDyn, Slice};

use crate::dataarray::DataArrayBase;
use crate::ops::to_f64;
use crate::{
    CoreError, DType, DataArray, Dataset, DatasetBase, Name, ReduceOptions, Reduction, Repr,
    Result, TypedArray, Variable, VariableBase,
};

/// Options for `rolling`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct RollingOptions {
    /// Centre the window on each position instead of ending it there.
    pub center: bool,
    /// The fewest non-NaN values that give a result (default: the window size).
    pub min_periods: Option<usize>,
}

impl RollingOptions {
    /// Centre the window.
    #[must_use]
    pub const fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// The fewest non-NaN values that give a result.
    #[must_use]
    pub const fn min_periods(mut self, n: usize) -> Self {
        self.min_periods = Some(n);
        self
    }
}

fn invalid(reason: impl Into<String>) -> CoreError {
    CoreError::InvalidReduction(reason.into())
}

/// A float element with a NaN to pad windows with.
trait Fl: Copy + PartialEq {
    const NAN: Self;
}

impl Fl for f32 {
    const NAN: Self = f32::NAN;
}

impl Fl for f64 {
    const NAN: Self = f64::NAN;
}

/// `a` with a trailing window axis: `out[.., i, j] = a[.., i - back + j]`, NaN outside.
fn windows<F: Fl>(a: &ArrayViewD<'_, F>, axis: usize, w: usize, back: usize) -> Result<ArrayD<F>> {
    let overflow = || CoreError::ArithmeticOverflow("rolling");
    let n = a.len_of(Axis(axis));
    let mut shape = a.shape().to_vec();
    shape.push(w);
    let last = a.ndim();
    let mut out = ArrayD::from_elem(IxDyn(&shape), F::NAN);
    let n = isize::try_from(n).map_err(|_| overflow())?;
    let back = isize::try_from(back).map_err(|_| overflow())?;
    for j in 0..w {
        let shift = isize::try_from(j)
            .ok()
            .and_then(|j| j.checked_sub(back))
            .ok_or_else(overflow)?;
        // Destination positions i with 0 <= i + shift < n.
        let lo = shift.checked_neg().ok_or_else(overflow)?.max(0);
        let hi = n.checked_sub(shift).ok_or_else(overflow)?.min(n);
        if lo >= hi {
            continue;
        }
        let (slo, shi) = (
            lo.checked_add(shift).ok_or_else(overflow)?,
            hi.checked_add(shift).ok_or_else(overflow)?,
        );
        let mut dst = out.index_axis_mut(Axis(last), j);
        dst.slice_axis_mut(Axis(axis), Slice::from(lo..hi))
            .assign(&a.slice_axis(Axis(axis), Slice::from(slo..shi)));
    }
    Ok(out)
}

/// A data array with a moving window along one dimension. Made by
/// [`DataArrayBase::rolling`].
pub struct DataArrayRolling<'a, S: Repr> {
    obj: &'a DataArrayBase<S>,
    dim: Name,
    window: usize,
    opts: RollingOptions,
}

/// The window positions before (and including) each position: `back` of them come before it.
fn back(window: usize, center: bool) -> usize {
    if center {
        window / 2
    } else {
        window.saturating_sub(1)
    }
}

/// The windowed data (as floats) of a variable.
fn windowed<S: Repr>(
    v: &VariableBase<S>,
    dim: &Name,
    window: usize,
    center: bool,
    window_dim: &Name,
) -> Result<Variable> {
    let axis = v.axis(dim)?;
    let b = back(window, center);
    let data = match v.typed()? {
        TypedArray::F32(a) => TypedArray::F32(windows(&a, axis, window, b)?),
        t if !matches!(
            t.dtype(),
            DType::Str | DType::Bool | DType::DateTime | DType::CfDate | DType::TimeDelta
        ) =>
        {
            TypedArray::F64(windows(&to_f64(&t, None)?.view(), axis, window, b)?)
        }
        t => {
            return Err(CoreError::UnsupportedDType {
                op: "rolling",
                dtype: t.dtype(),
            });
        }
    };
    let mut dims: Vec<Name> = v.dims().to_vec();
    dims.push(window_dim.clone());
    let mut out = Variable::from_typed(dims, data)?;
    out.set_unit(v.unit().cloned());
    Ok(out)
}

/// One lane of a rolling reduction: each window is a slice of the lane (no copying of the
/// lane, no per-window seeking).
fn roll_lane(
    lane: ArrayView1<'_, f64>,
    mut dst: ArrayViewMut1<'_, f64>,
    window: usize,
    back: usize,
    min_periods: usize,
    op: Reduction,
) {
    let n = lane.len();
    let ropts = ReduceOptions::default().skipna(true);
    let mut buf: Vec<f64> = Vec::with_capacity(window);
    for (i, slot) in dst.iter_mut().enumerate() {
        // The window covers padded positions `i - back ..`, of which `lo..hi` exist.
        let lo = i.saturating_sub(back);
        let hi = i.saturating_add(window).saturating_sub(back).min(n);
        let pad = back.saturating_sub(i);
        buf.clear();
        buf.extend(lane.slice(ndarray::s![lo..hi.max(lo)]).iter().copied());
        let valid = buf.iter().filter(|x| !x.is_nan()).count();
        if valid < min_periods || valid == 0 {
            continue;
        }
        *slot = match op {
            #[allow(clippy::cast_precision_loss)] // counts within a window
            Reduction::Count => valid as f64,
            Reduction::Min => buf
                .iter()
                .copied()
                .filter(|x| !x.is_nan())
                .fold(f64::INFINITY, f64::min),
            Reduction::Max => buf
                .iter()
                .copied()
                .filter(|x| !x.is_nan())
                .fold(f64::NEG_INFINITY, f64::max),
            Reduction::ArgMin | Reduction::ArgMax => {
                let want_max = op == Reduction::ArgMax;
                let mut best: Option<(usize, f64)> = None;
                for (j, x) in buf.iter().enumerate() {
                    if x.is_nan() {
                        continue;
                    }
                    let better = best.is_none_or(|(_, b)| if want_max { *x > b } else { *x < b });
                    if better {
                        best = Some((j, *x));
                    }
                }
                // Positions count from the window's (padded) start.
                #[allow(clippy::cast_precision_loss)] // positions within a window
                best.map_or(f64::NAN, |(j, _)| j.saturating_add(pad) as f64)
            }
            _ => crate::reduce::float_stat(op, &mut buf, true, &ropts),
        };
    }
}

/// Columns per parallel task of the sliding kernel.
const BLOCK: usize = 512;

/// A sliding statistic of [`running`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Running {
    Sum,
    Mean,
    Count,
    Var,
    Std,
}

impl Running {
    fn of(op: Reduction) -> Option<Self> {
        Some(match op {
            Reduction::Sum => Self::Sum,
            Reduction::Mean => Self::Mean,
            Reduction::Count => Self::Count,
            Reduction::Var => Self::Var,
            Reduction::Std => Self::Std,
            _ => return None,
        })
    }
}

/// One block of columns (every row) of [`running`]: each window is the previous one with one
/// value added and one removed (sums and counts; Welford's updates for the spread), O(1) per
/// position whatever the window. Rows are contiguous slices.
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
fn running_block<T: Copy, O: Copy>(
    input: ndarray::ArrayView2<'_, T>,
    mut out: ndarray::ArrayViewMut2<'_, O>,
    to: &(impl Fn(T) -> f64 + Sync),
    from: &(impl Fn(f64) -> O + Sync),
    window: usize,
    back: usize,
    min_periods: usize,
    stat: Running,
) {
    let (n, m) = input.dim();
    let need = min_periods.max(1) as f64;
    let row = |r: usize| input.row(r);
    let first = window.saturating_sub(back).min(n);
    match stat {
        Running::Sum | Running::Mean | Running::Count => {
            let (mut c, mut s) = (vec![0.0_f64; m], vec![0.0_f64; m]);
            let add = |r: usize, c: &mut [f64], s: &mut [f64]| {
                for ((v, c), s) in row(r).iter().zip(c.iter_mut()).zip(s.iter_mut()) {
                    let x = to(*v);
                    if !x.is_nan() {
                        *c += 1.0;
                        *s += x;
                    }
                }
            };
            let remove = |r: usize, c: &mut [f64], s: &mut [f64]| {
                for ((v, c), s) in row(r).iter().zip(c.iter_mut()).zip(s.iter_mut()) {
                    let x = to(*v);
                    if !x.is_nan() {
                        *c -= 1.0;
                        // Empty again: start afresh (no drift left behind).
                        *s = if *c <= 0.0 { 0.0 } else { *s - x };
                    }
                }
            };
            for r in 0..first {
                add(r, &mut c, &mut s);
            }
            for i in 0..n {
                for ((o, c), s) in out.row_mut(i).iter_mut().zip(&c).zip(&s) {
                    *o = from(if *c < need {
                        f64::NAN
                    } else {
                        match stat {
                            Running::Sum => *s,
                            Running::Mean => *s / *c,
                            _ => *c,
                        }
                    });
                }
                if let Some(r) = i
                    .saturating_add(window)
                    .checked_sub(back)
                    .filter(|r| *r < n)
                {
                    add(r, &mut c, &mut s);
                }
                if let Some(r) = i.checked_sub(back) {
                    remove(r, &mut c, &mut s);
                }
            }
        }
        Running::Var | Running::Std => {
            let (mut c, mut mu, mut q) = (vec![0.0_f64; m], vec![0.0_f64; m], vec![0.0_f64; m]);
            let add = |r: usize, c: &mut [f64], mu: &mut [f64], q: &mut [f64]| {
                for (((v, c), mu), q) in row(r)
                    .iter()
                    .zip(c.iter_mut())
                    .zip(mu.iter_mut())
                    .zip(q.iter_mut())
                {
                    let x = to(*v);
                    if !x.is_nan() {
                        *c += 1.0;
                        let d = x - *mu;
                        *mu += d / *c;
                        *q += d * (x - *mu);
                    }
                }
            };
            let remove = |r: usize, c: &mut [f64], mu: &mut [f64], q: &mut [f64]| {
                for (((v, c), mu), q) in row(r)
                    .iter()
                    .zip(c.iter_mut())
                    .zip(mu.iter_mut())
                    .zip(q.iter_mut())
                {
                    let x = to(*v);
                    if !x.is_nan() {
                        *c -= 1.0;
                        if *c <= 0.0 {
                            (*c, *mu, *q) = (0.0, 0.0, 0.0);
                        } else {
                            let d = x - *mu;
                            *mu -= d / *c;
                            *q -= d * (x - *mu);
                        }
                    }
                }
            };
            for r in 0..first {
                add(r, &mut c, &mut mu, &mut q);
            }
            for i in 0..n {
                for ((o, c), q) in out.row_mut(i).iter_mut().zip(&c).zip(&q) {
                    *o = from(if *c < need {
                        f64::NAN
                    } else {
                        let var = q.max(0.0) / *c;
                        if stat == Running::Std {
                            var.sqrt()
                        } else {
                            var
                        }
                    });
                }
                if let Some(r) = i
                    .saturating_add(window)
                    .checked_sub(back)
                    .filter(|r| *r < n)
                {
                    add(r, &mut c, &mut mu, &mut q);
                }
                if let Some(r) = i.checked_sub(back) {
                    remove(r, &mut c, &mut mu, &mut q);
                }
            }
        }
    }
}

/// [`running_block`] over a whole array along `axis`, in parallel blocks of columns (the
/// axis moved first, so each step reads one contiguous row).
#[allow(clippy::too_many_arguments)]
fn running<T: Copy + Send + Sync, O: Copy + Send + Sync>(
    x: &ArrayViewD<'_, T>,
    axis: usize,
    to: impl Fn(T) -> f64 + Sync + Send,
    from: impl Fn(f64) -> O + Sync + Send,
    fill: O,
    window: usize,
    back: usize,
    min_periods: usize,
    stat: Running,
) -> Result<ArrayD<O>> {
    let nd = x.ndim();
    let mut perm: Vec<usize> = vec![axis];
    perm.extend((0..nd).filter(|a| *a != axis));
    let moved = x.view().permuted_axes(IxDyn(&perm));
    let moved_shape = moved.shape().to_vec();
    let contiguous = moved.as_standard_layout();
    let n = moved_shape.first().copied().unwrap_or(1);
    let m = moved_shape.iter().skip(1).product::<usize>();
    let bad = |e: ndarray::ShapeError| invalid(format!("rolling: {e}"));
    let input = contiguous
        .view()
        .into_shape_with_order((n, m))
        .map_err(bad)?;
    // The kernel writes every element: a zero `fill` lets the allocator hand out untouched
    // pages, filled in parallel as the blocks write them.
    let mut out =
        ndarray::Array2::from_shape_vec((n, m), vec![fill; n.saturating_mul(m)]).map_err(bad)?;
    let work = |(k, block): (usize, ndarray::ArrayViewMut2<'_, O>)| {
        let c0 = k.saturating_mul(BLOCK);
        let cols = block.ncols();
        let part = input.slice(ndarray::s![.., c0..c0.saturating_add(cols)]);
        running_block(part, block, &to, &from, window, back, min_periods, stat);
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        out.axis_chunks_iter_mut(Axis(1), BLOCK)
            .into_par_iter()
            .enumerate()
            .for_each(work);
    }
    #[cfg(not(feature = "rayon"))]
    out.axis_chunks_iter_mut(Axis(1), BLOCK)
        .enumerate()
        .for_each(work);
    // Back to the input's axis order.
    let shaped = out
        .into_shape_with_order(IxDyn(&moved_shape))
        .map_err(bad)?;
    let mut inverse = vec![0; nd];
    for (i, p) in perm.iter().enumerate() {
        if let Some(slot) = inverse.get_mut(*p) {
            *slot = i;
        }
    }
    let in_order = shaped.permuted_axes(IxDyn(&inverse));
    Ok(if in_order.is_standard_layout() {
        in_order
    } else {
        in_order.as_standard_layout().into_owned()
    })
}

/// The O(1)-per-position kernel for sums, means, counts and the spread, in the data's own
/// type (no `f64` copy); `None` for other reductions and types.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn rolling_fast(
    typed: &TypedArray<crate::View<'_>>,
    axis: usize,
    window: usize,
    back: usize,
    min_periods: usize,
    op: Reduction,
) -> Option<Result<TypedArray<crate::Owned>>> {
    let stat = Running::of(op)?;
    let float_out = stat != Running::Count;
    macro_rules! go {
        ($x:expr, $to:expr) => {
            Some(
                running(
                    $x,
                    axis,
                    $to,
                    |v| v,
                    f64::NAN,
                    window,
                    back,
                    min_periods,
                    stat,
                )
                .map(TypedArray::F64),
            )
        };
    }
    match typed {
        TypedArray::F32(x) if float_out => Some(
            running(
                x,
                axis,
                f64::from,
                |v| v as f32,
                f32::NAN,
                window,
                back,
                min_periods,
                stat,
            )
            .map(TypedArray::F32),
        ),
        TypedArray::F32(x) => go!(x, f64::from),
        TypedArray::F64(x) => go!(x, |v: f64| v),
        TypedArray::I8(x) => go!(x, f64::from),
        TypedArray::I16(x) => go!(x, f64::from),
        TypedArray::I32(x) => go!(x, f64::from),
        TypedArray::I64(x) => go!(x, |v: i64| v as f64),
        TypedArray::U8(x) => go!(x, f64::from),
        TypedArray::U16(x) => go!(x, f64::from),
        TypedArray::U32(x) => go!(x, f64::from),
        TypedArray::U64(x) => go!(x, |v: u64| v as f64),
        _ => None,
    }
}

/// Reduce a variable over moving windows, one window at a time (memory stays that of the
/// input and output; [`windowed`] is only for `construct`).
fn rolling_reduce<S: Repr>(
    v: &VariableBase<S>,
    dim: &Name,
    window: usize,
    opts: &RollingOptions,
    op: Reduction,
) -> Result<Variable> {
    let min_periods = opts.min_periods.unwrap_or(window);
    let axis = v.axis(dim)?;
    let typed = v.typed()?;
    let dtype = typed.dtype();
    if matches!(
        dtype,
        DType::Str | DType::Bool | DType::DateTime | DType::CfDate | DType::TimeDelta
    ) {
        return Err(CoreError::UnsupportedDType {
            op: "rolling",
            dtype,
        });
    }
    if matches!(op, Reduction::Any | Reduction::All) {
        return Err(CoreError::UnsupportedDType {
            op: op.name(),
            dtype,
        });
    }
    let back = back(window, opts.center);
    if let Some(fast) = rolling_fast(&typed, axis, window, back, min_periods, op) {
        let mut result = Variable::from_typed(v.dims().iter().cloned(), fast?)?;
        result.set_unit(crate::reduce::reduced_unit(v.unit(), op, window)?);
        return Ok(result);
    }
    let data = to_f64(&typed, None)?;
    let mut out = ArrayD::from_elem(data.raw_dim(), f64::NAN);
    let zip = ndarray::Zip::from(data.lanes(Axis(axis))).and(out.lanes_mut(Axis(axis)));
    let each = |lane: ArrayView1<'_, f64>, dst: ArrayViewMut1<'_, f64>| {
        roll_lane(lane, dst, window, back, min_periods, op);
    };
    #[cfg(feature = "rayon")]
    zip.par_for_each(each);
    #[cfg(not(feature = "rayon"))]
    zip.for_each(each);
    // Integer results (count, argmin, argmax) are floats here, to hold NaN.
    let keeps_type = !matches!(op, Reduction::Count | Reduction::ArgMin | Reduction::ArgMax);
    #[allow(clippy::cast_possible_truncation)] // f32 data stays f32
    let data = if dtype == DType::F32 && keeps_type {
        TypedArray::F32(out.mapv(|x| x as f32))
    } else {
        TypedArray::F64(out)
    };
    let mut result = Variable::from_typed(v.dims().iter().cloned(), data)?;
    result.set_unit(crate::reduce::reduced_unit(v.unit(), op, window)?);
    Ok(result)
}

fn check(window: usize, opts: &RollingOptions) -> Result<()> {
    if window == 0 {
        return Err(invalid("the rolling window must be at least 1"));
    }
    if opts.min_periods.is_some_and(|m| m == 0 || m > window) {
        return Err(invalid("min_periods must be between 1 and the window size"));
    }
    Ok(())
}

macro_rules! rolling_shortcuts {
    ($out:ty; $($m:ident $op:ident $doc:literal),* $(,)?) => {$(
        #[doc = $doc]
        ///
        /// # Errors
        /// As for [`reduce`](Self::reduce).
        pub fn $m(&self) -> Result<$out> {
            self.reduce(Reduction::$op)
        }
    )*};
}

impl<S: Repr> DataArrayRolling<'_, S> {
    /// The windows as a new trailing dimension `window_dim` (NaN-padded, as floats).
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for non-numeric data; read errors for lazy data.
    pub fn construct(&self, window_dim: &str) -> Result<DataArray> {
        let v = windowed(
            self.obj.variable(),
            &self.dim,
            self.window,
            self.opts.center,
            &Name::from(window_dim),
        )?;
        let coords = self
            .obj
            .coords()
            .iter()
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(
            self.obj.name().cloned(),
            v,
            coords,
        ))
    }

    /// Reduce every window. Argmin and argmax give positions within the window.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for non-numeric data (or `any`/`all`); unit errors as
    /// for the reduction.
    pub fn reduce(&self, op: Reduction) -> Result<DataArray> {
        if matches!(op, Reduction::Any | Reduction::All) {
            return Err(CoreError::UnsupportedDType {
                op: op.name(),
                dtype: self.obj.dtype(),
            });
        }
        let mut v = rolling_reduce(self.obj.variable(), &self.dim, self.window, &self.opts, op)?;
        *v.attrs_mut() = self.obj.attrs().clone();
        let coords = self
            .obj
            .coords()
            .iter()
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(
            self.obj.name().cloned(),
            v,
            coords,
        ))
    }

    rolling_shortcuts!(DataArray;
        sum Sum "Moving sum.",
        prod Prod "Moving product.",
        mean Mean "Moving mean.",
        median Median "Moving median.",
        std Std "Moving standard deviation (`ddof = 0`).",
        var Var "Moving variance (`ddof = 0`).",
        min Min "Moving minimum.",
        max Max "Moving maximum.",
        count Count "Number of non-NaN values in each window.",
        argmin ArgMin "Position of the minimum within each window.",
        argmax ArgMax "Position of the maximum within each window.",
    );
}

/// A dataset with a moving window along one dimension. Made by [`DatasetBase::rolling`].
pub struct DatasetRolling<'a, S: Repr> {
    obj: &'a DatasetBase<S>,
    dim: Name,
    window: usize,
    opts: RollingOptions,
}

impl<S: Repr> DatasetRolling<'_, S> {
    /// Reduce every window of every numeric data variable with the dimension; other
    /// variables are kept as they are.
    ///
    /// # Errors
    /// As for [`DataArrayRolling::reduce`].
    pub fn reduce(&self, op: Reduction) -> Result<Dataset> {
        let mut out = Dataset::new();
        for (k, c) in self.obj.coords() {
            out.add_coord(k.clone(), c.to_owned())?;
        }
        for (k, v) in self.obj.data_vars() {
            let numeric = !matches!(
                v.dtype(),
                DType::Str | DType::Bool | DType::DateTime | DType::CfDate | DType::TimeDelta
            );
            let r = if v.dims().contains(&self.dim) && numeric {
                let mut r = rolling_reduce(v, &self.dim, self.window, &self.opts, op)?;
                *r.attrs_mut() = v.attrs().clone();
                r
            } else {
                v.to_owned()
            };
            out.add_var(k.clone(), r)?;
        }
        out.attrs_mut()
            .extend(self.obj.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
        Ok(out)
    }

    rolling_shortcuts!(Dataset;
        sum Sum "Moving sum.",
        prod Prod "Moving product.",
        mean Mean "Moving mean.",
        median Median "Moving median.",
        std Std "Moving standard deviation (`ddof = 0`).",
        var Var "Moving variance (`ddof = 0`).",
        min Min "Moving minimum.",
        max Max "Moving maximum.",
        count Count "Number of non-NaN values in each window.",
        argmin ArgMin "Position of the minimum within each window.",
        argmax ArgMax "Position of the maximum within each window.",
    );
}

impl<S: Repr> DataArrayBase<S> {
    /// A moving window of `window` positions along `dim`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`]; [`CoreError::InvalidReduction`] for a zero window or
    /// `min_periods` outside `1..=window`.
    pub fn rolling(
        &self,
        dim: &str,
        window: usize,
        opts: &RollingOptions,
    ) -> Result<DataArrayRolling<'_, S>> {
        let dim = Name::from(dim);
        self.variable().axis(&dim)?;
        check(window, opts)?;
        Ok(DataArrayRolling {
            obj: self,
            dim,
            window,
            opts: *opts,
        })
    }
}

impl<S: Repr> DatasetBase<S> {
    /// A moving window of `window` positions along `dim`.
    ///
    /// # Errors
    /// As for [`DataArrayBase::rolling`].
    pub fn rolling(
        &self,
        dim: &str,
        window: usize,
        opts: &RollingOptions,
    ) -> Result<DatasetRolling<'_, S>> {
        let dim = Name::from(dim);
        if !self.sizes().contains_key(&dim) {
            return Err(CoreError::NoSuchDim(dim));
        }
        check(window, opts)?;
        Ok(DatasetRolling {
            obj: self,
            dim,
            window,
            opts: *opts,
        })
    }
}
