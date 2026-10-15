//! Lazily loaded data: a source that can read hyperslabs, plus a composed selection.
//!
//! Selecting from a lazy variable reads nothing; it composes the selection. The data is read
//! (only the selected box) on first access and then cached.

use std::fmt;
use std::sync::{Arc, OnceLock};

use crate::indexer::SliceIndices;
use crate::{CoreError, DType, Owned, Result, TypedArray};

/// Something that can read rectangular slices of an n-dimensional array: a variable in a file.
pub trait LazySource: Send + Sync + fmt::Debug {
    /// Element type of the values `read_slice` returns.
    fn dtype(&self) -> DType;
    /// Full shape.
    fn shape(&self) -> &[usize];
    /// Read `[start, start + count)`. Implementations should touch only what the slice spans.
    ///
    /// # Errors
    /// [`CoreError::Read`] if the underlying file cannot be read.
    fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>>;

    /// How the data is chunked on disk along `axis`, if it is: chunks are `size` long and
    /// their boundaries fall at `offset + k * size` (`offset < size`). Readers that go
    /// through the data in slabs end them on boundaries, so no chunk is decoded twice.
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        let _ = axis;
        None
    }
}

/// The selection on one axis of the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sel {
    /// A single position: the axis is dropped.
    Index(usize),
    /// `len` positions starting at `start`, `step` apart (`step` may be negative).
    Range {
        start: usize,
        step: isize,
        len: usize,
    },
}

/// What indexing does to one visible axis (mirrors the variable's own indexing).
#[derive(Debug, Clone, Copy)]
pub(crate) enum LazyOp {
    At(usize),
    Slice(SliceIndices),
}

/// A lazily loaded array: a source and a selection on it.
#[derive(Clone)]
pub struct LazyArray {
    source: Arc<dyn LazySource>,
    sel: Vec<Sel>,
    shape: Vec<usize>,
    cache: Arc<OnceLock<TypedArray<Owned>>>,
}

impl fmt::Debug for LazyArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LazyArray({:?}, {}, loaded: {})",
            self.shape,
            self.dtype(),
            self.is_loaded()
        )
    }
}

fn overflow() -> CoreError {
    crate::IndexError::Overflow.into()
}

/// `base + i * step` as a position.
fn offset(base: usize, i: usize, step: isize) -> Result<usize> {
    let b = isize::try_from(base).map_err(|_| overflow())?;
    let i = isize::try_from(i).map_err(|_| overflow())?;
    let p = i
        .checked_mul(step)
        .and_then(|d| b.checked_add(d))
        .ok_or_else(overflow)?;
    usize::try_from(p).map_err(|_| overflow())
}

impl LazyArray {
    /// The whole of `source`.
    pub fn new(source: Arc<dyn LazySource>) -> Self {
        let shape = source.shape().to_vec();
        let sel = shape
            .iter()
            .map(|n| Sel::Range {
                start: 0,
                step: 1,
                len: *n,
            })
            .collect();
        Self {
            source,
            sel,
            shape,
            cache: Arc::new(OnceLock::new()),
        }
    }

    /// Shape of the selection.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.source.dtype()
    }

    /// Number of elements selected.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether nothing is selected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the selection has been read.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.cache.get().is_some()
    }

    /// Compose an indexing operation (one entry per visible axis). Reads nothing.
    pub(crate) fn isel(&self, ops: &[Option<LazyOp>]) -> Result<Self> {
        let mut sel = self.sel.clone();
        let mut visible = sel.iter_mut().filter(|s| matches!(s, Sel::Range { .. }));
        for op in ops {
            let Some(slot) = visible.next() else {
                return Err(overflow());
            };
            let Sel::Range { start, step, len } = *slot else {
                continue;
            };
            *slot = match op {
                None => *slot,
                Some(LazyOp::At(i)) => Sel::Index(offset(start, *i, step)?),
                Some(LazyOp::Slice(si)) => {
                    let new_start = if si.len == 0 {
                        start
                    } else {
                        offset(start, si.start, step)?
                    };
                    let new_step = step.checked_mul(si.step).ok_or_else(overflow)?;
                    let _ = len;
                    Sel::Range {
                        start: new_start,
                        step: new_step,
                        len: si.len,
                    }
                }
            };
        }
        let shape = sel
            .iter()
            .filter_map(|s| match s {
                Sel::Range { len, .. } => Some(*len),
                Sel::Index(_) => None,
            })
            .collect();
        Ok(Self {
            source: self.source.clone(),
            sel,
            shape,
            cache: Arc::new(OnceLock::new()),
        })
    }

    /// The selected values, reading them on first use.
    ///
    /// # Errors
    /// [`CoreError::Read`] if the source cannot be read.
    pub fn load(&self) -> Result<&TypedArray<Owned>> {
        if let Some(v) = self.cache.get() {
            return Ok(v);
        }
        let v = self.read()?;
        let _ = self.cache.set(v); // another thread may have won the race; either value is equal
        self.cache
            .get()
            .ok_or_else(|| CoreError::Read("lazy cache".into()))
    }

    /// Read the covering box of the selection, then apply steps and drop indexed axes.
    fn read(&self) -> Result<TypedArray<Owned>> {
        let mut start = Vec::with_capacity(self.sel.len());
        let mut count = Vec::with_capacity(self.sel.len());
        let mut post: Vec<Option<ndarray::Slice>> = Vec::with_capacity(self.sel.len());
        for s in &self.sel {
            match *s {
                Sel::Index(i) => {
                    start.push(i);
                    count.push(1);
                    post.push(None);
                }
                Sel::Range {
                    start: s0,
                    step,
                    len,
                } => {
                    if len == 0 {
                        start.push(0);
                        count.push(0);
                        post.push(Some(ndarray::Slice::new(0, None, 1)));
                        continue;
                    }
                    let last = offset(s0, len.saturating_sub(1), step)?;
                    let (lo, hi) = if step > 0 { (s0, last) } else { (last, s0) };
                    start.push(lo);
                    count.push(hi.saturating_sub(lo).saturating_add(1));
                    // A negative step walks the box backwards from its end, i.e. from `s0`.
                    post.push(Some(ndarray::Slice::new(0, None, step)));
                }
            }
        }
        let mut data = self.source.read_slice(&start, &count)?;
        if data.shape() != count.as_slice() {
            return Err(CoreError::Read(format!(
                "source returned shape {:?}, expected {count:?}",
                data.shape()
            )));
        }
        for (axis, p) in post.iter().enumerate() {
            if let Some(s) = p
                && s.step != 1
            {
                data.slice_axis_inplace(axis, *s);
            }
        }
        for (axis, p) in post.iter().enumerate().rev() {
            if p.is_none() {
                data = data.index_axis_move(axis, 0);
            }
        }
        Ok(data)
    }
}

impl LazyArray {
    /// The source's chunking along visible `axis`, shifted by the selection (unit steps only).
    pub(crate) fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        let (u, sel) = self
            .sel
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s, Sel::Range { .. }))
            .nth(axis)?;
        let Sel::Range { start, step: 1, .. } = *sel else {
            return None;
        };
        let (size, offset) = self.source.chunking(u)?;
        let shift = start.checked_rem(size)?;
        Some((
            size,
            offset
                .checked_add(size)?
                .checked_sub(shift)?
                .checked_rem(size)?,
        ))
    }

    /// Read `[start, start + count)` of the selection (not cached).
    pub(crate) fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>> {
        let ops: Vec<Option<LazyOp>> = start
            .iter()
            .zip(count)
            .map(|(s, n)| {
                Some(LazyOp::Slice(SliceIndices {
                    start: *s,
                    step: 1,
                    len: *n,
                }))
            })
            .collect();
        if ops.len() != self.shape.len()
            || start
                .iter()
                .zip(count)
                .zip(&self.shape)
                .any(|((s, n), len)| s.checked_add(*n).is_none_or(|e| e > *len))
        {
            return Err(CoreError::Read(format!(
                "slice {start:?}+{count:?} outside shape {:?}",
                self.shape
            )));
        }
        self.isel(&ops)?.read()
    }
}

/// A variable's lazy data as a source (its selection applied).
#[derive(Debug)]
pub(crate) struct ViewSource(pub(crate) LazyArray);

impl LazySource for ViewSource {
    fn dtype(&self) -> DType {
        self.0.dtype()
    }
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>> {
        self.0.read_slice(start, count)
    }
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        self.0.chunking(axis)
    }
}

/// A function applied to whichever box of `src` is read (elementwise maps such as
/// `isnull` or `fillna` on lazy data).
pub(crate) struct MapSource {
    src: Arc<dyn LazySource>,
    dtype: DType,
    what: &'static str,
    f: Box<dyn Fn(TypedArray<Owned>) -> Result<TypedArray<Owned>> + Send + Sync>,
}

impl fmt::Debug for MapSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MapSource({}, {:?})", self.what, self.src)
    }
}

impl MapSource {
    /// `f` over `src`; its element type comes from applying `f` to an empty array.
    pub(crate) fn new(
        src: Arc<dyn LazySource>,
        what: &'static str,
        f: impl Fn(TypedArray<Owned>) -> Result<TypedArray<Owned>> + Send + Sync + 'static,
    ) -> Result<Self> {
        let empty = TypedArray::empty(src.dtype(), &vec![0; src.shape().len().max(1)])
            .ok_or_else(|| CoreError::Read(format!("{what}: empty probe")))?;
        let dtype = f(empty)?.dtype();
        Ok(Self {
            src,
            dtype,
            what,
            f: Box::new(f),
        })
    }
}

impl LazySource for MapSource {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        self.src.shape()
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>> {
        (self.f)(self.src.read_slice(start, count)?)
    }
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        self.src.chunking(axis)
    }
}

/// Positions `indices` of `src` along `axis` (fancy indexing), read for whichever box is
/// asked: runs of consecutive positions are read as slices, so only the selected data is
/// read (or, for many scattered positions, the box covering them).
#[derive(Debug)]
pub(crate) struct TakeSource {
    src: Arc<dyn LazySource>,
    axis: usize,
    indices: Vec<usize>,
    shape: Vec<usize>,
}

/// Above this many runs, covering reads (of several positions each) beat one read per run.
const TAKE_MAX_RUNS: usize = 64;

/// How much larger than the positions it keeps a covering read may be.
const TAKE_SPAN: usize = 4;

/// Extra rows a covering read may span regardless.
const TAKE_SLACK: usize = 16;

impl TakeSource {
    pub(crate) fn new(src: Arc<dyn LazySource>, axis: usize, indices: Vec<usize>) -> Result<Self> {
        let len = src.shape().get(axis).copied().ok_or_else(overflow)?;
        if indices.iter().any(|i| *i >= len) {
            return Err(CoreError::Read("take: position outside the data".into()));
        }
        let mut shape = src.shape().to_vec();
        if let Some(n) = shape.get_mut(axis) {
            *n = indices.len();
        }
        Ok(Self {
            src,
            axis,
            indices,
            shape,
        })
    }
}

impl LazySource for TakeSource {
    fn dtype(&self) -> DType {
        self.src.dtype()
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>> {
        let (lo, n) = (
            start.get(self.axis).copied().ok_or_else(overflow)?,
            count.get(self.axis).copied().ok_or_else(overflow)?,
        );
        let want = self
            .indices
            .get(lo..lo.checked_add(n).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        let read_box = |a: usize, len: usize| {
            let (mut s, mut c) = (start.to_vec(), count.to_vec());
            if let (Some(x), Some(y)) = (s.get_mut(self.axis), c.get_mut(self.axis)) {
                *x = a;
                *y = len;
            }
            self.src.read_slice(&s, &c)
        };
        // Runs of consecutive positions.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for i in want {
            match runs.last_mut() {
                Some((a, len)) if a.checked_add(*len) == Some(*i) => *len = len.saturating_add(1),
                _ => runs.push((*i, 1)),
            }
        }
        if runs.len() <= TAKE_MAX_RUNS {
            let pieces = runs
                .iter()
                .map(|(a, len)| read_box(*a, *len))
                .collect::<Result<Vec<_>>>()?;
            if pieces.len() == 1 {
                return pieces.into_iter().next().ok_or_else(overflow);
            }
            let first = pieces.first().ok_or_else(overflow)?;
            return crate::array::typed_dispatch!(first, x => join_as(&pieces, self.axis, crate::array::marker(x)))
                .ok_or_else(|| CoreError::Read("take: joining runs".into()));
        }
        // Many scattered positions: windows of them (in order), each read as the box
        // covering it and picked from. A window grows only while its span stays within
        // `TAKE_SPAN` times its positions (plus slack), so reads stay near what is kept;
        // neighbouring positions share one read (one decode of their chunk).
        let mut pieces = Vec::new();
        let mut w = 0usize;
        while w < want.len() {
            let (mut lo, mut hi) = (usize::MAX, 0usize);
            let mut end = w;
            for (k, i) in want.iter().enumerate().skip(w) {
                let (nlo, nhi) = (lo.min(*i), hi.max(*i));
                let span = nhi.saturating_sub(nlo).saturating_add(1);
                let n = k.saturating_sub(w).saturating_add(1);
                if k > w && span > n.saturating_mul(TAKE_SPAN).saturating_add(TAKE_SLACK) {
                    break;
                }
                (lo, hi, end) = (nlo, nhi, k.saturating_add(1));
            }
            let covering = read_box(lo, hi.saturating_sub(lo).saturating_add(1))?;
            let rel: Vec<usize> = want
                .get(w..end)
                .unwrap_or_default()
                .iter()
                .map(|i| i.saturating_sub(lo))
                .collect();
            pieces.push(covering.view().select_axis(self.axis, &rel));
            w = end;
        }
        if pieces.len() == 1 {
            return pieces.into_iter().next().ok_or_else(overflow);
        }
        let first = pieces.first().ok_or_else(overflow)?;
        crate::array::typed_dispatch!(first, x => join_as(&pieces, self.axis, crate::array::marker(x)))
            .ok_or_else(|| CoreError::Read("take: joining windows".into()))
    }
}

/// Data already in memory, as a source (slices are copied out of it).
#[derive(Debug)]
pub struct MemorySource(pub TypedArray<Owned>);

fn slice_as<T: crate::Element>(
    a: &TypedArray<Owned>,
    start: &[usize],
    count: &[usize],
    _: crate::array::Marker<T>,
) -> Option<TypedArray<Owned>> {
    let x = T::peek(a)?;
    let mut ranges = start.iter().zip(count);
    let v = x.slice_each_axis(|_| {
        let (s, n) = ranges.next().map_or((0, 0), |(s, n)| (*s, *n));
        ndarray::Slice::from(s..s.saturating_add(n))
    });
    Some(T::wrap(v.to_owned()))
}

impl LazySource for MemorySource {
    fn dtype(&self) -> DType {
        self.0.dtype()
    }
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>> {
        let bad = start.len() != self.0.ndim()
            || start
                .iter()
                .zip(count)
                .zip(self.0.shape())
                .any(|((s, n), len)| s.checked_add(*n).is_none_or(|e| e > *len));
        if bad {
            return Err(CoreError::Read("slice outside the data".into()));
        }
        crate::array::typed_dispatch!(&self.0, x => slice_as(&self.0, start, count, crate::array::marker(x)))
            .ok_or_else(|| CoreError::Read("slice".into()))
    }
}

fn join_as<T: crate::Element>(
    parts: &[TypedArray<Owned>],
    axis: usize,
    _: crate::array::Marker<T>,
) -> Option<TypedArray<Owned>> {
    let views = parts
        .iter()
        .map(|p| T::peek(p).map(|a| a.view()))
        .collect::<Option<Vec<_>>>()?;
    ndarray::concatenate(ndarray::Axis(axis), &views)
        .ok()
        .map(T::wrap)
}

/// Sources joined along `axis`: an existing axis of every part, or (with `new_axis`) a new
/// one, each part contributing one position. Reads touch only the parts a slice overlaps.
#[derive(Debug)]
pub(crate) struct ConcatSource {
    parts: Vec<Arc<dyn LazySource>>,
    axis: usize,
    new_axis: bool,
    dtype: DType,
    shape: Vec<usize>,
}

impl ConcatSource {
    pub(crate) fn new(
        parts: Vec<Arc<dyn LazySource>>,
        axis: usize,
        new_axis: bool,
    ) -> Result<Self> {
        let first = parts
            .first()
            .ok_or_else(|| CoreError::Read("nothing to join".into()))?;
        let dtype = first.dtype();
        let mut shape = first.shape().to_vec();
        if new_axis {
            if axis > shape.len() {
                return Err(overflow());
            }
            shape.insert(axis, parts.len());
        } else {
            let total = parts
                .iter()
                .map(|p| p.shape().get(axis).copied().unwrap_or(0))
                .try_fold(0usize, usize::checked_add)
                .ok_or_else(overflow)?;
            *shape.get_mut(axis).ok_or_else(overflow)? = total;
        }
        for p in &parts {
            let mut s = p.shape().to_vec();
            if new_axis {
                s.insert(axis.min(s.len()), 1);
            } else if let Some(x) = s.get_mut(axis) {
                *x = shape.get(axis).copied().unwrap_or(0);
            }
            if p.dtype() != dtype
                || s.len() != shape.len()
                || s.iter()
                    .zip(&shape)
                    .enumerate()
                    .any(|(i, (a, b))| i != axis && a != b)
            {
                return Err(CoreError::Read(
                    "joined parts differ in type or shape".into(),
                ));
            }
        }
        Ok(Self {
            parts,
            axis,
            new_axis,
            dtype,
            shape,
        })
    }
}

impl LazySource for ConcatSource {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        // Along the joined axis the parts' chunks need not line up; elsewhere they must agree.
        if axis == self.axis {
            return None;
        }
        let part_axis = if self.new_axis && axis > self.axis {
            axis.checked_sub(1)?
        } else {
            axis
        };
        let mut grids = self.parts.iter().map(|p| p.chunking(part_axis));
        let first = grids.next()??;
        grids.all(|g| g == Some(first)).then_some(first)
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>> {
        let (lo, n) = (
            start.get(self.axis).copied().ok_or_else(overflow)?,
            count.get(self.axis).copied().ok_or_else(overflow)?,
        );
        let hi = lo.checked_add(n).ok_or_else(overflow)?;
        let mut pieces = Vec::new();
        let mut offset = 0usize;
        for p in &self.parts {
            let len = if self.new_axis {
                1
            } else {
                p.shape().get(self.axis).copied().unwrap_or(0)
            };
            let end = offset.checked_add(len).ok_or_else(overflow)?;
            let (a, b) = (lo.max(offset), hi.min(end));
            if a < b {
                let mut s = start.to_vec();
                let mut c = count.to_vec();
                if self.new_axis {
                    s.remove(self.axis);
                    c.remove(self.axis);
                    let piece = p.read_slice(&s, &c)?;
                    pieces.push(piece.insert_axis_owned(self.axis));
                } else {
                    *s.get_mut(self.axis).ok_or_else(overflow)? =
                        a.checked_sub(offset).ok_or_else(overflow)?;
                    *c.get_mut(self.axis).ok_or_else(overflow)? =
                        b.checked_sub(a).ok_or_else(overflow)?;
                    pieces.push(p.read_slice(&s, &c)?);
                }
            }
            offset = end;
        }
        match pieces.first() {
            None => TypedArray::empty(self.dtype, count).ok_or_else(overflow),
            Some(first) => crate::array::typed_dispatch!(first, x => join_as(&pieces, self.axis, crate::array::marker(x)))
                .ok_or_else(|| CoreError::Read("joining slices".into())),
        }
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, reason = "small test values")]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ndarray::{ArrayD, IxDyn};

    use super::*;

    /// A 4x5 source of `10*i + j` that counts elements read.
    #[derive(Debug)]
    struct Counting {
        shape: Vec<usize>,
        read: AtomicUsize,
    }

    impl LazySource for Counting {
        fn dtype(&self) -> DType {
            DType::I64
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<TypedArray<Owned>> {
            self.read
                .fetch_add(count.iter().product(), Ordering::SeqCst);
            Ok(TypedArray::I64(ArrayD::from_shape_fn(IxDyn(count), |ix| {
                ((start[0] + ix[0]) * 10 + (start[1] + ix[1])) as i64
            })))
        }
    }

    #[test]
    fn selections_compose_and_read_only_what_is_needed() {
        let src = Arc::new(Counting {
            shape: vec![4, 5],
            read: AtomicUsize::new(0),
        });
        let lazy = LazyArray::new(src.clone());
        // [1:4, ::-2] then [::2, 0]
        let s = lazy
            .isel(&[
                Some(LazyOp::Slice(
                    crate::PySlice::new(Some(1), Some(4), None)
                        .indices(4)
                        .unwrap(),
                )),
                Some(LazyOp::Slice(
                    crate::PySlice::new(None, None, Some(-2))
                        .indices(5)
                        .unwrap(),
                )),
            ])
            .unwrap();
        assert_eq!(s.shape(), &[3, 3]);
        let s = s
            .isel(&[
                Some(LazyOp::Slice(
                    crate::PySlice::new(None, None, Some(2)).indices(3).unwrap(),
                )),
                Some(LazyOp::At(0)),
            ])
            .unwrap();
        assert_eq!(s.shape(), &[2]);
        assert_eq!(
            src.read.load(Ordering::SeqCst),
            0,
            "selection must not read"
        );
        let TypedArray::I64(v) = s.load().unwrap() else {
            panic!()
        };
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![14, 34]);
        // Covering box: rows 1..=3, column 4 only.
        assert_eq!(src.read.load(Ordering::SeqCst), 3);
        s.load().unwrap();
        assert_eq!(
            src.read.load(Ordering::SeqCst),
            3,
            "second access uses the cache"
        );
    }
}
