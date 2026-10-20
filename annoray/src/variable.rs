//! [`VariableBase`]: an n-dimensional array with named dimensions and attributes.

use std::borrow::Borrow;
use std::fmt;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, IxDyn};
use std::sync::Arc;

use annoray_units::{TimeZone, Unit};

use crate::indexer::{SliceIndices, normalize_index};
use crate::lazy::{LazyArray, LazyOp};
use crate::{
    AttrValue, Attrs, CoreError, DType, Dims, Element, Encoding, IndexError, Indexer, IntoIndexer,
    Name, Owned, Repr, ReprMut, Result, TypedArray, View, ViewMut,
};

/// An n-dimensional array with named dimensions and attributes, in storage representation `S`.
///
/// Use the aliases [`Variable`], [`VariableView`] and [`VariableViewMut`].
pub struct VariableBase<S: Repr> {
    data: Data<S>,
    dims: Dims,
    /// Physical unit, if known. Views carry a copy; it is read-only through a `ViewMut`.
    unit: Option<Unit>,
    /// For date-times, the zone their wall-clock times are in (UTC when `None`).
    time_zone: Option<TimeZone>,
    /// How the variable is (or will be) stored on disk. Shared cheaply by views.
    encoding: Arc<Encoding>,
    attrs: S::Attrs,
}

/// A variable that owns its data and attributes.
pub type Variable = VariableBase<Owned>;
/// A shared view of a variable.
pub type VariableView<'a> = VariableBase<View<'a>>;
/// A view of a variable whose element data can be modified. Dimensions and attributes are read-only.
pub type VariableViewMut<'a> = VariableBase<ViewMut<'a>>;

/// A variable's values: in memory (in representation `S`) or read lazily from a file.
pub(crate) enum Data<S: Repr> {
    Loaded(TypedArray<S>),
    Lazy(LazyArray),
}

impl<S: Repr> fmt::Debug for Data<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loaded(a) => fmt::Debug::fmt(a, f),
            Self::Lazy(l) => fmt::Debug::fmt(l, f),
        }
    }
}

impl Clone for Data<Owned> {
    fn clone(&self) -> Self {
        match self {
            Self::Loaded(a) => Self::Loaded(a.clone()),
            Self::Lazy(l) => Self::Lazy(l.clone()),
        }
    }
}

impl Clone for Data<View<'_>> {
    fn clone(&self) -> Self {
        match self {
            Self::Loaded(a) => Self::Loaded(a.clone()),
            Self::Lazy(l) => Self::Lazy(l.clone()),
        }
    }
}

/// What an indexing operation does to one axis.
#[derive(Debug, Clone, Copy)]
enum AxisOp {
    At(usize),
    Slice(ndarray::Slice, SliceIndices),
}

fn check_dims(dims: &Dims, ndim: usize) -> Result<()> {
    if dims.len() != ndim {
        return Err(CoreError::DimCountMismatch {
            ndim,
            names: dims.len(),
        });
    }
    for (i, d) in dims.iter().enumerate() {
        if dims.iter().take(i).any(|prev| prev == d) {
            return Err(CoreError::DuplicateDim(d.clone()));
        }
    }
    Ok(())
}

/// Assign positional indexers (with at most one `..`) to dimension names.
///
/// # Errors
/// [`IndexError::MultipleEllipsis`], [`IndexError::TooManyIndexers`], or a conversion error.
pub(crate) fn positional_to_named<I>(dims: &[Name], indexers: I) -> Result<Vec<(Name, Indexer)>>
where
    I: IntoIterator,
    I::Item: IntoIndexer,
{
    let items = indexers
        .into_iter()
        .map(IntoIndexer::into_indexer)
        .collect::<Result<Vec<_>, IndexError>>()?;
    let ndim = dims.len();
    let is_ellipsis = |ix: &Indexer| matches!(ix, Indexer::Ellipsis);
    let n_ellipsis = items.iter().filter(|ix| is_ellipsis(ix)).count();
    if n_ellipsis > 1 {
        return Err(IndexError::MultipleEllipsis.into());
    }
    let given = items.len().saturating_sub(n_ellipsis);
    if given > ndim {
        return Err(IndexError::TooManyIndexers { given, ndim }.into());
    }
    let (head, tail) = match items.iter().position(is_ellipsis) {
        Some(p) => {
            let (head, rest) = items.split_at_checked(p).ok_or(IndexError::Overflow)?;
            (head, rest.get(1..).unwrap_or_default())
        }
        None => (items.as_slice(), &[][..]),
    };
    let tail_dims = dims
        .get(ndim.checked_sub(tail.len()).ok_or(IndexError::Overflow)?..)
        .unwrap_or_default();
    Ok(dims
        .iter()
        .zip(head)
        .chain(tail_dims.iter().zip(tail))
        .map(|(d, ix)| (d.clone(), *ix))
        .collect())
}

impl Variable {
    /// Create a variable from dimension names and an array.
    ///
    /// # Errors
    /// [`CoreError::DimCountMismatch`] if the number of names differs from `data.ndim()`;
    /// [`CoreError::DuplicateDim`] if a name repeats.
    pub fn new<T, D>(dims: D, data: ArrayD<T>) -> Result<Self>
    where
        T: Element,
        D: IntoIterator,
        D::Item: Into<Name>,
    {
        Self::from_typed(dims, T::wrap(data))
    }

    /// Create a variable from dimension names and a dtype-erased array.
    ///
    /// # Errors
    /// As for [`Variable::new`].
    pub fn from_typed<D>(dims: D, data: TypedArray<Owned>) -> Result<Self>
    where
        D: IntoIterator,
        D::Item: Into<Name>,
    {
        let dims: Dims = dims.into_iter().map(Into::into).collect();
        check_dims(&dims, data.ndim())?;
        Ok(Self {
            data: Data::Loaded(data),
            dims,
            unit: None,
            time_zone: None,
            encoding: Arc::default(),
            attrs: Attrs::new(),
        })
    }

    /// Add attributes, replacing any with the same names.
    #[must_use]
    pub fn with_attrs<K, V>(mut self, attrs: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<Name>,
        V: Into<AttrValue>,
    {
        self.attrs
            .extend(attrs.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// Set the unit.
    #[must_use]
    pub fn with_unit(mut self, unit: Unit) -> Self {
        self.unit = Some(unit);
        self
    }

    /// Replace the unit, returning the old one.
    pub fn set_unit(&mut self, unit: Option<Unit>) -> Option<Unit> {
        std::mem::replace(&mut self.unit, unit)
    }

    /// Set the time zone of date-time data: the zone its wall-clock times (components,
    /// resampling bins, display) are in. The instants are unchanged.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for data that are not date-times
    /// ([`DType::DateTime`]).
    pub fn with_time_zone(mut self, tz: TimeZone) -> Result<Self> {
        self.set_time_zone(Some(tz))?;
        Ok(self)
    }

    /// Replace the time zone of date-time data (`None` is UTC), returning the old one.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] when setting a zone on data that are not date-times.
    pub fn set_time_zone(&mut self, tz: Option<TimeZone>) -> Result<Option<TimeZone>> {
        if tz.is_some() && self.dtype() != DType::DateTime {
            return Err(CoreError::UnsupportedDType {
                op: "time zone",
                dtype: self.dtype(),
            });
        }
        Ok(std::mem::replace(&mut self.time_zone, tz))
    }

    /// Take `src`'s time zone when both hold date-times: a result computed from date-times
    /// keeps their zone.
    pub(crate) fn take_zone_from<S2: Repr>(&mut self, src: &VariableBase<S2>) {
        if src.dtype() == DType::DateTime && self.dtype() == DType::DateTime {
            self.time_zone.clone_from(&src.time_zone);
        }
    }

    /// Mutable access to the attributes.
    pub fn attrs_mut(&mut self) -> &mut Attrs {
        &mut self.attrs
    }

    /// Set the encoding.
    #[must_use]
    pub fn with_encoding(mut self, encoding: Encoding) -> Self {
        self.encoding = Arc::new(encoding);
        self
    }

    /// A variable whose data is read lazily from `data` (only the selected part, on first
    /// access).
    ///
    /// # Errors
    /// As for [`Variable::new`].
    pub fn from_lazy<D>(dims: D, data: LazyArray) -> Result<Self>
    where
        D: IntoIterator,
        D::Item: Into<Name>,
    {
        let dims: Dims = dims.into_iter().map(Into::into).collect();
        check_dims(&dims, data.shape().len())?;
        Ok(Self {
            data: Data::Lazy(data),
            dims,
            unit: None,
            time_zone: None,
            encoding: Arc::default(),
            attrs: Attrs::new(),
        })
    }

    /// Read lazily loaded data into memory (a no-op for data already in memory).
    ///
    /// # Errors
    /// [`CoreError::Read`] if the source cannot be read.
    pub fn load(&mut self) -> Result<()> {
        if let Data::Lazy(l) = &self.data {
            self.data = Data::Loaded(l.clone().into_loaded()?);
        }
        Ok(())
    }

    /// Take the data out of the variable (reading it if it is lazy).
    ///
    /// # Errors
    /// [`CoreError::Read`] if lazily loaded data cannot be read.
    pub fn into_typed(self) -> Result<TypedArray<Owned>> {
        match self.data {
            Data::Loaded(a) => Ok(a),
            Data::Lazy(l) => l.into_loaded(),
        }
    }
}

impl<S: Repr> VariableBase<S> {
    /// Mutable access to the encoding (copied on write: views get their own, the data is
    /// untouched).
    pub fn encoding_mut(&mut self) -> &mut Encoding {
        Arc::make_mut(&mut self.encoding)
    }

    /// Dimension names, in axis order.
    #[must_use]
    pub fn dims(&self) -> &[Name] {
        &self.dims
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Length of each axis, in axis order.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match &self.data {
            Data::Loaded(a) => a.shape(),
            Data::Lazy(l) => l.shape(),
        }
    }

    /// `(name, length)` for each dimension, in axis order.
    pub fn sizes(&self) -> impl Iterator<Item = (&Name, usize)> {
        self.dims.iter().zip(self.shape().iter().copied())
    }

    /// Total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Whether the variable has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        match &self.data {
            Data::Loaded(a) => a.dtype(),
            Data::Lazy(l) => l.dtype(),
        }
    }

    /// The data as a [`LazySource`](crate::LazySource): the lazy selection itself (reading
    /// only what is asked for), or the values in memory.
    ///
    /// # Errors
    /// Read errors for lazy data that is cached but cannot be viewed.
    pub fn as_lazy_source(&self) -> Result<std::sync::Arc<dyn crate::LazySource>> {
        Ok(match &self.data {
            Data::Lazy(l) if !l.is_loaded() => {
                std::sync::Arc::new(crate::lazy::ViewSource(l.clone()))
            }
            _ => std::sync::Arc::new(crate::lazy::MemorySource(self.typed()?.to_owned())),
        })
    }

    /// Whether the data is read lazily (and not yet read).
    #[must_use]
    pub fn is_lazy(&self) -> bool {
        matches!(&self.data, Data::Lazy(l) if !l.is_loaded())
    }

    /// The physical unit, if known.
    #[must_use]
    pub fn unit(&self) -> Option<&Unit> {
        self.unit.as_ref()
    }

    /// The time zone of date-time data, if one is set (otherwise their wall-clock times are in
    /// UTC).
    #[must_use]
    pub fn time_zone(&self) -> Option<&TimeZone> {
        self.time_zone.as_ref()
    }

    /// How the variable is stored on disk (fill values, packing, time units, chunking, …).
    #[must_use]
    pub fn encoding(&self) -> &Encoding {
        &self.encoding
    }

    /// Attributes.
    #[must_use]
    pub fn attrs(&self) -> &Attrs {
        self.attrs.borrow()
    }

    /// The dtype-erased data, as a view (lazily loaded data is read on first access).
    ///
    /// # Errors
    /// [`CoreError::Read`] if lazily loaded data cannot be read.
    pub fn typed(&self) -> Result<TypedArray<View<'_>>> {
        match &self.data {
            Data::Loaded(a) => Ok(a.view()),
            Data::Lazy(l) => Ok(l.load()?.view()),
        }
    }

    /// Axis number of dimension `dim`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] if the variable has no such dimension.
    pub fn axis(&self, dim: &str) -> Result<usize> {
        self.dims
            .iter()
            .position(|d| d == dim)
            .ok_or_else(|| CoreError::NoSuchDim(dim.into()))
    }

    /// Length of dimension `dim`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] if the variable has no such dimension.
    pub fn len_of(&self, dim: &str) -> Result<usize> {
        let axis = self.axis(dim)?;
        self.shape()
            .get(axis)
            .copied()
            .ok_or_else(|| CoreError::NoSuchDim(dim.into()))
    }

    /// The data as a typed ndarray view.
    ///
    /// # Errors
    /// [`CoreError::DTypeMismatch`] if the data is not of element type `T`.
    pub fn data<T: Element>(&self) -> Result<ArrayViewD<'_, T>> {
        let mismatch = CoreError::DTypeMismatch {
            expected: T::DTYPE,
            found: self.dtype(),
        };
        match &self.data {
            Data::Loaded(a) => T::peek(a).map(|a| a.view()).ok_or(mismatch),
            Data::Lazy(l) => T::peek(l.load()?).map(|a| a.view()).ok_or(mismatch),
        }
    }

    /// A shared view of the whole variable.
    #[must_use]
    pub fn view(&self) -> VariableView<'_> {
        VariableBase {
            data: match &self.data {
                Data::Loaded(a) => Data::Loaded(a.view()),
                Data::Lazy(l) => Data::Lazy(l.clone()),
            },
            dims: self.dims.clone(),
            unit: self.unit.clone(),
            time_zone: self.time_zone.clone(),
            encoding: self.encoding.clone(),
            attrs: self.attrs.borrow(),
        }
    }

    /// Copy into an owned variable.
    #[must_use]
    pub fn to_owned(&self) -> Variable {
        VariableBase {
            data: match &self.data {
                Data::Loaded(a) => Data::Loaded(a.to_owned()),
                Data::Lazy(l) => Data::Lazy(l.clone()),
            },
            dims: self.dims.clone(),
            unit: self.unit.clone(),
            time_zone: self.time_zone.clone(),
            encoding: self.encoding.clone(),
            attrs: self.attrs().clone(),
        }
    }

    /// Select by dimension name, returning a view. Dimensions not named are kept whole.
    ///
    /// Integers drop the dimension; slices keep it. Semantics follow Python: negative values
    /// count from the end, slice bounds are clamped, negative steps reverse.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`], or an [`IndexError`] for out-of-bounds integers, zero steps,
    /// repeated dimensions or `..`.
    pub fn isel<I, K, X>(&self, indexers: I) -> Result<VariableView<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoIndexer,
    {
        self.view().isel_move(indexers)
    }

    /// Select by axis position, returning a view. Trailing axes not given are kept whole;
    /// one [`Indexer::Ellipsis`] stands for as many whole axes as needed.
    ///
    /// # Errors
    /// As for [`isel`](Self::isel), plus [`IndexError::TooManyIndexers`] and
    /// [`IndexError::MultipleEllipsis`].
    pub fn isel_pos<I>(&self, indexers: I) -> Result<VariableView<'_>>
    where
        I: IntoIterator,
        I::Item: IntoIndexer,
    {
        self.view().isel_pos_move(indexers)
    }

    /// [`isel`](Self::isel), consuming `self` and keeping its representation.
    ///
    /// # Errors
    /// As for [`isel`](Self::isel).
    pub fn isel_move<I, K, X>(self, indexers: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoIndexer,
    {
        let mut ops = vec![None; self.ndim()];
        for (dim, ix) in indexers {
            let axis = self.axis(dim.as_ref())?;
            match ix.into_indexer()? {
                Indexer::Ellipsis => return Err(IndexError::EllipsisInNamed.into()),
                ix => self.set_op(&mut ops, axis, ix)?,
            }
        }
        self.apply(&ops)
    }

    /// [`isel_pos`](Self::isel_pos), consuming `self` and keeping its representation.
    ///
    /// # Errors
    /// As for [`isel_pos`](Self::isel_pos).
    pub fn isel_pos_move<I>(self, indexers: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: IntoIndexer,
    {
        let named = positional_to_named(&self.dims, indexers)?;
        self.isel_move(named)
    }

    /// Resolve `ix` against `axis` and record it in `ops`.
    fn set_op(&self, ops: &mut [Option<AxisOp>], axis: usize, ix: Indexer) -> Result<()> {
        let missing = || CoreError::DimCountMismatch {
            ndim: self.shape().len(),
            names: self.dims.len(),
        };
        let dim = self.dims.get(axis).ok_or_else(missing)?;
        let len = self.shape().get(axis).copied().ok_or_else(missing)?;
        let slot = ops.get_mut(axis).ok_or_else(missing)?;
        if slot.is_some() {
            return Err(IndexError::DuplicateDim(dim.clone()).into());
        }
        *slot = Some(match ix {
            Indexer::Int(i) => AxisOp::At(normalize_index(i, len, dim)?),
            Indexer::Slice(s) => {
                let si = s.indices(len)?;
                AxisOp::Slice(si.to_ndarray()?, si)
            }
            Indexer::Ellipsis => return Err(IndexError::EllipsisInNamed.into()),
        });
        Ok(())
    }

    /// Apply validated per-axis operations (composing the selection for lazy data).
    fn apply(self, ops: &[Option<AxisOp>]) -> Result<Self> {
        let Self {
            data,
            dims,
            unit,
            time_zone,
            encoding,
            attrs,
        } = self;
        let data = match data {
            Data::Loaded(mut data) => {
                // Slices first: they keep axis numbering intact.
                for (axis, op) in ops.iter().enumerate() {
                    if let Some(AxisOp::Slice(s, _)) = op {
                        data.slice_axis_inplace(axis, *s);
                    }
                }
                // Then drop integer-indexed axes from the highest down, so lower axis numbers stay valid.
                for (axis, op) in ops.iter().enumerate().rev() {
                    if let Some(AxisOp::At(i)) = op {
                        data = data.index_axis_move(axis, *i);
                    }
                }
                Data::Loaded(data)
            }
            Data::Lazy(l) => {
                let lops: Vec<Option<LazyOp>> = ops
                    .iter()
                    .map(|op| match op {
                        Some(AxisOp::At(i)) => Some(LazyOp::At(*i)),
                        Some(AxisOp::Slice(_, si)) => Some(LazyOp::Slice(*si)),
                        None => None,
                    })
                    .collect();
                Data::Lazy(l.isel(&lops)?)
            }
        };
        let dims = dims
            .into_iter()
            .zip(ops)
            .filter(|(_, op)| !matches!(op, Some(AxisOp::At(_))))
            .map(|(d, _)| d)
            .collect();
        Ok(Self {
            data,
            dims,
            unit,
            time_zone,
            encoding,
            attrs,
        })
    }
}

impl<S: Repr> VariableBase<S> {
    /// Copy the elements at `indices` along `dim` (fancy indexing). Negative indices count
    /// from the end; repeats and any order are allowed. Returns owned data.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] or [`IndexError::OutOfBounds`].
    pub fn take(&self, dim: &str, indices: &[isize]) -> Result<Variable> {
        let axis = self.axis(dim)?;
        let len = self
            .shape()
            .get(axis)
            .copied()
            .ok_or_else(|| CoreError::NoSuchDim(dim.into()))?;
        let name = Name::from(dim);
        let idx = indices
            .iter()
            .map(|i| normalize_index(*i, len, &name))
            .collect::<Result<Vec<_>, IndexError>>()?;
        if self.is_lazy() {
            // Still on disk: read the chosen positions only when (and where) needed.
            let src = crate::lazy::TakeSource::new(self.as_lazy_source()?, axis, idx)?;
            return Ok(VariableBase {
                data: Data::Lazy(LazyArray::new(std::sync::Arc::new(src))),
                dims: self.dims.clone(),
                unit: self.unit.clone(),
                time_zone: self.time_zone.clone(),
                encoding: self.encoding.clone(),
                attrs: self.attrs().clone(),
            });
        }
        Ok(VariableBase {
            data: Data::Loaded(self.typed()?.select_axis(axis, &idx)),
            dims: self.dims.clone(),
            unit: self.unit.clone(),
            time_zone: self.time_zone.clone(),
            encoding: self.encoding.clone(),
            attrs: self.attrs().clone(),
        })
    }
}

impl<S: ReprMut> VariableBase<S> {
    /// A mutable view of the whole variable.
    #[must_use]
    pub fn view_mut(&mut self) -> VariableViewMut<'_> {
        VariableBase {
            data: match &mut self.data {
                Data::Loaded(a) => Data::Loaded(a.view_mut()),
                Data::Lazy(l) => Data::Lazy(l.clone()),
            },
            dims: self.dims.clone(),
            unit: self.unit.clone(),
            time_zone: self.time_zone.clone(),
            encoding: self.encoding.clone(),
            attrs: self.attrs.borrow(),
        }
    }

    /// The data as a typed mutable ndarray view.
    ///
    /// # Errors
    /// [`CoreError::DTypeMismatch`] if the data is not of element type `T`.
    pub fn data_mut<T: Element>(&mut self) -> Result<ArrayViewMutD<'_, T>> {
        let found = self.dtype();
        let Data::Loaded(data) = &mut self.data else {
            return Err(CoreError::NotLoaded);
        };
        T::peek_mut(data)
            .map(S::view_mut)
            .ok_or(CoreError::DTypeMismatch {
                expected: T::DTYPE,
                found,
            })
    }

    /// [`isel`](Self::isel) returning a mutable view. Lazily loaded data stays lazy (and
    /// read-only until loaded).
    ///
    /// # Errors
    /// As for [`isel`](Self::isel).
    pub fn isel_mut<I, K, X>(&mut self, indexers: I) -> Result<VariableViewMut<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoIndexer,
    {
        self.view_mut().isel_move(indexers)
    }

    /// [`isel_pos`](Self::isel_pos) returning a mutable view.
    ///
    /// # Errors
    /// As for [`isel_pos`](Self::isel_pos).
    pub fn isel_pos_mut<I>(&mut self, indexers: I) -> Result<VariableViewMut<'_>>
    where
        I: IntoIterator,
        I::Item: IntoIndexer,
    {
        self.view_mut().isel_pos_move(indexers)
    }
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            dims: self.dims.clone(),
            unit: self.unit.clone(),
            time_zone: self.time_zone.clone(),
            encoding: self.encoding.clone(),
            attrs: self.attrs.clone(),
        }
    }
}

impl Clone for VariableView<'_> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            dims: self.dims.clone(),
            unit: self.unit.clone(),
            time_zone: self.time_zone.clone(),
            encoding: self.encoding.clone(),
            attrs: self.attrs,
        }
    }
}

impl<S: Repr> fmt::Debug for VariableBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("dims", &self.dims)
            .field("data", &self.data)
            .field("unit", &self.unit)
            .field("time_zone", &self.time_zone)
            .field("attrs", self.attrs())
            .finish()
    }
}

impl Variable {
    /// The variable with every dimension renamed by `f` at once (so names can be swapped).
    pub(crate) fn map_dims(mut self, f: impl Fn(&Name) -> Name) -> Result<Self> {
        let dims: Vec<Name> = self.dims.iter().map(&f).collect();
        for (i, d) in dims.iter().enumerate() {
            if dims.iter().skip(i.saturating_add(1)).any(|e| e == d) {
                return Err(CoreError::DuplicateDim(d.clone()));
            }
        }
        for (slot, d) in self.dims.iter_mut().zip(dims) {
            *slot = d;
        }
        Ok(self)
    }

    /// The variable with dimension `from` called `to` (the data untouched).
    pub(crate) fn rename_dim(mut self, from: &str, to: &Name) -> Result<Self> {
        if from != to.as_str() && self.dims.iter().any(|d| d == to) {
            return Err(CoreError::DuplicateDim(to.clone()));
        }
        for d in &mut self.dims {
            if d.as_str() == from {
                *d = to.clone();
            }
        }
        Ok(self)
    }
}

impl<S: Repr> VariableBase<S> {
    /// The values as `f64` (xarray's `.astype(float)`): numbers and booleans; NaN stays NaN.
    /// Lazy data is read.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedDType`] for strings and times; [`CoreError::Read`] if lazy
    /// data cannot be read.
    pub fn to_f64(&self) -> Result<ArrayD<f64>> {
        match self.typed()? {
            TypedArray::Bool(a) => Ok(a.mapv(|b| if b { 1.0 } else { 0.0 })),
            t => crate::ops::to_f64(&t, None),
        }
    }
}

fn permuted_as<T: crate::Element, S: Repr>(
    a: &TypedArray<S>,
    order: &[usize],
    _: crate::array::Marker<T>,
) -> Option<TypedArray<Owned>> {
    let x = T::peek(a)?;
    Some(T::wrap(
        x.view()
            .permuted_axes(IxDyn(order))
            .as_standard_layout()
            .into_owned(),
    ))
}

/// Positions in `dims` of each of `order`, which must name every dimension exactly once
/// (`"..."` stands for the dimensions not named, in their current order).
pub(crate) fn permutation(dims: &[Name], order: &[&str]) -> Result<Vec<usize>> {
    let named: Vec<&str> = order.iter().copied().filter(|d| *d != "...").collect();
    for d in &named {
        if !dims.iter().any(|x| x == d) {
            return Err(CoreError::NoSuchDim(Name::from(*d)));
        }
    }
    let mut out = Vec::with_capacity(dims.len());
    for d in order {
        if *d == "..." {
            out.extend(
                dims.iter()
                    .enumerate()
                    .filter(|(_, x)| !named.iter().any(|n| *x == *n))
                    .map(|(i, _)| i),
            );
        } else if let Some(i) = dims.iter().position(|x| x == d) {
            if out.contains(&i) {
                return Err(CoreError::DuplicateDim(Name::from(*d)));
            }
            out.push(i);
        }
    }
    if out.len() != dims.len() {
        return Err(CoreError::DimCountMismatch {
            ndim: dims.len(),
            names: out.len(),
        });
    }
    Ok(out)
}

impl<S: Repr> VariableBase<S> {
    /// The variable with its dimensions in `order` (data copied in the new layout). `"..."`
    /// stands for the dimensions not named; an empty `order` reverses them, as in xarray.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`], [`CoreError::DuplicateDim`], or
    /// [`CoreError::DimCountMismatch`] if `order` leaves dimensions out (without `"..."`);
    /// [`CoreError::Read`] if lazy data cannot be read.
    pub fn transpose(&self, order: &[&str]) -> Result<Variable> {
        let rev: Vec<&str>;
        let order = if order.is_empty() {
            rev = self.dims().iter().rev().map(Name::as_str).collect();
            &rev[..]
        } else {
            order
        };
        let perm = permutation(self.dims(), order)?;
        let dims: Vec<Name> = perm
            .iter()
            .filter_map(|i| self.dims().get(*i).cloned())
            .collect();
        let typed = self.typed()?;
        let data = crate::array::typed_dispatch!(&typed, x => permuted_as(&typed, &perm, crate::array::marker(x)))
            .ok_or_else(|| CoreError::Read("transpose".into()))?;
        Ok(VariableBase {
            data: Data::Loaded(data),
            dims: dims.into(),
            unit: self.unit.clone(),
            time_zone: self.time_zone.clone(),
            encoding: self.encoding.clone(),
            attrs: self.attrs().clone(),
        })
    }
}

/// `order` resolved against `dims` (`"..."` expanded, empty reversed), as names.
fn resolved<'a>(dims: &'a [Name], order: &[&str]) -> Result<Vec<&'a str>> {
    if order.is_empty() {
        return Ok(dims.iter().rev().map(Name::as_str).collect());
    }
    Ok(permutation(dims, order)?
        .into_iter()
        .filter_map(|i| dims.get(i).map(Name::as_str))
        .collect())
}

/// `variable` transposed into the relative order its dimensions have in `order`.
fn transposed_like<S: Repr>(v: &VariableBase<S>, order: &[&str]) -> Result<Variable> {
    let mine: Vec<&str> = order
        .iter()
        .copied()
        .filter(|d| v.dims().iter().any(|x| x == d))
        .collect();
    if v.dims().iter().map(Name::as_str).eq(mine.iter().copied()) {
        return Ok(v.to_owned());
    }
    v.transpose(&mine)
}

impl<S: Repr> crate::DataArrayBase<S> {
    /// The array with its dimensions in `order` (`"..."` for the rest; empty reverses).
    /// Multi-dimensional coordinates are transposed to match, as in xarray.
    ///
    /// # Errors
    /// As for [`VariableBase::transpose`].
    pub fn transpose(&self, order: &[&str]) -> Result<crate::DataArray> {
        let order = resolved(self.dims(), order)?;
        let var = self.variable().transpose(&order)?;
        let coords = self
            .coords()
            .iter()
            .map(|(k, c)| Ok((k.clone(), transposed_like(c, &order)?)))
            .collect::<Result<_>>()?;
        Ok(crate::DataArrayBase::from_parts(
            self.name().cloned(),
            var,
            coords,
        ))
    }
}

impl<S: Repr> crate::DatasetBase<S> {
    /// Every variable with its dimensions in the relative order they have in `order`
    /// (`"..."` for the rest; empty reverses the dataset's dimensions).
    ///
    /// # Errors
    /// As for [`VariableBase::transpose`].
    pub fn transpose(&self, order: &[&str]) -> Result<crate::Dataset> {
        let dims = self.dims();
        let order = resolved(&dims, order)?;
        let mut out = crate::Dataset::new();
        for (k, c) in self.coords() {
            out.add_coord(k.clone(), transposed_like(c, &order)?)?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(k.clone(), transposed_like(v, &order)?)?;
        }
        *out.attrs_mut() = self.attrs().clone();
        *out.encoding_mut() = self.encoding().clone();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};

    use super::*;
    use crate::PySlice;

    fn var() -> Variable {
        // time=2, lat=3, lon=4, values 0..24
        let a =
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(f64::from).collect()).unwrap();
        Variable::new(["time", "lat", "lon"], a)
            .unwrap()
            .with_attrs([("long_name", "test")])
    }

    #[test]
    fn construction_errors() {
        let a = ArrayD::<f64>::zeros(IxDyn(&[2, 2]));
        assert_eq!(
            Variable::new(["x"], a.clone()).unwrap_err(),
            CoreError::DimCountMismatch { ndim: 2, names: 1 }
        );
        assert_eq!(
            Variable::new(["x", "x"], a).unwrap_err(),
            CoreError::DuplicateDim("x".into())
        );
    }

    #[test]
    fn named_isel() {
        let v = var();
        let s = v
            .isel([
                ("time", Indexer::Int(-1)),
                ("lon", Indexer::Slice(PySlice::new(None, None, Some(-2)))),
            ])
            .unwrap();
        assert_eq!(s.dims(), &["lat", "lon"]);
        assert_eq!(s.shape(), &[3, 2]);
        let d = s.data::<f64>().unwrap();
        assert_eq!(
            d.iter().copied().collect::<Vec<_>>(),
            vec![15., 13., 19., 17., 23., 21.]
        );
        assert_eq!(s.attrs().get("long_name"), Some(&AttrValue::from("test")));
    }

    #[test]
    fn positional_isel_with_ellipsis() {
        let v = var();
        let s = v.isel_pos([Indexer::Ellipsis, Indexer::Int(0)]).unwrap();
        assert_eq!(s.dims(), &["time", "lat"]);
        assert_eq!(s.data::<f64>().unwrap()[[1, 2]], 20.0);

        let s = v
            .isel_pos([
                Indexer::Int(1),
                Indexer::Ellipsis,
                Indexer::Slice(PySlice::new(Some(1), Some(3), None)),
            ])
            .unwrap();
        assert_eq!(s.dims(), &["lat", "lon"]);
        assert_eq!(s.shape(), &[3, 2]);

        // Ranges work as indexers too.
        let s = v.isel_pos([0..1, 1..3]).unwrap();
        assert_eq!(s.shape(), &[1, 2, 4]);
    }

    #[test]
    fn isel_errors_do_not_panic() {
        let v = var();
        assert!(matches!(
            v.isel([("time", 2)]),
            Err(CoreError::Index(IndexError::OutOfBounds { .. }))
        ));
        assert_eq!(
            v.isel([("depth", 0)]).unwrap_err(),
            CoreError::NoSuchDim("depth".into())
        );
        assert_eq!(
            v.isel([("lat", 0), ("lat", 1)]).unwrap_err(),
            CoreError::Index(IndexError::DuplicateDim("lat".into()))
        );
        assert_eq!(
            v.isel([("lat", Indexer::Slice(PySlice::new(None, None, Some(0))))])
                .unwrap_err(),
            CoreError::Index(IndexError::ZeroStep)
        );
        assert_eq!(
            v.isel_pos([0, 0, 0, 0]).unwrap_err(),
            CoreError::Index(IndexError::TooManyIndexers { given: 4, ndim: 3 })
        );
        assert_eq!(
            v.isel_pos([Indexer::Ellipsis, Indexer::Ellipsis])
                .unwrap_err(),
            CoreError::Index(IndexError::MultipleEllipsis)
        );
        assert_eq!(
            v.isel([("lat", Indexer::Ellipsis)]).unwrap_err(),
            CoreError::Index(IndexError::EllipsisInNamed)
        );
        assert_eq!(
            v.data::<i32>().unwrap_err(),
            CoreError::DTypeMismatch {
                expected: DType::I32,
                found: DType::F64
            }
        );
    }

    #[test]
    fn mutation_through_view_mut_reaches_owner() {
        let mut v = var();
        {
            let mut s = v
                .isel_mut([
                    ("time", Indexer::Int(1)),
                    ("lat", Indexer::Slice(PySlice::new(None, None, Some(-1)))),
                ])
                .unwrap();
            // Nested isel on a mutable view, then write.
            let mut corner = s.isel_mut([("lat", 0), ("lon", -1)]).unwrap();
            assert_eq!(corner.ndim(), 0);
            corner.data_mut::<f64>().unwrap().fill(-1.0);
        }
        // time=1, lat reversed → lat index 0 of the view is lat=2; lon=-1 is lon=3.
        assert_eq!(v.data::<f64>().unwrap()[[1, 2, 3]], -1.0);
        assert_eq!(
            v.data::<f64>()
                .unwrap()
                .iter()
                .filter(|x| **x < 0.0)
                .count(),
            1
        );
    }

    #[test]
    fn views_and_owned_round_trip() {
        let v = var();
        let s = v.isel([("lat", 1..3)]).unwrap();
        let o = s.to_owned();
        assert_eq!(o.shape(), &[2, 2, 4]);
        assert_eq!(o.data::<f64>().unwrap(), s.data::<f64>().unwrap());
        let s2 = s.clone().isel_move([("time", 0)]).unwrap();
        assert_eq!(s2.dims(), &["lat", "lon"]);
    }
}
