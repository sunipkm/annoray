//! [`DataArrayBase`]: a variable with a name and coordinates.

use annoray_units::Unit;
use indexmap::IndexMap;
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

use crate::select::{isel_map, label_indexers, merge_sizes, named_indexers};
use crate::variable::positional_to_named;
use crate::{
    AttrValue, Attrs, CoreError, DType, Element, IntoIndexer, IntoLabelIndexer, Name, Owned, Repr,
    ReprMut, Result, Variable, VariableBase, VariableView, VariableViewMut, View, ViewMut,
};

/// Coordinates, held in the coordinate representation of `S`.
pub type Coords<S> = IndexMap<Name, VariableBase<<S as Repr>::Coord>>;

/// A labelled n-dimensional array: a [`VariableBase`] plus a name and coordinates.
///
/// Use the aliases [`DataArray`], [`DataArrayView`] and [`DataArrayViewMut`]. In a mutable
/// view only the data can be written; coordinates are shared views.
pub struct DataArrayBase<S: Repr> {
    pub(crate) name: Option<Name>,
    pub(crate) variable: VariableBase<S>,
    pub(crate) coords: Coords<S>,
}

/// A data array that owns its data and coordinates.
pub type DataArray = DataArrayBase<Owned>;
/// A shared view of a data array.
pub type DataArrayView<'a> = DataArrayBase<View<'a>>;
/// A view whose data can be modified; coordinates are read-only.
pub type DataArrayViewMut<'a> = DataArrayBase<ViewMut<'a>>;

impl DataArray {
    /// A data array with no name and no coordinates.
    #[must_use]
    pub fn new(variable: Variable) -> Self {
        Self {
            name: None,
            variable,
            coords: IndexMap::new(),
        }
    }

    /// A data array from dimension names and an array.
    ///
    /// # Errors
    /// As for [`Variable::new`].
    pub fn from_array<T, D>(dims: D, data: ArrayD<T>) -> Result<Self>
    where
        T: Element,
        D: IntoIterator,
        D::Item: Into<Name>,
    {
        Ok(Self::new(Variable::new(dims, data)?))
    }

    /// A 0-dimensional array holding `value` (e.g. a quantity, with
    /// [`with_unit`](Self::with_unit)).
    ///
    /// # Errors
    /// None in practice; the `Result` matches [`from_array`](Self::from_array).
    pub fn scalar<T: Element>(value: T) -> Result<Self> {
        Ok(Self::new(Variable::new(
            Vec::<Name>::new(),
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), value),
        )?))
    }

    /// Set the name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<Name>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the unit of the data.
    #[must_use]
    pub fn with_unit(mut self, unit: Unit) -> Self {
        self.variable = self.variable.with_unit(unit);
        self
    }

    /// Add attributes to the data.
    #[must_use]
    pub fn with_attrs<K: Into<Name>, V: Into<AttrValue>>(
        mut self,
        attrs: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        self.variable = self.variable.with_attrs(attrs);
        self
    }

    /// Add (or replace) a coordinate. Its dimensions must be dimensions of the data, with the
    /// same lengths.
    ///
    /// # Errors
    /// [`CoreError::CoordDimNotInData`] or [`CoreError::DimLengthMismatch`].
    pub fn with_coord(mut self, name: impl Into<Name>, coord: Variable) -> Result<Self> {
        let name = name.into();
        for d in coord.dims() {
            if !self.variable.dims().contains(d) {
                return Err(CoreError::CoordDimNotInData {
                    coord: name,
                    dim: d.clone(),
                });
            }
        }
        let mut sizes: IndexMap<Name, usize> =
            self.variable.sizes().map(|(d, n)| (d.clone(), n)).collect();
        merge_sizes(&mut sizes, &coord)?;
        self.coords.insert(name, coord);
        Ok(self)
    }

    /// A 1-D dimension coordinate named after its dimension: `with_dim_coord("time", array)`.
    ///
    /// # Errors
    /// As for [`with_coord`](Self::with_coord).
    pub fn with_dim_coord<T: Element>(
        self,
        dim: impl Into<Name>,
        values: ndarray::Array1<T>,
    ) -> Result<Self> {
        let dim = dim.into();
        let coord = Variable::new([dim.clone()], values.into_dyn())?;
        self.with_coord(dim, coord)
    }

    /// Mutable access to the attributes.
    pub fn attrs_mut(&mut self) -> &mut Attrs {
        self.variable.attrs_mut()
    }

    /// Replace the unit, returning the old one.
    pub fn set_unit(&mut self, unit: Option<Unit>) -> Option<Unit> {
        self.variable.set_unit(unit)
    }

    /// Split into name, variable and coordinates.
    #[must_use]
    pub fn into_parts(self) -> (Option<Name>, Variable, IndexMap<Name, Variable>) {
        (self.name, self.variable, self.coords)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// Assemble from parts already known to be consistent.
    pub(crate) fn from_parts(
        name: Option<Name>,
        variable: VariableBase<S>,
        coords: Coords<S>,
    ) -> Self {
        Self {
            name,
            variable,
            coords,
        }
    }

    /// The name, if any.
    #[must_use]
    pub fn name(&self) -> Option<&Name> {
        self.name.as_ref()
    }

    /// The underlying variable.
    #[must_use]
    pub fn variable(&self) -> &VariableBase<S> {
        &self.variable
    }

    /// Dimension names, in axis order.
    #[must_use]
    pub fn dims(&self) -> &[Name] {
        self.variable.dims()
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.variable.ndim()
    }

    /// Length of each axis.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.variable.shape()
    }

    /// `(name, length)` for each dimension.
    pub fn sizes(&self) -> impl Iterator<Item = (&Name, usize)> {
        self.variable.sizes()
    }

    /// Length of dimension `dim`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`].
    pub fn len_of(&self, dim: &str) -> Result<usize> {
        self.variable.len_of(dim)
    }

    /// Total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.variable.len()
    }

    /// Whether there are no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variable.is_empty()
    }

    /// Element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.variable.dtype()
    }

    /// Unit of the data, if known.
    #[must_use]
    pub fn unit(&self) -> Option<&Unit> {
        self.variable.unit()
    }

    /// Attributes of the data.
    #[must_use]
    pub fn attrs(&self) -> &Attrs {
        self.variable.attrs()
    }

    /// The data as a typed ndarray view.
    ///
    /// # Errors
    /// [`CoreError::DTypeMismatch`].
    pub fn data<T: Element>(&self) -> Result<ArrayViewD<'_, T>> {
        self.variable.data()
    }

    /// The values as `f64` (xarray's `.astype(float)`): numbers and booleans; NaN stays NaN.
    ///
    /// # Errors
    /// As for [`VariableBase::to_f64`].
    pub fn to_f64(&self) -> Result<ndarray::ArrayD<f64>> {
        self.variable.to_f64()
    }

    /// The coordinates.
    #[must_use]
    pub fn coords(&self) -> &Coords<S> {
        &self.coords
    }

    /// One coordinate.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`].
    pub fn coord(&self, name: &str) -> Result<&VariableBase<S::Coord>> {
        self.coords
            .get(name)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))
    }

    /// A shared view.
    #[must_use]
    pub fn view(&self) -> DataArrayView<'_> {
        DataArrayBase {
            name: self.name.clone(),
            variable: self.variable.view(),
            coords: self
                .coords
                .iter()
                .map(|(k, v)| (k.clone(), v.view()))
                .collect(),
        }
    }

    /// Copy into an owned data array.
    #[must_use]
    pub fn to_owned(&self) -> DataArray {
        DataArrayBase {
            name: self.name.clone(),
            variable: self.variable.to_owned(),
            coords: self
                .coords
                .iter()
                .map(|(k, v)| (k.clone(), v.to_owned()))
                .collect(),
        }
    }

    /// Select by dimension name and position, returning a view. Coordinates are sliced along.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] or an indexing error.
    pub fn isel<I, K, X>(&self, indexers: I) -> Result<DataArrayView<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoIndexer,
    {
        self.view().isel_move(indexers)
    }

    /// Select by position, returning a view.
    ///
    /// # Errors
    /// As for [`VariableBase::isel_pos`].
    pub fn isel_pos<I>(&self, indexers: I) -> Result<DataArrayView<'_>>
    where
        I: IntoIterator,
        I::Item: IntoIndexer,
    {
        self.view().isel_pos_move(indexers)
    }

    /// Select by coordinate label, returning a view. Slices are inclusive at both ends.
    ///
    /// # Errors
    /// [`CoreError::NoIndex`] if a dimension has no coordinate, [`CoreError::LabelNotFound`],
    /// [`CoreError::NotMonotonic`], [`CoreError::InvalidLabel`], or a unit error.
    pub fn sel<I, K, X>(&self, indexers: I) -> Result<DataArrayView<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoLabelIndexer,
    {
        self.view().sel_move(indexers)
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
        let ix = named_indexers(indexers, &|d| self.variable.dims().iter().any(|x| x == d))?;
        Ok(Self {
            name: self.name,
            variable: self.variable.isel_move(ix.iter().cloned())?,
            coords: isel_map(self.coords, &ix)?,
        })
    }

    /// [`isel_pos`](Self::isel_pos), consuming `self`.
    ///
    /// # Errors
    /// As for [`isel_pos`](Self::isel_pos).
    pub fn isel_pos_move<I>(self, indexers: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: IntoIndexer,
    {
        let named = positional_to_named(self.variable.dims(), indexers)?;
        self.isel_move(named)
    }

    /// [`sel`](Self::sel), consuming `self`.
    ///
    /// # Errors
    /// As for [`sel`](Self::sel).
    pub fn sel_move<I, K, X>(self, indexers: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoLabelIndexer,
    {
        let ix = label_indexers(
            indexers,
            &|d| self.variable.dims().iter().any(|x| x == d),
            &self.coords,
        )?;
        self.isel_move(ix)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// Copy the elements at `indices` along `dim` (fancy indexing); coordinates along `dim`
    /// are taken too. Returns owned data.
    ///
    /// # Errors
    /// As for [`VariableBase::take`].
    pub fn take(&self, dim: &str, indices: &[isize]) -> Result<DataArray> {
        let coords = self
            .coords
            .iter()
            .map(|(k, c)| {
                Ok((
                    k.clone(),
                    if c.dims().iter().any(|d| d == dim) {
                        c.take(dim, indices)?
                    } else {
                        c.to_owned()
                    },
                ))
            })
            .collect::<Result<_>>()?;
        Ok(DataArrayBase {
            name: self.name.clone(),
            variable: self.variable.take(dim, indices)?,
            coords,
        })
    }

    /// Select a list of labels along `dim` (fancy label indexing). Each item may be a label
    /// (exact match) or any [`LabelIndexer`](crate::LabelIndexer) (e.g. `nearest`).
    ///
    /// # Errors
    /// As for [`sel`](Self::sel).
    pub fn sel_take<L: IntoLabelIndexer>(
        &self,
        dim: &str,
        labels: impl IntoIterator<Item = L>,
    ) -> Result<DataArray> {
        let labels = labels
            .into_iter()
            .map(IntoLabelIndexer::into_label_indexer)
            .collect::<Result<Vec<_>>>()?;
        let pos = crate::select::label_positions(&self.coords, dim, labels)?;
        self.take(dim, &pos)
    }
}

impl<S: ReprMut> DataArrayBase<S> {
    /// A mutable view: data writable, coordinates shared.
    #[must_use]
    pub fn view_mut(&mut self) -> DataArrayViewMut<'_> {
        DataArrayBase {
            name: self.name.clone(),
            variable: self.variable.view_mut(),
            coords: self
                .coords
                .iter()
                .map(|(k, v)| (k.clone(), v.view()))
                .collect(),
        }
    }

    /// The data as a typed mutable ndarray view.
    ///
    /// # Errors
    /// [`CoreError::DTypeMismatch`].
    pub fn data_mut<T: Element>(&mut self) -> Result<ArrayViewMutD<'_, T>> {
        self.variable.data_mut()
    }

    /// The variable, mutably viewed.
    #[must_use]
    pub fn variable_mut(&mut self) -> VariableViewMut<'_> {
        self.variable.view_mut()
    }

    /// [`isel`](DataArrayBase::isel) returning a mutable view.
    ///
    /// # Errors
    /// As for [`isel`](DataArrayBase::isel).
    pub fn isel_mut<I, K, X>(&mut self, indexers: I) -> Result<DataArrayViewMut<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoIndexer,
    {
        self.view_mut().isel_move(indexers)
    }

    /// [`isel_pos`](DataArrayBase::isel_pos) returning a mutable view.
    ///
    /// # Errors
    /// As for [`isel_pos`](DataArrayBase::isel_pos).
    pub fn isel_pos_mut<I>(&mut self, indexers: I) -> Result<DataArrayViewMut<'_>>
    where
        I: IntoIterator,
        I::Item: IntoIndexer,
    {
        self.view_mut().isel_pos_move(indexers)
    }

    /// [`sel`](DataArrayBase::sel) returning a mutable view.
    ///
    /// # Errors
    /// As for [`sel`](DataArrayBase::sel).
    pub fn sel_mut<I, K, X>(&mut self, indexers: I) -> Result<DataArrayViewMut<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoLabelIndexer,
    {
        self.view_mut().sel_move(indexers)
    }
}

impl Clone for DataArray {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            variable: self.variable.clone(),
            coords: self.coords.clone(),
        }
    }
}

impl Clone for DataArrayView<'_> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            variable: self.variable.clone(),
            coords: self.coords.clone(),
        }
    }
}

impl From<Variable> for DataArray {
    fn from(v: Variable) -> Self {
        Self::new(v)
    }
}

impl<'a> From<VariableView<'a>> for DataArrayView<'a> {
    fn from(v: VariableView<'a>) -> Self {
        Self {
            name: None,
            variable: v,
            coords: IndexMap::new(),
        }
    }
}
