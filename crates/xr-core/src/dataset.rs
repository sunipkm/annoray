//! [`DatasetBase`]: named variables sharing dimensions and coordinates.

use std::borrow::Borrow;
use std::sync::Arc;

use indexmap::IndexMap;

use crate::dataarray::{Coords, DataArrayBase, DataArrayView, DataArrayViewMut};
use crate::select::{isel_map, label_indexers, merge_sizes, named_indexers};
use crate::variable::positional_to_named;
use crate::{
    AttrValue, Attrs, CoreError, DataArray, DatasetEncoding, IntoIndexer, IntoLabelIndexer, Name,
    Owned, Repr, ReprMut, Result, Variable, VariableBase, VariableViewMut, View, ViewMut,
};

/// A collection of data variables and coordinates whose shared dimensions agree in length.
///
/// Use the aliases [`Dataset`], [`DatasetView`] and [`DatasetViewMut`]. In a mutable view, data
/// variables can be written; coordinates are shared views and cannot.
pub struct DatasetBase<S: Repr> {
    data_vars: IndexMap<Name, VariableBase<S>>,
    coords: Coords<S>,
    attrs: S::Attrs,
    encoding: Arc<DatasetEncoding>,
}

/// A dataset that owns its variables.
pub type Dataset = DatasetBase<Owned>;
/// A shared view of a dataset.
pub type DatasetView<'a> = DatasetBase<View<'a>>;
/// A view whose data variables can be modified; coordinates and attributes are read-only.
pub type DatasetViewMut<'a> = DatasetBase<ViewMut<'a>>;

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}

impl Dataset {
    /// An empty dataset.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_vars: IndexMap::new(),
            coords: IndexMap::new(),
            attrs: Attrs::new(),
            encoding: Arc::default(),
        }
    }

    fn check_new(&self, name: &Name, var: &Variable) -> Result<()> {
        if self.data_vars.contains_key(name) || self.coords.contains_key(name) {
            return Err(CoreError::NameConflict(name.clone()));
        }
        let mut sizes = self.sizes();
        merge_sizes(&mut sizes, var)
    }

    /// Add a data variable.
    ///
    /// # Errors
    /// [`CoreError::NameConflict`] if the name is taken; [`CoreError::DimLengthMismatch`] if a
    /// dimension's length disagrees with the dataset.
    pub fn add_var(&mut self, name: impl Into<Name>, var: Variable) -> Result<()> {
        let name = name.into();
        self.check_new(&name, &var)?;
        self.data_vars.insert(name, var);
        Ok(())
    }

    /// Add a coordinate.
    ///
    /// # Errors
    /// As for [`add_var`](Self::add_var).
    pub fn add_coord(&mut self, name: impl Into<Name>, coord: Variable) -> Result<()> {
        let name = name.into();
        self.check_new(&name, &coord)?;
        self.coords.insert(name, coord);
        Ok(())
    }

    /// Builder form of [`add_var`](Self::add_var).
    ///
    /// # Errors
    /// As for [`add_var`](Self::add_var).
    pub fn with_var(mut self, name: impl Into<Name>, var: Variable) -> Result<Self> {
        self.add_var(name, var)?;
        Ok(self)
    }

    /// Builder form of [`add_coord`](Self::add_coord).
    ///
    /// # Errors
    /// As for [`add_var`](Self::add_var).
    pub fn with_coord(mut self, name: impl Into<Name>, coord: Variable) -> Result<Self> {
        self.add_coord(name, coord)?;
        Ok(self)
    }

    /// Add a data array: its data becomes a data variable (named `name`), its coordinates
    /// are merged in (they must agree with existing coordinates of the same name).
    ///
    /// # Errors
    /// [`CoreError::NameConflict`] if a coordinate of the same name differs.
    pub fn add_array(&mut self, name: impl Into<Name>, array: DataArray) -> Result<()> {
        let name = name.into();
        let (_, var, coords) = array.into_parts();
        for (cname, c) in coords {
            match self.coords.get(&cname) {
                Some(existing) if crate::ops::variables_equal(existing, &c) => {}
                Some(_) => return Err(CoreError::NameConflict(cname)),
                None => self.add_coord(cname, c)?,
            }
        }
        self.add_var(name, var)
    }

    /// Add global attributes.
    #[must_use]
    pub fn with_attrs<K: Into<Name>, V: Into<AttrValue>>(
        mut self,
        attrs: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        self.attrs
            .extend(attrs.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// How variable (or coordinate) `name` is to be stored on disk.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] if there is no such variable.
    pub fn encoding_of_mut(&mut self, name: &str) -> Result<&mut crate::Encoding> {
        self.data_vars
            .get_mut(name)
            .or_else(|| self.coords.get_mut(name))
            .map(Variable::encoding_mut)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))
    }

    /// Mutable access to the global attributes.
    pub fn attrs_mut(&mut self) -> &mut Attrs {
        &mut self.attrs
    }

    /// Mutable access to the dataset-level encoding.
    pub fn encoding_mut(&mut self) -> &mut DatasetEncoding {
        Arc::make_mut(&mut self.encoding)
    }

    /// Remove a data variable or coordinate.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`].
    pub fn remove(&mut self, name: &str) -> Result<Variable> {
        self.data_vars
            .shift_remove(name)
            .or_else(|| self.coords.shift_remove(name))
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))
    }

    /// Read every lazily loaded variable into memory (xarray's `load`).
    ///
    /// # Errors
    /// [`CoreError::Read`] if a source cannot be read.
    pub fn load(&mut self) -> Result<()> {
        for v in self.data_vars.values_mut().chain(self.coords.values_mut()) {
            v.load()?;
        }
        Ok(())
    }

    /// [`load`](Self::load), by value.
    ///
    /// # Errors
    /// [`CoreError::Read`] if a source cannot be read.
    pub fn loaded(mut self) -> Result<Self> {
        self.load()?;
        Ok(self)
    }

    /// Turn a data variable into a coordinate (xarray's `set_coords`).
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`].
    pub fn set_coord(&mut self, name: &str) -> Result<()> {
        let (k, v) = self
            .data_vars
            .shift_remove_entry(name)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))?;
        self.coords.insert(k, v);
        Ok(())
    }
}

impl<S: Repr> DatasetBase<S> {
    /// Length of every dimension, in order of first appearance (coordinates first).
    #[must_use]
    pub fn sizes(&self) -> IndexMap<Name, usize> {
        let mut sizes = IndexMap::new();
        let coord_sizes = self.coords.values().flat_map(|v| v.sizes());
        for (d, n) in coord_sizes.chain(self.data_vars.values().flat_map(|v| v.sizes())) {
            sizes.entry(d.clone()).or_insert(n);
        }
        sizes
    }

    /// Dimension names.
    #[must_use]
    pub fn dims(&self) -> Vec<Name> {
        self.sizes().into_keys().collect()
    }

    fn has_dim(&self, dim: &str) -> bool {
        self.coords
            .values()
            .any(|v| v.dims().iter().any(|d| d == dim))
            || self
                .data_vars
                .values()
                .any(|v| v.dims().iter().any(|d| d == dim))
    }

    /// Data variables.
    #[must_use]
    pub fn data_vars(&self) -> &IndexMap<Name, VariableBase<S>> {
        &self.data_vars
    }

    /// Coordinates.
    #[must_use]
    pub fn coords(&self) -> &Coords<S> {
        &self.coords
    }

    /// Dataset-level encoding (unlimited dimensions, file format hints).
    #[must_use]
    pub fn encoding(&self) -> &DatasetEncoding {
        &self.encoding
    }

    /// Global attributes.
    #[must_use]
    pub fn attrs(&self) -> &Attrs {
        self.attrs.borrow()
    }

    /// Whether a data variable or coordinate has this name.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.data_vars.contains_key(name) || self.coords.contains_key(name)
    }

    /// A data variable or coordinate as a data array view, with the coordinates that apply
    /// to it (those whose dimensions it has).
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`].
    pub fn get(&self, name: &str) -> Result<DataArrayView<'_>> {
        let var = match self.data_vars.get_key_value(name) {
            Some((_, v)) => v.view(),
            None => self
                .coords
                .get(name)
                .ok_or_else(|| CoreError::NoSuchVariable(name.into()))?
                .view(),
        };
        let coords = self
            .coords
            .iter()
            .filter(|(_, c)| c.dims().iter().all(|d| var.dims().contains(d)))
            .map(|(k, c)| (k.clone(), c.view()))
            .collect();
        Ok(DataArrayBase::from_parts(Some(name.into()), var, coords))
    }

    /// A shared view.
    #[must_use]
    pub fn view(&self) -> DatasetView<'_> {
        DatasetBase {
            data_vars: self
                .data_vars
                .iter()
                .map(|(k, v)| (k.clone(), v.view()))
                .collect(),
            coords: self
                .coords
                .iter()
                .map(|(k, v)| (k.clone(), v.view()))
                .collect(),
            attrs: self.attrs.borrow(),
            encoding: self.encoding.clone(),
        }
    }

    /// Copy into an owned dataset.
    #[must_use]
    pub fn to_owned(&self) -> Dataset {
        DatasetBase {
            data_vars: self
                .data_vars
                .iter()
                .map(|(k, v)| (k.clone(), v.to_owned()))
                .collect(),
            coords: self
                .coords
                .iter()
                .map(|(k, v)| (k.clone(), v.to_owned()))
                .collect(),
            attrs: self.attrs().clone(),
            encoding: self.encoding.clone(),
        }
    }

    /// Select by dimension name and position across all variables, returning a view.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] or an indexing error.
    pub fn isel<I, K, X>(&self, indexers: I) -> Result<DatasetView<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoIndexer,
    {
        self.view().isel_move(indexers)
    }

    /// Select by coordinate label across all variables, returning a view.
    ///
    /// # Errors
    /// As for [`DataArrayBase::sel`].
    pub fn sel<I, K, X>(&self, indexers: I) -> Result<DatasetView<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoLabelIndexer,
    {
        self.view().sel_move(indexers)
    }

    /// [`isel`](Self::isel), consuming `self`.
    ///
    /// # Errors
    /// As for [`isel`](Self::isel).
    pub fn isel_move<I, K, X>(self, indexers: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoIndexer,
    {
        let ix = named_indexers(indexers, &|d| self.has_dim(d))?;
        Ok(Self {
            data_vars: isel_map(self.data_vars, &ix)?,
            coords: isel_map(self.coords, &ix)?,
            attrs: self.attrs,
            encoding: self.encoding.clone(),
        })
    }

    /// Positional selection in the order of [`dims`](Self::dims), consuming `self`.
    ///
    /// # Errors
    /// As for [`VariableBase::isel_pos`].
    pub fn isel_pos_move<I>(self, indexers: I) -> Result<Self>
    where
        I: IntoIterator,
        I::Item: IntoIndexer,
    {
        let named = positional_to_named(&self.dims(), indexers)?;
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
        let ix = label_indexers(indexers, &|d| self.has_dim(d), &self.coords)?;
        self.isel_move(ix)
    }
}

impl<S: Repr> DatasetBase<S> {
    /// Copy the elements at `indices` along `dim` in every variable that has it (fancy
    /// indexing). Returns an owned dataset.
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] or [`IndexError::OutOfBounds`](crate::IndexError::OutOfBounds).
    pub fn take(&self, dim: &str, indices: &[isize]) -> Result<Dataset> {
        if !self.has_dim(dim) {
            return Err(CoreError::NoSuchDim(dim.into()));
        }
        let take = |v: &VariableBase<_>| -> Result<Variable> {
            if v.dims().iter().any(|d| d == dim) {
                v.take(dim, indices)
            } else {
                Ok(v.to_owned())
            }
        };
        Ok(DatasetBase {
            data_vars: self
                .data_vars
                .iter()
                .map(|(k, v)| Ok((k.clone(), take(v)?)))
                .collect::<Result<_>>()?,
            coords: self
                .coords
                .iter()
                .map(|(k, v)| {
                    Ok((
                        k.clone(),
                        if v.dims().iter().any(|d| d == dim) {
                            v.take(dim, indices)?
                        } else {
                            v.to_owned()
                        },
                    ))
                })
                .collect::<Result<_>>()?,
            attrs: self.attrs().clone(),
            encoding: self.encoding.clone(),
        })
    }

    /// Select a list of labels along `dim` in every variable (fancy label indexing).
    ///
    /// # Errors
    /// As for [`sel`](Self::sel).
    pub fn sel_take<L: IntoLabelIndexer>(
        &self,
        dim: &str,
        labels: impl IntoIterator<Item = L>,
    ) -> Result<Dataset> {
        let labels = labels
            .into_iter()
            .map(IntoLabelIndexer::into_label_indexer)
            .collect::<Result<Vec<_>>>()?;
        let pos = crate::select::label_positions(&self.coords, dim, labels)?;
        self.take(dim, &pos)
    }
}

impl<S: ReprMut> DatasetBase<S> {
    /// A mutable view: data variables writable, coordinates and attributes shared.
    #[must_use]
    pub fn view_mut(&mut self) -> DatasetViewMut<'_> {
        DatasetBase {
            data_vars: self
                .data_vars
                .iter_mut()
                .map(|(k, v)| (k.clone(), v.view_mut()))
                .collect(),
            coords: self
                .coords
                .iter()
                .map(|(k, v)| (k.clone(), v.view()))
                .collect(),
            attrs: self.attrs.borrow(),
            encoding: self.encoding.clone(),
        }
    }

    /// One data variable, mutably viewed.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] (coordinates cannot be modified this way).
    pub fn data_var_mut(&mut self, name: &str) -> Result<VariableViewMut<'_>> {
        self.data_vars
            .get_mut(name)
            .map(VariableBase::view_mut)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))
    }

    /// A data variable as a mutable data array view (data writable, coordinates shared).
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`].
    pub fn get_mut(&mut self, name: &str) -> Result<DataArrayViewMut<'_>> {
        let var = self
            .data_vars
            .get_mut(name)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))?
            .view_mut();
        let coords = self
            .coords
            .iter()
            .filter(|(_, c)| c.dims().iter().all(|d| var.dims().contains(d)))
            .map(|(k, c)| (k.clone(), c.view()))
            .collect();
        Ok(DataArrayBase::from_parts(Some(name.into()), var, coords))
    }

    /// [`isel`](DatasetBase::isel) returning a mutable view.
    ///
    /// # Errors
    /// As for [`isel`](DatasetBase::isel).
    pub fn isel_mut<I, K, X>(&mut self, indexers: I) -> Result<DatasetViewMut<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoIndexer,
    {
        self.view_mut().isel_move(indexers)
    }

    /// [`sel`](DatasetBase::sel) returning a mutable view.
    ///
    /// # Errors
    /// As for [`sel`](DatasetBase::sel).
    pub fn sel_mut<I, K, X>(&mut self, indexers: I) -> Result<DatasetViewMut<'_>>
    where
        I: IntoIterator<Item = (K, X)>,
        K: AsRef<str>,
        X: IntoLabelIndexer,
    {
        self.view_mut().sel_move(indexers)
    }
}

impl Clone for Dataset {
    fn clone(&self) -> Self {
        Self {
            data_vars: self.data_vars.clone(),
            coords: self.coords.clone(),
            attrs: self.attrs.clone(),
            encoding: self.encoding.clone(),
        }
    }
}

impl Clone for DatasetView<'_> {
    fn clone(&self) -> Self {
        Self {
            data_vars: self.data_vars.clone(),
            coords: self.coords.clone(),
            attrs: self.attrs,
            encoding: self.encoding.clone(),
        }
    }
}
