//! In-place assignment: data variables and coordinates added or replaced, names and
//! per-variable attributes changed (xarray's `ds[name] = …`, `ds.coords[name] = …`,
//! `da.name = …`).
//!
//! Each setter checks everything before changing anything: on an error the container is as
//! it was. Setters that replace variables return the variables they displaced, so that a
//! caller that lent out their memory (zero-copy views) can keep it alive.

use indexmap::IndexMap;

use crate::select::merge_sizes;
use crate::{Attrs, CoreError, DataArray, Dataset, Name, Result, Variable};

/// What adding an array does with one of its coordinates.
enum CoordAction {
    /// The dataset's coordinate stays (equal, or an index against a non-index).
    Keep,
    /// The array's coordinate replaces the dataset's (an index against a non-index).
    Replace,
    /// A new coordinate.
    Add,
}

impl Dataset {
    /// Dimension lengths of every variable except those in `skip`.
    fn sizes_without(&self, skip: &[&Name]) -> Result<IndexMap<Name, usize>> {
        let mut sizes = IndexMap::new();
        for (k, v) in self.coords().iter().chain(self.data_vars()) {
            if !skip.contains(&k) {
                merge_sizes(&mut sizes, v)?;
            }
        }
        Ok(sizes)
    }

    /// Set data variable `name` to `array` (xarray's `ds[name] = array`): added, or replacing
    /// the variable of that name; its coordinates merge in as in
    /// [`add_array`](Self::add_array). When `name` is a coordinate, the array's data replaces
    /// that coordinate. Returns the displaced variables.
    ///
    /// # Errors
    /// [`CoreError::NameConflict`] if a coordinate of the array conflicts with the dataset's;
    /// [`CoreError::DimLengthMismatch`] if a dimension's length disagrees with the rest of the
    /// dataset. The dataset is unchanged on error.
    pub fn set_array(&mut self, name: impl Into<Name>, array: DataArray) -> Result<Vec<Variable>> {
        let name = name.into();
        let (_, var, coords) = array.into_parts();
        if self.coords().contains_key(&name) {
            return Ok(self.set_coord_var(name, var)?.into_iter().collect());
        }
        let index = |n: &Name, v: &Variable| v.dims() == std::slice::from_ref(n);
        let mut actions = Vec::with_capacity(coords.len());
        for (cname, c) in &coords {
            if *cname == name || self.data_vars().contains_key(cname) {
                return Err(CoreError::NameConflict(cname.clone()));
            }
            actions.push(match self.coords().get(cname) {
                Some(e) if crate::ops::variables_equal(e, c) => CoordAction::Keep,
                Some(e) if index(cname, e) && !index(cname, c) => CoordAction::Keep,
                Some(e) if !index(cname, e) && index(cname, c) => CoordAction::Replace,
                Some(_) => return Err(CoreError::NameConflict(cname.clone())),
                None => CoordAction::Add,
            });
        }
        let mut skip: Vec<&Name> = vec![&name];
        skip.extend(
            coords
                .keys()
                .zip(&actions)
                .filter(|(_, a)| matches!(a, CoordAction::Replace))
                .map(|(k, _)| k),
        );
        let mut sizes = self.sizes_without(&skip)?;
        for ((_, c), a) in coords.iter().zip(&actions) {
            if !matches!(a, CoordAction::Keep) {
                merge_sizes(&mut sizes, c)?;
            }
        }
        merge_sizes(&mut sizes, &var)?;

        let (data_vars, own_coords) = self.parts_mut();
        let mut displaced = Vec::new();
        match data_vars.get_mut(&name) {
            Some(slot) => displaced.push(std::mem::replace(slot, var)),
            None => {
                data_vars.insert(name, var);
            }
        }
        for ((cname, c), a) in coords.into_iter().zip(actions) {
            match (a, own_coords.get_mut(&cname)) {
                (CoordAction::Keep, _) => {}
                (CoordAction::Replace, Some(slot)) => displaced.push(std::mem::replace(slot, c)),
                _ => {
                    own_coords.insert(cname, c);
                }
            }
        }
        Ok(displaced)
    }

    /// Set coordinate `name` (xarray's `ds.coords[name] = …`): added, or replacing the
    /// coordinate of that name, which is returned.
    ///
    /// # Errors
    /// [`CoreError::NameConflict`] if `name` is a data variable;
    /// [`CoreError::DimLengthMismatch`] if a dimension's length disagrees with the dataset.
    /// The dataset is unchanged on error.
    pub fn set_coord_var(
        &mut self,
        name: impl Into<Name>,
        coord: Variable,
    ) -> Result<Option<Variable>> {
        let name = name.into();
        if self.data_vars().contains_key(&name) {
            return Err(CoreError::NameConflict(name));
        }
        let mut sizes = self.sizes_without(&[&name])?;
        merge_sizes(&mut sizes, &coord)?;
        let (_, coords) = self.parts_mut();
        Ok(match coords.get_mut(&name) {
            Some(slot) => Some(std::mem::replace(slot, coord)),
            None => {
                coords.insert(name, coord);
                None
            }
        })
    }

    /// Turn a coordinate into a data variable (xarray's `reset_coords`). Index
    /// coordinates (named after their dimension) stay.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] if `name` is not a coordinate;
    /// [`CoreError::InvalidReduction`] for an index coordinate.
    pub fn reset_coord(&mut self, name: &str) -> Result<()> {
        let (data_vars, coords) = self.parts_mut();
        let c = coords
            .get(name)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))?;
        if c.dims().len() == 1 && c.dims().first().is_some_and(|d| d == name) {
            return Err(CoreError::InvalidReduction(format!(
                "cannot reset the index coordinate {name:?}"
            )));
        }
        if let Some((k, v)) = coords.shift_remove_entry(name) {
            data_vars.insert(k, v);
        }
        Ok(())
    }

    /// Mutable access to the attributes of variable (or coordinate) `name`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`].
    pub fn attrs_of_mut(&mut self, name: &str) -> Result<&mut Attrs> {
        let (data_vars, coords) = self.parts_mut();
        data_vars
            .get_mut(name)
            .or_else(|| coords.get_mut(name))
            .map(Variable::attrs_mut)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))
    }
}

impl DataArray {
    /// Set (or clear) the name.
    pub fn set_name(&mut self, name: Option<Name>) {
        self.name = name;
    }

    /// Set coordinate `name` (xarray's `da.coords[name] = …`): added, or replacing the
    /// coordinate of that name, which is returned. Its dimensions must be the data's, with
    /// the same lengths.
    ///
    /// # Errors
    /// [`CoreError::CoordDimNotInData`], [`CoreError::DimLengthMismatch`]. The array is
    /// unchanged on error.
    pub fn set_coord_var(
        &mut self,
        name: impl Into<Name>,
        coord: Variable,
    ) -> Result<Option<Variable>> {
        let name = name.into();
        if let Some(d) = coord.dims().iter().find(|d| !self.dims().contains(d)) {
            return Err(CoreError::CoordDimNotInData {
                coord: name,
                dim: d.clone(),
            });
        }
        let mut sizes: IndexMap<Name, usize> = self.sizes().map(|(d, n)| (d.clone(), n)).collect();
        merge_sizes(&mut sizes, &coord)?;
        Ok(match self.coords.get_mut(&name) {
            Some(slot) => Some(std::mem::replace(slot, coord)),
            None => {
                self.coords.insert(name, coord);
                None
            }
        })
    }

    /// Remove coordinate `name`, returning it.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`].
    pub fn remove_coord(&mut self, name: &str) -> Result<Variable> {
        self.coords
            .shift_remove(name)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))
    }

    /// Mutable access to the on-disk encoding of the data.
    pub fn encoding_mut(&mut self) -> &mut crate::Encoding {
        self.variable.encoding_mut()
    }

    /// Mutable access to the on-disk encoding of coordinate `name`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`].
    pub fn coord_encoding_mut(&mut self, name: &str) -> Result<&mut crate::Encoding> {
        self.coords
            .get_mut(name)
            .map(Variable::encoding_mut)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))
    }

    /// Mutable access to the attributes of coordinate `name`.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`].
    pub fn coord_attrs_mut(&mut self, name: &str) -> Result<&mut Attrs> {
        self.coords
            .get_mut(name)
            .map(Variable::attrs_mut)
            .ok_or_else(|| CoreError::NoSuchVariable(name.into()))
    }
}
