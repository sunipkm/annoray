//! Renaming a dataset's variables and dimensions (xarray's `rename`, `rename_vars`,
//! `rename_dims`), adding a dimension to every data variable (`expand_dims`), and dropping a
//! data array's coordinates (`drop_vars`).
//!
//! Renames apply all at once, so two names can be swapped. A coordinate renamed away from
//! its dimension's name stops being that dimension's index (xarray keeps the index under the
//! new name); [`DatasetBase::swap_dims`] makes it the index again.

use indexmap::IndexMap;

use crate::{
    CoreError, DataArray, DataArrayBase, Dataset, DatasetBase, Name, Repr, Result, Variable,
};

/// The new name of `name` under `pairs`, or `name` itself.
fn renamed(pairs: &[(&str, &str)], name: &Name) -> Name {
    pairs
        .iter()
        .find(|(old, _)| name == *old)
        .map_or_else(|| name.clone(), |(_, new)| Name::from(*new))
}

impl<S: Repr> DatasetBase<S> {
    /// Variables (data variables and coordinates) and dimensions renamed, `old → new` (a
    /// name may be both a variable and a dimension; both are renamed).
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] for a name that is neither; [`CoreError::NameConflict`]
    /// or [`CoreError::DuplicateDim`] if a new name is taken.
    pub fn rename(&self, pairs: &[(&str, &str)]) -> Result<Dataset> {
        let dims = self.dims();
        for (old, _) in pairs {
            if !self.contains(old) && !dims.iter().any(|d| d == *old) {
                return Err(CoreError::NoSuchVariable(Name::from(*old)));
            }
        }
        let (vars, dim_pairs): (Vec<_>, Vec<_>) = (
            pairs
                .iter()
                .copied()
                .filter(|(o, _)| self.contains(o))
                .collect(),
            pairs
                .iter()
                .copied()
                .filter(|(o, _)| dims.iter().any(|d| d == *o))
                .collect(),
        );
        self.renamed(&vars, &dim_pairs)
    }

    /// Variables (data variables and coordinates) renamed, `old → new`; dimensions kept.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] for a name that is not a variable;
    /// [`CoreError::NameConflict`] if a new name is taken.
    pub fn rename_vars(&self, pairs: &[(&str, &str)]) -> Result<Dataset> {
        if let Some((old, _)) = pairs.iter().find(|(o, _)| !self.contains(o)) {
            return Err(CoreError::NoSuchVariable(Name::from(*old)));
        }
        self.renamed(pairs, &[])
    }

    /// Dimensions renamed, `old → new`; variable names kept (a dimension coordinate stays,
    /// along the renamed dimension).
    ///
    /// # Errors
    /// [`CoreError::NoSuchDim`] for a name that is not a dimension;
    /// [`CoreError::DuplicateDim`] if a new name is already a dimension.
    pub fn rename_dims(&self, pairs: &[(&str, &str)]) -> Result<Dataset> {
        let dims = self.dims();
        if let Some((old, _)) = pairs.iter().find(|(o, _)| !dims.iter().any(|d| d == *o)) {
            return Err(CoreError::NoSuchDim(Name::from(*old)));
        }
        self.renamed(&[], pairs)
    }

    fn renamed(&self, vars: &[(&str, &str)], dims: &[(&str, &str)]) -> Result<Dataset> {
        // A new dimension name may only be one that is renamed away.
        for d in self.dims() {
            let after = renamed(dims, &d);
            if after != d {
                continue;
            }
            if let Some((old, _)) = dims.iter().find(|(o, n)| *n == d.as_str() && *o != *n) {
                return Err(CoreError::InvalidReduction(format!(
                    "cannot rename dimension {old:?} to {d:?}: {d:?} is already a dimension"
                )));
            }
        }
        let mut seen: IndexMap<Name, ()> = IndexMap::new();
        for k in self.coords().keys().chain(self.data_vars().keys()) {
            if seen.insert(renamed(vars, k), ()).is_some() {
                return Err(CoreError::NameConflict(renamed(vars, k)));
            }
        }
        let map = |v: Variable| v.map_dims(|d| renamed(dims, d));
        let mut out = Dataset::new();
        for (k, c) in self.coords() {
            out.add_coord(renamed(vars, k), map(c.to_owned())?)?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(renamed(vars, k), map(v.to_owned())?)?;
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }

    /// A new length-1 dimension `dim` in every data variable, at position `axis` (counted
    /// from the end when negative); coordinates are unchanged, except a scalar coordinate
    /// named `dim`, which becomes its coordinate.
    ///
    /// # Errors
    /// [`CoreError::DuplicateDim`] if the dataset has the dimension;
    /// [`CoreError::InvalidReduction`] for an axis out of range for some variable.
    pub fn expand_dims(&self, dim: &str, axis: isize) -> Result<Dataset> {
        let name = Name::from(dim);
        if self.dims().contains(&name) {
            return Err(CoreError::DuplicateDim(name));
        }
        let mut out = Dataset::new();
        for (k, c) in self.coords() {
            let c = if *k == name && c.ndim() == 0 {
                c.expand_dims(dim, 0)?
            } else {
                c.to_owned()
            };
            out.add_coord(k.clone(), c)?;
        }
        for (k, v) in self.data_vars() {
            out.add_var(k.clone(), v.expand_dims(dim, axis)?)?;
        }
        *out.attrs_mut() = self.attrs().clone();
        Ok(out)
    }
}

impl<S: Repr> DataArrayBase<S> {
    /// The data array without the named coordinates (xarray's `DataArray.drop_vars`); a
    /// dimension coordinate may be dropped, leaving the dimension without labels.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] for a name that is not a coordinate.
    pub fn drop_coords(&self, names: &[&str]) -> Result<DataArray> {
        if let Some(n) = names.iter().find(|n| !self.coords().contains_key(**n)) {
            return Err(CoreError::NoSuchVariable(Name::from(*n)));
        }
        let coords = self
            .coords()
            .iter()
            .filter(|(k, _)| !names.contains(&k.as_str()))
            .map(|(k, c)| (k.clone(), c.to_owned()))
            .collect();
        Ok(DataArrayBase::from_parts(
            self.name().cloned(),
            self.variable().to_owned(),
            coords,
        ))
    }
}
