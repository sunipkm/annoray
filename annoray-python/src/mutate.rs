//! Assignment on DataArray and Dataset objects (`ds["v"] = …`, `ds.coords["x"] = …`,
//! `da.name = …`, writes to `.attrs`), and the bookkeeping that keeps it memory-safe and
//! xarray-like.
//!
//! - Sharing: data variables, coordinates and child arrays hold reference-counted buffers, and
//!   a numpy view (`.values`, `to_xarray()`) holds its own reference, so a view outlives any
//!   replacement of the variable it came from. Data buffers are never written in place: values
//!   are immutable from Python.
//! - Parents: a data array taken from a dataset (`ds["t"]`, `ds.t`, `ds.data_vars`, …) or a
//!   coordinate taken from a data array shares its buffer, and remembers where it came from.
//!   Writes to its attributes also reach the source variable, as xarray's shared variables do,
//!   as long as the source has not replaced that variable since (a generation count per name).

use std::collections::HashMap;
use std::sync::Mutex;

use annoray::{Attrs, DataArray, Name, Unit};
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{OrPy, attrs_to_py};
use crate::native::{
    PyDataArray, PyDataset, new_child_da, new_da, spec_from_py, spec_with, units_attr,
};

/// How many times each variable of an object has been replaced; see the module notes.
#[derive(Default)]
pub(crate) struct Generations(Mutex<HashMap<Name, u64>>);

impl Generations {
    /// The generation of variable `name` (bumped whenever it is replaced or removed).
    pub(crate) fn generation(&self, name: &str) -> u64 {
        self.0
            .lock()
            .ok()
            .and_then(|g| g.get(name).copied())
            .unwrap_or(0)
    }

    /// Mark variables as replaced: copies taken earlier no longer write through.
    pub(crate) fn bump<'a>(&self, names: impl IntoIterator<Item = &'a str>) {
        if let Ok(mut g) = self.0.lock() {
            for n in names {
                let e = g.entry(Name::from(n)).or_insert(0);
                *e = e.saturating_add(1);
            }
        }
    }
}

/// Where a data array was taken from: variable `name` of `owner` (a Dataset, or a DataArray
/// for its coordinates), at `generation`.
pub(crate) struct Source {
    pub(crate) owner: Py<PyAny>,
    pub(crate) name: Name,
    pub(crate) generation: u64,
}

impl Source {
    /// A link to variable `name` of `owner` as it is now.
    pub(crate) fn of(owner: &Bound<'_, PyAny>, name: &str) -> Option<Self> {
        let generation = if let Ok(ds) = owner.cast::<PyDataset>() {
            ds.borrow().generations.generation(name)
        } else if let Ok(da) = owner.cast::<PyDataArray>() {
            da.borrow().generations.generation(name)
        } else {
            return None;
        };
        Some(Self {
            owner: owner.clone().unbind(),
            name: Name::from(name),
            generation,
        })
    }

    /// Give the source variable `attrs` and `unit`, if it is still the one this array came
    /// from.
    pub(crate) fn write_attrs(
        &self,
        py: Python<'_>,
        attrs: &Attrs,
        unit: Option<&Unit>,
    ) -> PyResult<()> {
        let owner = self.owner.bind(py);
        if let Ok(ds) = owner.cast::<PyDataset>() {
            let mut ds = ds.try_borrow_mut()?;
            if ds.generations.generation(&self.name) == self.generation
                && let Ok(a) = ds.inner.attrs_of_mut(&self.name)
            {
                a.clone_from(attrs);
                ds.inner.set_unit_of(&self.name, unit.cloned()).py_err()?;
            }
        } else if let Ok(da) = owner.cast::<PyDataArray>() {
            let mut da = da.try_borrow_mut()?;
            if da.generations.generation(&self.name) == self.generation
                && let Ok(a) = da.inner.coord_attrs_mut(&self.name)
            {
                a.clone_from(attrs);
                da.inner
                    .set_coord_unit(&self.name, unit.cloned())
                    .py_err()?;
            }
        }
        Ok(())
    }
}

impl Source {
    /// Give the source variable `enc`, if it is still the one this copy came from.
    pub(crate) fn write_encoding(&self, py: Python<'_>, enc: &annoray::Encoding) -> PyResult<()> {
        let owner = self.owner.bind(py);
        if let Ok(ds) = owner.cast::<PyDataset>() {
            let mut ds = ds.try_borrow_mut()?;
            if ds.generations.generation(&self.name) == self.generation
                && let Ok(e) = ds.inner.encoding_of_mut(&self.name)
            {
                e.clone_from(enc);
            }
        } else if let Ok(da) = owner.cast::<PyDataArray>() {
            let mut da = da.try_borrow_mut()?;
            if da.generations.generation(&self.name) == self.generation
                && let Ok(e) = da.inner.coord_encoding_mut(&self.name)
            {
                e.clone_from(enc);
            }
        }
        Ok(())
    }
}

/// `enc` with the keys of `changed` (xarray's names) set and the `removed` ones reset.
fn edited_encoding(
    enc: &annoray::Encoding,
    changed: &Bound<'_, PyAny>,
    removed: Vec<String>,
) -> PyResult<annoray::Encoding> {
    let mut out = enc.clone();
    for k in removed {
        crate::raw::clear_encoding_key(&mut out, &k)?;
    }
    crate::raw::update_encoding(&mut out, changed.cast::<PyDict>()?)?;
    Ok(out)
}

#[pymethods]
impl PyDataArray {
    /// How the data is (or will be) stored on disk, in xarray's encoding names
    /// (`dtype`, `_FillValue`, `scale_factor`, `units`/`calendar` of times, `zlib`,
    /// `complevel`, `chunksizes`, `compression`, …); changes apply to this array (and to
    /// the dataset variable it came from).
    #[getter]
    fn encoding<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let d = crate::raw::encoding_to_py(slf.py(), slf.borrow().inner.variable().encoding())?;
        live_dict(slf.as_any(), "EncodingDict", d.into_any())
    }

    #[setter(encoding)]
    fn set_encoding_py(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let enc = crate::raw::encoding_from_py(value.cast::<PyDict>()?)?;
        Self::replace_encoding(slf, enc)
    }

    fn _plain_encoding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        crate::raw::encoding_to_py(py, self.inner.variable().encoding())
    }

    fn _edit_encoding(
        slf: &Bound<'_, Self>,
        changed: &Bound<'_, PyAny>,
        removed: Vec<String>,
    ) -> PyResult<()> {
        let enc = edited_encoding(slf.borrow().inner.variable().encoding(), changed, removed)?;
        Self::replace_encoding(slf, enc)
    }
}

impl PyDataArray {
    fn replace_encoding(slf: &Bound<'_, Self>, enc: annoray::Encoding) -> PyResult<()> {
        slf.try_borrow_mut()?.inner.encoding_mut().clone_from(&enc);
        if let Some(p) = &slf.borrow().parent {
            p.write_encoding(slf.py(), &enc)?;
        }
        Ok(())
    }
}

#[pymethods]
impl PyDataset {
    /// Dataset-level storage settings (`unlimited_dims`); changes apply to this dataset.
    #[getter]
    fn encoding<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let d = Self::plain_encoding(&slf.borrow(), slf.py())?;
        live_dict(slf.as_any(), "EncodingDict", d.into_any())
    }

    #[setter(encoding)]
    fn set_encoding_py(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.encoding_mut().unlimited_dims.clear();
        self._edit_encoding(value, Vec::new())
    }

    fn _plain_encoding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        Self::plain_encoding(self, py)
    }

    fn _edit_encoding(&mut self, changed: &Bound<'_, PyAny>, removed: Vec<String>) -> PyResult<()> {
        for k in removed {
            if k == "unlimited_dims" {
                self.inner.encoding_mut().unlimited_dims.clear();
            }
        }
        for (k, v) in crate::native::dict_iter(Some(changed))? {
            if k == "unlimited_dims" {
                let dims: Vec<String> = match v.extract::<String>() {
                    Ok(d) => vec![d],
                    Err(_) => v.extract()?,
                };
                self.inner.encoding_mut().unlimited_dims =
                    dims.into_iter().map(Name::from).collect();
            }
        }
        Ok(())
    }
}

impl PyDataset {
    fn plain_encoding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let u = &self.inner.encoding().unlimited_dims;
        if !u.is_empty() {
            d.set_item(
                "unlimited_dims",
                u.iter().map(Name::as_str).collect::<Vec<_>>(),
            )?;
        }
        Ok(d)
    }
}

/// `cls(owner, data)` from `annoray._mapping`: a dict that writes changes back to `owner`.
pub(crate) fn live_dict<'py>(
    owner: &Bound<'py, PyAny>,
    cls: &str,
    data: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    owner
        .py()
        .import("annoray._mapping")?
        .getattr(cls)?
        .call1((owner, data))
}

/// `attrs` with the keys of `changed` set (converted from Python) and the `removed` keys
/// dropped; the other attributes are kept as they are, never converted, so their exact
/// types (a file's `int16`, `float32`, string lists) survive any edit.
fn edited(attrs: &Attrs, changed: &Bound<'_, PyAny>, removed: Vec<String>) -> PyResult<Attrs> {
    let mut out = attrs.clone();
    for k in removed {
        out.shift_remove(k.as_str());
    }
    crate::native::set_attrs(&mut out, Some(changed))?;
    Ok(out)
}

/// A data array's `attrs` and `unit` after an edit of its attributes, where `"units"` stands
/// for the unit: a string that parses becomes the unit (the values are unchanged); any other
/// value stays an attribute, and the array then has no unit. Removing `"units"` clears both.
pub(crate) fn edited_with_unit(
    attrs: &Attrs,
    unit: Option<&Unit>,
    changed: &Bound<'_, PyAny>,
    removed: Vec<String>,
) -> PyResult<(Attrs, Option<Unit>)> {
    let mut out = attrs.clone();
    let mut unit = unit.cloned();
    for k in removed {
        if k == "units" {
            unit = None;
        }
        out.shift_remove(k.as_str());
    }
    for (k, v) in crate::native::dict_iter(Some(changed))? {
        if k == "units" {
            out.shift_remove(k.as_str());
            unit = match v.extract::<String>() {
                Ok(text) => units_attr(changed.py(), &text)?,
                Err(_) => None,
            };
            if unit.is_some() {
                continue;
            }
        }
        let value = crate::convert::named_attr_from_py(&k, &v)?;
        out.insert(k.into(), value);
    }
    Ok((out, unit))
}

impl PyDataArray {
    /// Replace the attributes and unit, writing through to the source variable if there is
    /// one.
    pub(crate) fn replace_attrs(
        slf: &Bound<'_, Self>,
        attrs: Attrs,
        unit: Option<Unit>,
    ) -> PyResult<()> {
        {
            let mut me = slf.try_borrow_mut()?;
            me.inner.attrs_mut().clone_from(&attrs);
            me.inner.set_unit(unit.clone());
        }
        if let Some(p) = &slf.borrow().parent {
            p.write_attrs(slf.py(), &attrs, unit.as_ref())?;
        }
        Ok(())
    }

    /// The attributes as `.attrs` shows them: with the unit as `"units"`.
    pub(crate) fn attrs_with_units<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = attrs_to_py(py, self.inner.attrs())?;
        if let Some(u) = self.inner.unit() {
            d.set_item("units", u.to_string())?;
        }
        Ok(d)
    }

    /// The coordinates as data arrays linked to this one.
    pub(crate) fn plain_coords<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let names: Vec<String> = slf
            .borrow()
            .inner
            .coords()
            .keys()
            .map(ToString::to_string)
            .collect();
        let d = PyDict::new(slf.py());
        for n in names {
            let c = slf.borrow().coord_array(&n)?;
            d.set_item(&n, new_child_da(slf.as_any(), &n, c)?)?;
        }
        Ok(d)
    }

    fn set_coord_value(
        slf: &Bound<'_, Self>,
        name: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let var = spec_from_py(slf.py(), name, value)?;
        let mut me = slf.try_borrow_mut()?;
        me.inner.set_coord_var(name, var).py_err()?;
        me.generations.bump([name]);
        Ok(())
    }
}

#[pymethods]
impl PyDataArray {
    fn _edit_attrs(
        slf: &Bound<'_, Self>,
        changed: &Bound<'_, PyAny>,
        removed: Vec<String>,
    ) -> PyResult<()> {
        let (attrs, unit) = {
            let me = slf.borrow();
            edited_with_unit(me.inner.attrs(), me.inner.unit(), changed, removed)?
        };
        Self::replace_attrs(slf, attrs, unit)
    }

    fn _plain_attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.attrs_with_units(py)
    }

    /// The attributes other than the unit, as they are stored.
    fn _stored_attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        attrs_to_py(py, self.inner.attrs())
    }

    fn _plain_coords<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        Self::plain_coords(slf)
    }

    fn _set_coord(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::set_coord_value(slf, name, value)
    }

    fn _del_coord(&mut self, name: &str) -> PyResult<()> {
        self.inner.remove_coord(name).py_err()?;
        self.generations.bump([name]);
        Ok(())
    }

    /// `da["x"] = …` sets coordinate `x`. Values are immutable: other keys are refused (use
    /// `put`/`put_sel`, which return an updated copy, or `where`).
    fn __setitem__(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if let Ok(name) = key.extract::<String>() {
            return Self::set_coord_value(slf, &name, value);
        }
        Err(PyTypeError::new_err(
            "annoray arrays' values are immutable: da = da.put(value, x=...) (positions) or \
             da.put_sel(value, x=...) (labels) returns an updated copy; see also \
             da.where(cond, other) (da[\"name\"] = ... sets a coordinate)",
        ))
    }

    /// `del da["x"]` removes coordinate `x`.
    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self._del_coord(key)
    }
}

impl PyDataset {
    /// Variables `names` as data arrays linked to this dataset.
    pub(crate) fn children<'py>(
        slf: &Bound<'py, Self>,
        names: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(slf.py());
        for n in names {
            let a = slf.borrow().array(&n)?;
            d.set_item(&n, new_child_da(slf.as_any(), &n, a)?)?;
        }
        Ok(d)
    }

    /// The coordinates as data arrays linked to this dataset.
    pub(crate) fn plain_coords<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let names: Vec<String> = slf
            .borrow()
            .inner
            .coords()
            .keys()
            .map(ToString::to_string)
            .collect();
        Self::children(slf, names)
    }

    /// `ds[key] = value`: a data array, `(dims, data[, attrs])`, or 1-D data along `key`; numpy
    /// arrays kept as they are without `copy`.
    fn set_value(
        slf: &Bound<'_, Self>,
        key: &str,
        value: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<()> {
        let array = match value.cast::<PyDataArray>() {
            Ok(da) => da.borrow().inner.clone(),
            Err(_) => DataArray::new(spec_with(slf.py(), key, value, copy)?),
        }
        .with_name(key);
        let coords: Vec<String> = array.coords().keys().map(ToString::to_string).collect();
        let mut me = slf.try_borrow_mut()?;
        me.inner.set_array(key, array).py_err()?;
        me.generations
            .bump(std::iter::once(key).chain(coords.iter().map(String::as_str)));
        Ok(())
    }
}

#[pymethods]
impl PyDataset {
    /// `ds["v"] = value`: add or replace a data variable (a DataArray, whose coordinates merge
    /// in; `(dims, data[, attrs])`; or 1-D data along a dimension of that name). Assigning
    /// to a coordinate's name replaces the coordinate.
    fn __setitem__(slf: &Bound<'_, Self>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::set_value(slf, key, value, true)
    }

    /// `del ds["v"]` removes a data variable or coordinate.
    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self.inner.remove(key).py_err()?;
        self.generations.bump([key]);
        Ok(())
    }

    /// Add or replace variables in place from a mapping or another dataset (its data
    /// variables, with their coordinates); returns this dataset. Values are copied unless
    /// `copy=False`, as for the `Dataset` constructor.
    #[pyo3(signature = (other, *, copy=true))]
    fn update<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        copy: bool,
    ) -> PyResult<Bound<'py, Self>> {
        if let Ok(ds) = other.cast::<PyDataset>() {
            let arrays: Vec<(String, DataArray)> = {
                let o = ds.borrow();
                o.inner
                    .data_vars()
                    .keys()
                    .map(|k| Ok((k.to_string(), o.inner.get(k).py_err()?.to_owned())))
                    .collect::<PyResult<_>>()?
            };
            for (k, a) in arrays {
                let v = new_da(slf.py(), a)?.into_bound(slf.py()).into_any();
                Self::set_value(slf, &k, &v, copy)?;
            }
        } else {
            for (k, v) in crate::native::dict_iter(Some(other))? {
                Self::set_value(slf, &k, &v, copy)?;
            }
        }
        Ok(slf.clone())
    }

    fn _edit_attrs(&mut self, changed: &Bound<'_, PyAny>, removed: Vec<String>) -> PyResult<()> {
        *self.inner.attrs_mut() = edited(self.inner.attrs(), changed, removed)?;
        Ok(())
    }

    fn _plain_attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        attrs_to_py(py, self.inner.attrs())
    }

    fn _plain_coords<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        Self::plain_coords(slf)
    }

    fn _set_coord(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let var = spec_from_py(slf.py(), name, value)?;
        let mut me = slf.try_borrow_mut()?;
        me.inner.set_coord_var(name, var).py_err()?;
        me.generations.bump([name]);
        Ok(())
    }

    fn _reset_coords(&mut self, names: Vec<String>) -> PyResult<()> {
        for n in names {
            self.inner.reset_coord(&n).py_err()?;
        }
        Ok(())
    }

    fn _del_coord(&mut self, name: &str) -> PyResult<()> {
        if !self.inner.coords().contains_key(name) {
            return Err(PyKeyError::new_err(name.to_owned()));
        }
        self.__delitem__(name)
    }
}
