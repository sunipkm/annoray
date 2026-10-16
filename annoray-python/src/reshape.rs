//! `squeeze`, `expand_dims`, `swap_dims`, `drop_sel`, `sortby`, `assign_coords`, `diff`,
//! `shift`, `roll`, `cumsum`, `cumprod` for data arrays and datasets.

use annoray::{DataArray, Dataset, Label, Name};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::convert::{OrPy, label_from_py};
use crate::native::{PyDataArray, PyDataset, merge_kwargs, new_da, new_ds, spec_from_py};

/// The operations, for both container types.
trait Ops: Sized + Send + Sync {
    fn squeeze_(&self, dims: &[&str]) -> annoray::Result<Self>;
    fn swap_dims_(&self, old: &str, new: &str) -> annoray::Result<Self>;
    fn drop_sel_(&self, dim: &str, labels: Vec<Label>) -> annoray::Result<Self>;
    fn sortby_(&self, dim: &str, by: &str, ascending: bool) -> annoray::Result<Self>;
    fn coord_dims(&self, name: &str) -> Option<Vec<Name>>;
}

macro_rules! ops {
    ($t:ty) => {
        impl Ops for $t {
            fn squeeze_(&self, dims: &[&str]) -> annoray::Result<Self> {
                self.squeeze(dims)
            }
            fn swap_dims_(&self, old: &str, new: &str) -> annoray::Result<Self> {
                self.swap_dims(old, new)
            }
            fn drop_sel_(&self, dim: &str, labels: Vec<Label>) -> annoray::Result<Self> {
                self.drop_sel(dim, labels)
            }
            fn sortby_(&self, dim: &str, by: &str, ascending: bool) -> annoray::Result<Self> {
                self.sortby(dim, by, ascending)
            }
            fn coord_dims(&self, name: &str) -> Option<Vec<Name>> {
                self.coords().get(name).map(|c| c.dims().to_vec())
            }
        }
    };
}

ops!(DataArray);
ops!(Dataset);

/// Dimension names: `None` for all, a name, or a list of names.
fn names(dim: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    match dim {
        None => Ok(Vec::new()),
        Some(d) if d.is_none() => Ok(Vec::new()),
        Some(d) if d.is_instance_of::<PyString>() => Ok(vec![d.extract()?]),
        Some(d) => d.extract(),
    }
}

/// Labels: one, or a list/tuple of them.
fn labels(v: &Bound<'_, PyAny>) -> PyResult<Vec<Label>> {
    if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() {
        return v.try_iter()?.map(|x| label_from_py(&x?)).collect();
    }
    if v.hasattr("tolist")? && !v.is_instance_of::<PyString>() {
        let list = v.call_method0("tolist")?;
        if list.is_instance_of::<PyList>() {
            return list.try_iter()?.map(|x| label_from_py(&x?)).collect();
        }
    }
    Ok(vec![label_from_py(v)?])
}

fn upper(label: &str) -> PyResult<bool> {
    match label {
        "upper" => Ok(true),
        "lower" => Ok(false),
        other => Err(PyValueError::new_err(format!(
            "label must be \"upper\" or \"lower\", not {other:?}"
        ))),
    }
}

fn do_squeeze<T: Ops>(py: Python<'_>, x: &T, dim: Option<&Bound<'_, PyAny>>) -> PyResult<T> {
    let dims = names(dim)?;
    let refs: Vec<&str> = dims.iter().map(String::as_str).collect();
    py.detach(|| x.squeeze_(&refs)).py_err()
}

fn do_swap<T: Ops>(
    py: Python<'_>,
    x: &T,
    dims: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<T> {
    let mut out: Option<T> = None;
    for (old, new) in merge_kwargs(dims, kwargs)? {
        let new: String = new.extract()?;
        let cur = out.as_ref().unwrap_or(x);
        out = Some(py.detach(|| cur.swap_dims_(&old, &new)).py_err()?);
    }
    out.ok_or_else(|| PyValueError::new_err("swap_dims needs at least one {old: new} pair"))
}

fn do_drop_sel<T: Ops>(
    py: Python<'_>,
    x: &T,
    indexers: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<T> {
    let mut out: Option<T> = None;
    for (dim, v) in merge_kwargs(indexers, kwargs)? {
        let ls = labels(&v)?;
        let cur = out.as_ref().unwrap_or(x);
        out = Some(py.detach(|| cur.drop_sel_(&dim, ls)).py_err()?);
    }
    out.ok_or_else(|| PyValueError::new_err("drop_sel needs at least one dim=labels pair"))
}

fn do_sortby<T: Ops>(
    py: Python<'_>,
    x: &T,
    variables: &Bound<'_, PyAny>,
    ascending: bool,
) -> PyResult<T> {
    let by = names(Some(variables))?;
    let mut out: Option<T> = None;
    // As xarray: the last key is the primary one (applied last, with a stable sort).
    for name in &by {
        let cur = out.as_ref().unwrap_or(x);
        let dims = cur
            .coord_dims(name)
            .ok_or_else(|| PyValueError::new_err(format!("sortby: no coordinate {name:?}")))?;
        let [dim] = dims.as_slice() else {
            return Err(PyValueError::new_err(format!(
                "sortby: {name:?} must be one-dimensional"
            )));
        };
        out = Some(py.detach(|| cur.sortby_(dim, name, ascending)).py_err()?);
    }
    out.ok_or_else(|| PyValueError::new_err("sortby needs a coordinate name"))
}

fn shifts<'py>(
    shifts: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Vec<(String, isize)>> {
    merge_kwargs(shifts, kwargs)?
        .into_iter()
        .map(|(d, n)| Ok((d, n.extract()?)))
        .collect()
}

#[pymethods]
impl PyDataArray {
    /// Drop length-1 dimensions (all, or `dim`: a name or names), keeping their coordinates
    /// as scalars.
    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, py: Python<'_>, dim: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyDataArray>> {
        new_da(py, do_squeeze(py, &self.inner, dim)?)
    }

    /// New dimensions: a name, names, or `{name: size or labels}` (also as keywords), at
    /// `axis` (0 by default); a scalar coordinate of a new name becomes its coordinate.
    #[pyo3(signature = (dim=None, axis=None, **dim_kwargs))]
    fn expand_dims(
        &self,
        py: Python<'_>,
        dim: Option<&Bound<'_, PyAny>>,
        axis: Option<&Bound<'_, PyAny>>,
        dim_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        new_da(
            py,
            crate::utility::do_expand(py, &self.inner, dim, axis, dim_kwargs)?,
        )
    }

    /// Make coordinates the indexes of dimensions: `da.swap_dims(x="label")`.
    #[pyo3(signature = (dims_dict=None, **kwargs))]
    fn swap_dims(
        &self,
        py: Python<'_>,
        dims_dict: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        new_da(py, do_swap(py, &self.inner, dims_dict, kwargs)?)
    }

    /// Drop entries by label: `da.drop_sel(x=[1, 2])`.
    #[pyo3(signature = (labels=None, **kwargs))]
    fn drop_sel(
        &self,
        py: Python<'_>,
        labels: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        new_da(py, do_drop_sel(py, &self.inner, labels, kwargs)?)
    }

    /// Sort along the dimension of a 1-D coordinate (or several, the last one primary);
    /// NaN/NaT last.
    #[pyo3(signature = (variables, ascending=true))]
    fn sortby(
        &self,
        py: Python<'_>,
        variables: &Bound<'_, PyAny>,
        ascending: bool,
    ) -> PyResult<Py<PyDataArray>> {
        new_da(py, do_sortby(py, &self.inner, variables, ascending)?)
    }

    /// Add or replace coordinates: `da.assign_coords(x=[...], lat=("y", values))`.
    #[pyo3(signature = (coords=None, **kwargs))]
    fn assign_coords(
        &self,
        py: Python<'_>,
        coords: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let mut out = self.inner.clone();
        for (name, v) in merge_kwargs(coords, kwargs)? {
            out = out
                .with_coord(name.as_str(), spec_from_py(py, &name, &v)?)
                .py_err()?;
        }
        new_da(py, out)
    }

    /// The `n`-th difference along `dim`, labelled `"upper"` or `"lower"`.
    #[pyo3(signature = (dim, n=1, label="upper"))]
    fn diff(&self, py: Python<'_>, dim: &str, n: usize, label: &str) -> PyResult<Py<PyDataArray>> {
        let up = upper(label)?;
        new_da(py, py.detach(|| self.inner.diff(dim, n, up)).py_err()?)
    }

    /// Shift values (not coordinates), filling with NaN: `da.shift(time=1)`.
    #[pyo3(signature = (shifts=None, **kwargs))]
    fn shift(
        &self,
        py: Python<'_>,
        shifts: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let mut out = self.inner.clone();
        for (dim, n) in self::shifts(shifts, kwargs)? {
            out = py.detach(|| out.shift(&dim, n)).py_err()?;
        }
        new_da(py, out)
    }

    /// Rotate values (coordinates too with `roll_coords=True`): `da.roll(lon=10)`.
    #[pyo3(signature = (shifts=None, roll_coords=false, **kwargs))]
    fn roll(
        &self,
        py: Python<'_>,
        shifts: Option<&Bound<'_, PyAny>>,
        roll_coords: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let mut out = self.inner.clone();
        for (dim, n) in self::shifts(shifts, kwargs)? {
            out = py.detach(|| out.roll(&dim, n, roll_coords)).py_err()?;
        }
        new_da(py, out)
    }

    /// Running sum along `dim`, skipping NaN.
    fn cumsum(&self, py: Python<'_>, dim: &str) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.cumsum(dim)).py_err()?)
    }

    /// Running product along `dim`, skipping NaN (dimensionless data only).
    fn cumprod(&self, py: Python<'_>, dim: &str) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.cumprod(dim)).py_err()?)
    }
}

#[pymethods]
impl PyDataset {
    /// Drop length-1 dimensions (all, or `dim`), keeping their coordinates as scalars.
    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, py: Python<'_>, dim: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyDataset>> {
        new_ds(py, do_squeeze(py, &self.inner, dim)?)
    }

    /// Make coordinates the indexes of dimensions: `ds.swap_dims(x="label")`.
    #[pyo3(signature = (dims_dict=None, **kwargs))]
    fn swap_dims(
        &self,
        py: Python<'_>,
        dims_dict: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        new_ds(py, do_swap(py, &self.inner, dims_dict, kwargs)?)
    }

    /// Drop entries by label from every variable: `ds.drop_sel(x=[1, 2])`.
    #[pyo3(signature = (labels=None, **kwargs))]
    fn drop_sel(
        &self,
        py: Python<'_>,
        labels: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        new_ds(py, do_drop_sel(py, &self.inner, labels, kwargs)?)
    }

    /// Sort along the dimension of a 1-D coordinate; NaN/NaT last.
    #[pyo3(signature = (variables, ascending=true))]
    fn sortby(
        &self,
        py: Python<'_>,
        variables: &Bound<'_, PyAny>,
        ascending: bool,
    ) -> PyResult<Py<PyDataset>> {
        new_ds(py, do_sortby(py, &self.inner, variables, ascending)?)
    }

    /// Add or replace coordinates: `ds.assign_coords(x=[...])`.
    #[pyo3(signature = (coords=None, **kwargs))]
    fn assign_coords(
        &self,
        py: Python<'_>,
        coords: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let mut out = self.inner.clone();
        for (name, v) in merge_kwargs(coords, kwargs)? {
            out = out
                .with_coord(name.as_str(), spec_from_py(py, &name, &v)?)
                .py_err()?;
        }
        new_ds(py, out)
    }

    /// The `n`-th difference of every variable along `dim`.
    #[pyo3(signature = (dim, n=1, label="upper"))]
    fn diff(&self, py: Python<'_>, dim: &str, n: usize, label: &str) -> PyResult<Py<PyDataset>> {
        let up = upper(label)?;
        new_ds(py, py.detach(|| self.inner.diff(dim, n, up)).py_err()?)
    }

    /// Shift every variable's values, filling with NaN.
    #[pyo3(signature = (shifts=None, **kwargs))]
    fn shift(
        &self,
        py: Python<'_>,
        shifts: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let mut out = self.inner.clone();
        for (dim, n) in self::shifts(shifts, kwargs)? {
            out = py.detach(|| out.shift(&dim, n)).py_err()?;
        }
        new_ds(py, out)
    }

    /// Rotate every variable's values (coordinates too with `roll_coords=True`).
    #[pyo3(signature = (shifts=None, roll_coords=false, **kwargs))]
    fn roll(
        &self,
        py: Python<'_>,
        shifts: Option<&Bound<'_, PyAny>>,
        roll_coords: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let mut out = self.inner.clone();
        for (dim, n) in self::shifts(shifts, kwargs)? {
            out = py.detach(|| out.roll(&dim, n, roll_coords)).py_err()?;
        }
        new_ds(py, out)
    }

    /// Running sums along `dim`, skipping NaN.
    fn cumsum(&self, py: Python<'_>, dim: &str) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.cumsum(dim)).py_err()?)
    }

    /// Running products along `dim`, skipping NaN.
    fn cumprod(&self, py: Python<'_>, dim: &str) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.cumprod(dim)).py_err()?)
    }
}
