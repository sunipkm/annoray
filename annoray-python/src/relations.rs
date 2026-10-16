//! Whole-object comparison (`equals`, `identical`, `broadcast_equals`), broadcasting
//! (`broadcast`, `broadcast_like`) and `cov`/`corr`.

use annoray::{DataArray, Name};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::convert::OrPy;
use crate::native::{PyDataArray, PyDataset, new_da, reduce_dims};

fn names(exclude: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    Ok(reduce_dims(exclude)?
        .unwrap_or_default()
        .iter()
        .map(ToString::to_string)
        .collect())
}

#[pymethods]
impl PyDataArray {
    /// Same dimensions, coordinates and values (units converted: 1 km equals 1000 m; NaN
    /// equals NaN). Names and attributes are not compared. `False` for a non-DataArray.
    fn equals(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(o) = other.cast::<PyDataArray>() else {
            return Ok(false);
        };
        let o = o.borrow();
        let o = &o.inner;
        py.detach(|| self.inner.equals(o)).py_err()
    }

    /// `equals`, and the same units, name and attributes (also of the coordinates).
    fn identical(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(o) = other.cast::<PyDataArray>() else {
            return Ok(false);
        };
        let o = o.borrow();
        let o = &o.inner;
        py.detach(|| self.inner.identical(o)).py_err()
    }

    /// `equals` once the two are broadcast against each other.
    fn broadcast_equals(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(o) = other.cast::<PyDataArray>() else {
            return Ok(false);
        };
        let o = o.borrow();
        let o = &o.inner;
        py.detach(|| self.inner.broadcast_equals(o)).py_err()
    }

    /// This array broadcast against `other` (aligned with it, outer join): the missing
    /// dimensions are added first, with `other`'s lengths and coordinates.
    #[pyo3(signature = (other, exclude=None))]
    fn broadcast_like(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyDataArray>,
        exclude: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataArray>> {
        let ex = names(exclude)?;
        let refs: Vec<&str> = ex.iter().map(String::as_str).collect();
        let o = other.borrow();
        let o = &o.inner;
        new_da(
            py,
            py.detach(|| self.inner.broadcast_like(o, &refs)).py_err()?,
        )
    }
}

#[pymethods]
impl PyDataset {
    /// Same variables (any order), each with the same dimensions, coordinates and values
    /// (units converted; NaN equals NaN). Attributes are not compared.
    fn equals(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(o) = other.cast::<PyDataset>() else {
            return Ok(false);
        };
        let o = o.borrow();
        let o = &o.inner;
        py.detach(|| self.inner.equals(o)).py_err()
    }

    /// `equals`, and the same units and attributes (global and per variable).
    fn identical(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(o) = other.cast::<PyDataset>() else {
            return Ok(false);
        };
        let o = o.borrow();
        let o = &o.inner;
        py.detach(|| self.inner.identical(o)).py_err()
    }
}

/// Data arrays aligned (outer join) and broadcast against one another.
#[pyfunction]
#[pyo3(signature = (*args, exclude=None))]
pub(crate) fn broadcast<'py>(
    py: Python<'py>,
    args: Vec<Bound<'py, PyAny>>,
    exclude: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let held = args
        .iter()
        .map(|a| {
            a.cast::<PyDataArray>()
                .map(Bound::borrow)
                .map_err(|_| PyTypeError::new_err("broadcast takes DataArrays"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let refs: Vec<&DataArray> = held.iter().map(|a| &a.inner).collect();
    let ex = names(exclude)?;
    let ex: Vec<&str> = ex.iter().map(String::as_str).collect();
    let out = py.detach(|| annoray::broadcast(&refs, &ex)).py_err()?;
    let items = out
        .into_iter()
        .map(|a| new_da(py, a))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, items)
}

/// The covariance of `da_a` and `da_b` over `dim` (all dimensions by default), normalised
/// by the number of valid pairs less `ddof`. Units multiply.
#[pyfunction]
#[pyo3(signature = (da_a, da_b, dim=None, ddof=1))]
pub(crate) fn cov(
    py: Python<'_>,
    da_a: &Bound<'_, PyDataArray>,
    da_b: &Bound<'_, PyDataArray>,
    dim: Option<&Bound<'_, PyAny>>,
    ddof: usize,
) -> PyResult<Py<PyDataArray>> {
    let dims: Option<Vec<Name>> = reduce_dims(dim)?;
    let (a, b) = (da_a.borrow(), da_b.borrow());
    let (a, b) = (&a.inner, &b.inner);
    let out = py
        .detach(|| annoray::cov(a, b, dims.as_deref(), ddof))
        .py_err()?;
    new_da(py, out)
}

/// The Pearson correlation of `da_a` and `da_b` over `dim` (all dimensions by default).
#[pyfunction]
#[pyo3(signature = (da_a, da_b, dim=None))]
pub(crate) fn corr(
    py: Python<'_>,
    da_a: &Bound<'_, PyDataArray>,
    da_b: &Bound<'_, PyDataArray>,
    dim: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyDataArray>> {
    let dims: Option<Vec<Name>> = reduce_dims(dim)?;
    let (a, b) = (da_a.borrow(), da_b.borrow());
    let (a, b) = (&a.inner, &b.inner);
    let out = py
        .detach(|| annoray::corr(a, b, dims.as_deref()))
        .py_err()?;
    new_da(py, out)
}
