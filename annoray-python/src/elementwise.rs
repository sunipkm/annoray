//! `clip`, `round`, `isin` and `dot` for data arrays and datasets, and the module function
//! `dot`.

use annoray::{ClipBound, DataArray, Label};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

use crate::convert::{OrPy, label_from_py, parse_unit, quantity_parts};
use crate::native::{PyDataArray, PyDataset, new_da, new_ds};

/// A bound: a number (the data's unit), a quantity, or a `DataArray`.
fn bound(v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<ClipBound>> {
    let Some(v) = v.filter(|v| !v.is_none()) else {
        return Ok(None);
    };
    if let Ok(da) = v.cast::<PyDataArray>() {
        return Ok(Some(ClipBound::from(da.get().inner.clone())));
    }
    if let Some((mag, unit)) = quantity_parts(v)? {
        return Ok(Some(ClipBound::Quantity(
            mag.extract()?,
            parse_unit(&unit)?,
        )));
    }
    let x: f64 = v.extract().map_err(|_| {
        PyTypeError::new_err("clip bounds are numbers, quantities or annoray DataArrays")
    })?;
    Ok(Some(ClipBound::Value(x)))
}

/// Test elements: one label, any iterable of them, or a `DataArray`/array of them.
fn element_labels(v: &Bound<'_, PyAny>) -> PyResult<Vec<Label>> {
    let v = match v.cast::<PyDataArray>() {
        Ok(_) => v.getattr("values")?,
        Err(_) => v.clone(),
    };
    if quantity_parts(&v)?.is_some() || v.is_instance_of::<PyString>() {
        return Ok(vec![label_from_py(&v)?]);
    }
    let list = if v.hasattr("tolist")? {
        v.call_method0("tolist")?
    } else {
        v.clone()
    };
    match list.try_iter() {
        Ok(it) => it.map(|x| label_from_py(&x?)).collect(),
        Err(_) => Ok(vec![label_from_py(&list)?]),
    }
}

/// `dim` of `dot`: `None` (shared dimensions), a name, names, or `...` (all of them).
fn dot_dims(
    py: Python<'_>,
    dim: Option<&Bound<'_, PyAny>>,
    arrays: &[&DataArray],
) -> PyResult<Option<Vec<String>>> {
    let Some(d) = dim.filter(|d| !d.is_none()) else {
        return Ok(None);
    };
    if d.is(py.Ellipsis()) {
        let mut all: Vec<String> = Vec::new();
        for a in arrays {
            for n in a.dims() {
                if !all.iter().any(|x| x == n.as_str()) {
                    all.push(n.to_string());
                }
            }
        }
        return Ok(Some(all));
    }
    if d.is_instance_of::<PyString>() {
        return Ok(Some(vec![d.extract()?]));
    }
    Ok(Some(d.extract()?))
}

fn run_dot(
    py: Python<'_>,
    arrays: &[&DataArray],
    dim: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyDataArray>> {
    let dims = dot_dims(py, dim, arrays)?;
    let out = py.detach(|| {
        let refs: Option<Vec<&str>> = dims
            .as_ref()
            .map(|d| d.iter().map(String::as_str).collect());
        annoray::dot(arrays, refs.as_deref())
    });
    new_da(py, out.py_err()?)
}

#[pymethods]
impl PyDataArray {
    /// Limit the values to `[min, max]`: numbers in the data's unit, quantities (converted,
    /// °C ↔ K included) or DataArrays (broadcast and aligned). NaN stays NaN.
    #[pyo3(signature = (min=None, max=None, keep_attrs=None))]
    fn clip(
        &self,
        py: Python<'_>,
        min: Option<&Bound<'_, PyAny>>,
        max: Option<&Bound<'_, PyAny>>,
        keep_attrs: Option<bool>,
    ) -> PyResult<Py<PyDataArray>> {
        let (lo, hi) = (bound(min)?, bound(max)?);
        let mut out = py.detach(|| self.inner.clip(lo, hi)).py_err()?;
        if keep_attrs == Some(false) {
            out.attrs_mut().clear();
        }
        new_da(py, out)
    }

    /// Round to `decimals` (negative: tens, hundreds…), half to even as numpy.
    #[pyo3(signature = (decimals=0))]
    fn round(&self, py: Python<'_>, decimals: i32) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.round(decimals)).py_err()?)
    }

    #[pyo3(signature = (ndigits=None))]
    fn __round__(&self, py: Python<'_>, ndigits: Option<i32>) -> PyResult<Py<PyDataArray>> {
        self.round(py, ndigits.unwrap_or(0))
    }

    /// Whether each value is one of `test_elements` (numbers in the data's unit or
    /// quantities, strings, dates, durations).
    fn isin(&self, py: Python<'_>, test_elements: &Bound<'_, PyAny>) -> PyResult<Py<PyDataArray>> {
        let labels = element_labels(test_elements)?;
        new_da(py, py.detach(|| self.inner.isin(&labels)).py_err()?)
    }

    /// The product with `other` summed over their shared dimensions (or `dim`; `...` for
    /// all). Units multiply; NaN propagates.
    #[pyo3(signature = (other, dim=None))]
    fn dot(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyDataArray>,
        dim: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataArray>> {
        run_dot(py, &[&self.inner, &other.get().inner], dim)
    }

    fn __matmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match other.cast::<PyDataArray>() {
            Ok(o) => Ok(run_dot(py, &[&self.inner, &o.get().inner], None)?.into_any()),
            Err(_) => Ok(py.NotImplemented()),
        }
    }

    fn __rmatmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match other.cast::<PyDataArray>() {
            Ok(o) => Ok(run_dot(py, &[&o.get().inner, &self.inner], None)?.into_any()),
            Err(_) => Ok(py.NotImplemented()),
        }
    }
}

#[pymethods]
impl PyDataset {
    /// `DataArray.clip` of every numeric data variable.
    #[pyo3(signature = (min=None, max=None, keep_attrs=None))]
    fn clip(
        &self,
        py: Python<'_>,
        min: Option<&Bound<'_, PyAny>>,
        max: Option<&Bound<'_, PyAny>>,
        keep_attrs: Option<bool>,
    ) -> PyResult<Py<PyDataset>> {
        let (lo, hi) = (bound(min)?, bound(max)?);
        let mut out = py.detach(|| self.inner.clip(lo, hi)).py_err()?;
        if keep_attrs == Some(false) {
            out.attrs_mut().clear();
        }
        new_ds(py, out)
    }

    /// `DataArray.round` of every numeric data variable.
    #[pyo3(signature = (decimals=0))]
    fn round(&self, py: Python<'_>, decimals: i32) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.round(decimals)).py_err()?)
    }

    #[pyo3(signature = (ndigits=None))]
    fn __round__(&self, py: Python<'_>, ndigits: Option<i32>) -> PyResult<Py<PyDataset>> {
        self.round(py, ndigits.unwrap_or(0))
    }

    /// `DataArray.isin` of every data variable.
    fn isin(&self, py: Python<'_>, test_elements: &Bound<'_, PyAny>) -> PyResult<Py<PyDataset>> {
        let labels = element_labels(test_elements)?;
        new_ds(py, py.detach(|| self.inner.isin(&labels)).py_err()?)
    }
}

/// The product of `arrays` summed over the dimensions more than one of them has (or `dim`;
/// `...` for all), as `xr.dot`. Units multiply; NaN propagates.
#[pyfunction]
#[pyo3(signature = (*arrays, dim=None))]
pub(crate) fn dot(
    py: Python<'_>,
    arrays: &Bound<'_, PyTuple>,
    dim: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyDataArray>> {
    let owned: Vec<DataArray> = arrays
        .iter()
        .map(|a| {
            a.cast::<PyDataArray>()
                .map(|d| d.get().inner.clone())
                .map_err(|_| PyTypeError::new_err("dot takes annoray DataArrays"))
        })
        .collect::<PyResult<_>>()?;
    let refs: Vec<&DataArray> = owned.iter().collect();
    run_dot(py, &refs, dim)
}
