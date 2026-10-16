//! `stack` and `unstack` for data arrays and datasets.

use annoray::{DataArray, Dataset, Name};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::convert::OrPy;
use crate::native::{PyDataArray, PyDataset, new_da, new_ds};

trait Stack: Sized + Send + Sync {
    fn stack_(&self, new: &str, dims: &[&str]) -> annoray::Result<Self>;
    fn unstack_(&self, dim: &str, fill: Option<f64>) -> annoray::Result<Self>;
    fn all_dims(&self) -> Vec<Name>;
    fn stacked(&self) -> Vec<Name>;
}

impl Stack for DataArray {
    fn stack_(&self, new: &str, dims: &[&str]) -> annoray::Result<Self> {
        self.stack(new, dims)
    }
    fn unstack_(&self, dim: &str, fill: Option<f64>) -> annoray::Result<Self> {
        self.unstack(dim, fill)
    }
    fn all_dims(&self) -> Vec<Name> {
        self.dims().to_vec()
    }
    fn stacked(&self) -> Vec<Name> {
        self.stacked_dims()
    }
}

impl Stack for Dataset {
    fn stack_(&self, new: &str, dims: &[&str]) -> annoray::Result<Self> {
        self.stack(new, dims)
    }
    fn unstack_(&self, dim: &str, fill: Option<f64>) -> annoray::Result<Self> {
        self.unstack(dim, fill)
    }
    fn all_dims(&self) -> Vec<Name> {
        self.dims()
    }
    fn stacked(&self) -> Vec<Name> {
        self.stacked_dims()
    }
}

/// `{new: dims}` from `dim` and keywords; `...` in `dims` stands for the dimensions not
/// named.
fn stacks<T: Stack>(
    obj: &T,
    dim: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(String, Vec<String>)>> {
    let mut out = Vec::new();
    for d in [dim.filter(|d| !d.is_none()), kwargs.map(|k| k.as_any())]
        .into_iter()
        .flatten()
    {
        let d = d
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("stack takes {new_dim: dims} or new_dim=dims"))?;
        for (k, v) in d.iter() {
            let mut names: Vec<Option<String>> = Vec::new();
            for x in v.try_iter()? {
                let x = x?;
                names.push(if x.is(x.py().Ellipsis()) {
                    None
                } else {
                    Some(x.extract()?)
                });
            }
            let named: Vec<String> = names.iter().flatten().cloned().collect();
            let mut dims = Vec::new();
            for n in names {
                match n {
                    Some(n) => dims.push(n),
                    None => dims.extend(
                        obj.all_dims()
                            .iter()
                            .map(ToString::to_string)
                            .filter(|d| !named.contains(d)),
                    ),
                }
            }
            out.push((k.extract()?, dims));
        }
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("stack needs new_dim=dims"));
    }
    Ok(out)
}

fn do_stack<T: Stack + Clone>(
    py: Python<'_>,
    obj: &T,
    plan: Vec<(String, Vec<String>)>,
) -> PyResult<T> {
    py.detach(|| {
        let mut out = obj.clone();
        for (new, dims) in &plan {
            let refs: Vec<&str> = dims.iter().map(String::as_str).collect();
            out = out.stack_(new, &refs)?;
        }
        Ok::<_, annoray::CoreError>(out)
    })
    .py_err()
}

fn do_unstack<T: Stack + Clone>(
    py: Python<'_>,
    obj: &T,
    dim: Option<&Bound<'_, PyAny>>,
    fill_value: Option<f64>,
) -> PyResult<T> {
    let dims: Vec<String> = match dim.filter(|d| !d.is_none()) {
        None => obj.stacked().iter().map(ToString::to_string).collect(),
        Some(d) if d.is_instance_of::<PyString>() => vec![d.extract()?],
        Some(d) => d.extract()?,
    };
    py.detach(|| {
        let mut out = obj.clone();
        for d in &dims {
            out = out.unstack_(d, fill_value)?;
        }
        Ok::<_, annoray::CoreError>(out)
    })
    .py_err()
}

#[pymethods]
impl PyDataArray {
    /// Flatten dimensions into new ones at the end: `da.stack(z=("x", "y"))` (`...` for the
    /// dimensions not named). The stacked dimensions' coordinates become levels along `z`;
    /// `create_index` is accepted for compatibility (levels are always recorded).
    #[pyo3(signature = (dim=None, create_index=true, **kwargs))]
    fn stack(
        &self,
        py: Python<'_>,
        dim: Option<&Bound<'_, PyAny>>,
        create_index: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let _ = create_index;
        let plan = stacks(&self.inner, dim, kwargs)?;
        new_da(py, do_stack(py, &self.inner, plan)?)
    }

    /// Split stacked dimensions (all of them by default) back into their levels; absent
    /// combinations get `fill_value` (NaN/NaT by default).
    #[pyo3(signature = (dim=None, fill_value=None, sparse=false))]
    fn unstack(
        &self,
        py: Python<'_>,
        dim: Option<&Bound<'_, PyAny>>,
        fill_value: Option<f64>,
        sparse: bool,
    ) -> PyResult<Py<PyDataArray>> {
        if sparse {
            return Err(PyValueError::new_err("sparse arrays are not supported"));
        }
        new_da(py, do_unstack(py, &self.inner, dim, fill_value)?)
    }
}

#[pymethods]
impl PyDataset {
    /// `DataArray.stack` of every variable.
    #[pyo3(signature = (dim=None, create_index=true, **kwargs))]
    fn stack(
        &self,
        py: Python<'_>,
        dim: Option<&Bound<'_, PyAny>>,
        create_index: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let _ = create_index;
        let plan = stacks(&self.inner, dim, kwargs)?;
        new_ds(py, do_stack(py, &self.inner, plan)?)
    }

    /// `DataArray.unstack` of every variable.
    #[pyo3(signature = (dim=None, fill_value=None, sparse=false))]
    fn unstack(
        &self,
        py: Python<'_>,
        dim: Option<&Bound<'_, PyAny>>,
        fill_value: Option<f64>,
        sparse: bool,
    ) -> PyResult<Py<PyDataset>> {
        if sparse {
            return Err(PyValueError::new_err("sparse arrays are not supported"));
        }
        new_ds(py, do_unstack(py, &self.inner, dim, fill_value)?)
    }
}
