//! `rolling_exp`: exponentially weighted moving statistics (native; see the core module).

use annoray::{DataArray, WindowType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::OrPy;
use crate::native::{PyDataArray, PyDataset, merge_kwargs, new_da, new_ds};

enum Owner {
    Da(Py<PyDataArray>),
    Ds(Py<PyDataset>),
}

/// Exponentially weighted moving statistics along one dimension (`rolling_exp`).
#[pyclass(frozen, name = "RollingExp", module = "annoray")]
pub(crate) struct PyRollingExp {
    owner: Owner,
    dim: String,
    window: f64,
    window_type: WindowType,
    min_weight: f64,
}

fn window_type(s: &str) -> PyResult<WindowType> {
    Ok(match s {
        "span" => WindowType::Span,
        "com" => WindowType::Com,
        "halflife" => WindowType::Halflife,
        "alpha" => WindowType::Alpha,
        other => {
            return Err(PyValueError::new_err(format!(
                "window_type must be \"span\", \"com\", \"halflife\" or \"alpha\", not {other:?}"
            )));
        }
    })
}

fn make(
    obj: &Bound<'_, PyAny>,
    window: Option<&Bound<'_, PyAny>>,
    window_type_: &str,
    min_weight: f64,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyRollingExp> {
    let pairs = merge_kwargs(window, kwargs)?;
    let [(dim, w)] = <[_; 1]>::try_from(pairs).map_err(|_| {
        PyValueError::new_err("rolling_exp takes exactly one dimension, e.g. time=10")
    })?;
    let owner = match obj.cast::<PyDataArray>() {
        Ok(da) => Owner::Da(da.clone().unbind()),
        Err(_) => Owner::Ds(obj.cast::<PyDataset>()?.clone().unbind()),
    };
    let r = PyRollingExp {
        owner,
        dim,
        window: w.extract()?,
        window_type: window_type(window_type_)?,
        min_weight,
    };
    // Check the window and dimension now, as xarray does.
    let py = obj.py();
    match &r.owner {
        Owner::Da(da) => {
            da.borrow(py)
                .inner
                .rolling_exp(&r.dim, r.window, r.window_type, r.min_weight)
                .py_err()?;
        }
        Owner::Ds(ds) => {
            if !ds
                .borrow(py)
                .inner
                .dims()
                .iter()
                .any(|d| d.as_str() == r.dim)
            {
                return Err(PyValueError::new_err(format!(
                    "{:?} is not a dimension",
                    r.dim
                )));
            }
        }
    }
    Ok(r)
}

impl PyRollingExp {
    fn apply(
        &self,
        py: Python<'_>,
        keep_attrs: Option<bool>,
        f: impl Fn(&annoray::RollingExp<'_, annoray::Owned>) -> annoray::Result<DataArray>,
    ) -> PyResult<Py<PyAny>> {
        let one = |da: &DataArray| -> annoray::Result<DataArray> {
            let mut out =
                f(&da.rolling_exp(&self.dim, self.window, self.window_type, self.min_weight)?)?;
            if keep_attrs == Some(false) {
                out.attrs_mut().clear();
            }
            Ok(out)
        };
        match &self.owner {
            Owner::Da(da) => {
                let held = da.borrow(py);
                Ok(new_da(py, one(&held.inner).py_err()?)?.into_any())
            }
            Owner::Ds(ds) => {
                let held = ds.borrow(py);
                let dim = self.dim.as_str();
                let out = held
                    .inner
                    .map_data_vars(keep_attrs.unwrap_or(true), |v| {
                        if v.dims().iter().any(|d| d.as_str() == dim) {
                            one(&v.to_owned())
                        } else {
                            Ok(v.to_owned())
                        }
                    })
                    .py_err()?;
                Ok(new_ds(py, out)?.into_any())
            }
        }
    }
}

#[pymethods]
impl PyRollingExp {
    /// The exponentially weighted moving mean.
    #[pyo3(signature = (keep_attrs=None))]
    fn mean(&self, py: Python<'_>, keep_attrs: Option<bool>) -> PyResult<Py<PyAny>> {
        self.apply(py, keep_attrs, |r| r.mean())
    }

    /// The exponentially weighted moving sum.
    #[pyo3(signature = (keep_attrs=None))]
    fn sum(&self, py: Python<'_>, keep_attrs: Option<bool>) -> PyResult<Py<PyAny>> {
        self.apply(py, keep_attrs, |r| r.sum())
    }

    /// The exponentially weighted moving variance (bias-corrected; numerically stable).
    #[pyo3(signature = (keep_attrs=None))]
    fn var(&self, py: Python<'_>, keep_attrs: Option<bool>) -> PyResult<Py<PyAny>> {
        self.apply(py, keep_attrs, |r| r.var())
    }

    /// The exponentially weighted moving standard deviation.
    #[pyo3(signature = (keep_attrs=None))]
    fn std(&self, py: Python<'_>, keep_attrs: Option<bool>) -> PyResult<Py<PyAny>> {
        self.apply(py, keep_attrs, |r| r.std())
    }

    /// The decayed count of values.
    #[pyo3(signature = (keep_attrs=None))]
    fn count(&self, py: Python<'_>, keep_attrs: Option<bool>) -> PyResult<Py<PyAny>> {
        self.apply(py, keep_attrs, |r| r.count())
    }

    /// The exponentially weighted moving covariance with `other` (a DataArray).
    #[pyo3(signature = (other, keep_attrs=None))]
    fn cov(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyDataArray>,
        keep_attrs: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let o = other.borrow().inner.clone();
        self.apply(py, keep_attrs, |r| r.cov(&o))
    }

    /// The exponentially weighted moving correlation with `other` (a DataArray).
    #[pyo3(signature = (other, keep_attrs=None))]
    fn corr(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyDataArray>,
        keep_attrs: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let o = other.borrow().inner.clone();
        self.apply(py, keep_attrs, |r| r.corr(&o))
    }

    fn __repr__(&self) -> String {
        format!(
            "<annoray.RollingExp {}={} ({:?}), min_weight={}>",
            self.dim, self.window, self.window_type, self.min_weight
        )
    }
}

#[pymethods]
impl PyDataArray {
    /// Exponentially weighted moving statistics: `da.rolling_exp(time=10).mean()`
    /// (`window_type` `"span"`, `"com"`, `"halflife"` or `"alpha"`; native and numerically
    /// stable, no numbagg needed).
    #[pyo3(signature = (window=None, window_type="span", min_weight=0.0, **window_kwargs))]
    fn rolling_exp(
        slf: &Bound<'_, Self>,
        window: Option<&Bound<'_, PyAny>>,
        window_type: &str,
        min_weight: f64,
        window_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRollingExp> {
        make(slf.as_any(), window, window_type, min_weight, window_kwargs)
    }
}

#[pymethods]
impl PyDataset {
    /// Exponentially weighted moving statistics of every variable along the dimension.
    #[pyo3(signature = (window=None, window_type="span", min_weight=0.0, **window_kwargs))]
    fn rolling_exp(
        slf: &Bound<'_, Self>,
        window: Option<&Bound<'_, PyAny>>,
        window_type: &str,
        min_weight: f64,
        window_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRollingExp> {
        make(slf.as_any(), window, window_type, min_weight, window_kwargs)
    }
}
