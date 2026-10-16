//! `ffill`, `bfill` and `interpolate_na` for data arrays and datasets.

use annoray::{FillMethod, InterpNaOptions, Label, Unit};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString};

use crate::convert::{OrPy, label_from_py};
use crate::native::{PyDataArray, PyDataset, new_da, new_ds};

fn method(m: &str) -> PyResult<FillMethod> {
    Ok(match m {
        "linear" => FillMethod::Linear,
        "nearest" => FillMethod::Nearest,
        "zero" => FillMethod::Zero,
        "slinear" => FillMethod::Slinear,
        "cubic" => FillMethod::Cubic,
        "pchip" => FillMethod::Pchip,
        "akima" => FillMethod::Akima,
        "makima" => FillMethod::Makima,
        other => {
            return Err(PyValueError::new_err(format!(
                "method {other:?} is not supported (linear, nearest, zero, slinear, cubic, \
                 pchip, akima, makima)"
            )));
        }
    })
}

/// `max_gap`: a number, quantity, timedelta, unit string (`"3 h"`) or pandas offset (`"1D"`).
fn gap(v: &Bound<'_, PyAny>) -> PyResult<Label> {
    if let Ok(s) = v.cast::<PyString>() {
        let text = s.to_str()?;
        if text.parse::<Unit>().is_ok() {
            return Ok(Label::Str(text.to_owned()));
        }
        let td = v
            .py()
            .import("pandas")
            .and_then(|pd| pd.call_method1("Timedelta", (v,)))
            .map_err(|_| {
                PyValueError::new_err(format!(
                    "max_gap {text:?} is neither a unit nor a pandas time offset"
                ))
            })?;
        return label_from_py(&td.call_method0("to_timedelta64")?);
    }
    if v.hasattr("to_timedelta64")? {
        return label_from_py(&v.call_method0("to_timedelta64")?);
    }
    label_from_py(v)
}

#[allow(clippy::too_many_arguments)]
fn options(
    method_name: &str,
    limit: Option<usize>,
    use_coordinate: &Bound<'_, PyAny>,
    max_gap: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<InterpNaOptions> {
    let mut o = InterpNaOptions::new(method(method_name)?);
    if let Some(l) = limit {
        o = o.limit(l);
    }
    if let Some(g) = max_gap.filter(|g| !g.is_none()) {
        o = o.max_gap(gap(g)?);
    }
    if let Ok(name) = use_coordinate.cast::<PyString>() {
        o = o.use_coordinate(name.to_str()?);
    } else if use_coordinate.is_instance_of::<PyBool>() {
        if !use_coordinate.extract::<bool>()? {
            o = o.use_positions();
        }
    } else {
        return Err(PyTypeError::new_err(
            "use_coordinate must be a bool or a coordinate name",
        ));
    }
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key: String = k.extract()?;
            match key.as_str() {
                "fill_value" if v.extract::<String>().is_ok_and(|s| s == "extrapolate") => {
                    o = o.extrapolate(true);
                }
                "fill_value" if v.is_none() => {}
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported keyword {key}={v} (only fill_value=\"extrapolate\")"
                    )));
                }
            }
        }
    }
    Ok(o)
}

fn need_dim(dim: Option<&str>) -> PyResult<&str> {
    dim.ok_or_else(|| PyValueError::new_err("interpolate_na needs dim"))
}

#[pymethods]
impl PyDataArray {
    /// Carry the last valid value forward over NaN/NaT along `dim`, at most `limit` steps.
    #[pyo3(signature = (dim, limit=None))]
    fn ffill(&self, py: Python<'_>, dim: &str, limit: Option<usize>) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.ffill(dim, limit)).py_err()?)
    }

    /// Carry the next valid value backward over NaN/NaT along `dim`, at most `limit` steps.
    #[pyo3(signature = (dim, limit=None))]
    fn bfill(&self, py: Python<'_>, dim: &str, limit: Option<usize>) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.bfill(dim, limit)).py_err()?)
    }

    /// Fill NaN along `dim` by interpolating through the valid points (scipy's methods:
    /// linear, nearest, zero, slinear, cubic, pchip, akima, makima). `limit` fills at most
    /// that many values per run; `max_gap` only runs whose bracketing points are at most
    /// that far apart (a number in the coordinate's unit, a quantity, a timedelta, a unit
    /// string such as `"3 h"` or a pandas offset); `use_coordinate` is `True`, `False`
    /// (positions) or a coordinate name; `fill_value="extrapolate"` also fills the ends.
    #[pyo3(signature = (dim=None, method="linear", limit=None, use_coordinate=None, max_gap=None, keep_attrs=None, **kwargs))]
    #[allow(clippy::too_many_arguments)]
    fn interpolate_na(
        &self,
        py: Python<'_>,
        dim: Option<&str>,
        method: &str,
        limit: Option<usize>,
        use_coordinate: Option<&Bound<'_, PyAny>>,
        max_gap: Option<&Bound<'_, PyAny>>,
        keep_attrs: Option<bool>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let dim = need_dim(dim)?;
        let yes = PyBool::new(py, true);
        let o = options(
            method,
            limit,
            use_coordinate.unwrap_or(yes.as_any()),
            max_gap,
            kwargs,
        )?;
        let mut out = py.detach(|| self.inner.interpolate_na(dim, &o)).py_err()?;
        if keep_attrs == Some(false) {
            out.attrs_mut().clear();
        }
        new_da(py, out)
    }
}

#[pymethods]
impl PyDataset {
    /// `DataArray.ffill` of every data variable along `dim`.
    #[pyo3(signature = (dim, limit=None))]
    fn ffill(&self, py: Python<'_>, dim: &str, limit: Option<usize>) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.ffill(dim, limit)).py_err()?)
    }

    /// `DataArray.bfill` of every data variable along `dim`.
    #[pyo3(signature = (dim, limit=None))]
    fn bfill(&self, py: Python<'_>, dim: &str, limit: Option<usize>) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.bfill(dim, limit)).py_err()?)
    }

    /// `DataArray.interpolate_na` of every data variable along `dim`.
    #[pyo3(signature = (dim=None, method="linear", limit=None, use_coordinate=None, max_gap=None, keep_attrs=None, **kwargs))]
    #[allow(clippy::too_many_arguments)]
    fn interpolate_na(
        &self,
        py: Python<'_>,
        dim: Option<&str>,
        method: &str,
        limit: Option<usize>,
        use_coordinate: Option<&Bound<'_, PyAny>>,
        max_gap: Option<&Bound<'_, PyAny>>,
        keep_attrs: Option<bool>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let dim = need_dim(dim)?;
        let yes = PyBool::new(py, true);
        let o = options(
            method,
            limit,
            use_coordinate.unwrap_or(yes.as_any()),
            max_gap,
            kwargs,
        )?;
        let mut out = py.detach(|| self.inner.interpolate_na(dim, &o)).py_err()?;
        if keep_attrs == Some(false) {
            let mut bare = annoray::Dataset::new();
            for (k, c) in out.coords() {
                bare.add_coord(k.clone(), c.clone()).py_err()?;
            }
            for (k, v) in out.data_vars() {
                let mut v = v.clone();
                v.attrs_mut().clear();
                bare.add_var(k.clone(), v).py_err()?;
            }
            out = bare;
        }
        new_ds(py, out)
    }
}
