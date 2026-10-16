//! `differentiate`, `integrate`, `cumulative_integrate`, `polyfit` for data arrays and
//! datasets, and the module function `polyval`.

use annoray::{Covariance, PolyfitOptions, Unit};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString};

use crate::convert::OrPy;
use crate::native::{PyDataArray, PyDataset, new_da, new_ds};
use crate::units::unit_from_py;

/// xarray's `datetime_unit` (numpy codes of fixed length) or any time unit. The parameter only
/// takes times, so numpy's codes come first: `"m"` is the minute here, not the metre.
fn time_unit(u: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Unit>> {
    let Some(u) = u.filter(|u| !u.is_none()) else {
        return Ok(None);
    };
    if let Ok(s) = u.cast::<PyString>() {
        let code = match s.to_str()? {
            "W" => Some("week"),
            "D" => Some("day"),
            "h" => Some("hour"),
            "m" => Some("minute"),
            "s" => Some("s"),
            "ms" => Some("ms"),
            "us" => Some("us"),
            "ns" => Some("ns"),
            "ps" => Some("ps"),
            "fs" => Some("fs"),
            "as" => Some("as"),
            "Y" | "M" => {
                return Err(PyValueError::new_err(
                    "years and months have no fixed length; use \"D\" or a time unit",
                ));
            }
            _ => None,
        };
        if let Some(code) = code {
            return code.parse::<Unit>().py_err().map(Some);
        }
    }
    unit_from_py(u).map(Some)
}

/// One coordinate name or a list of them.
fn coord_names(coord: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if coord.is_instance_of::<PyString>() {
        return Ok(vec![coord.extract()?]);
    }
    coord.extract()
}

fn covariance(cov: &Bound<'_, PyAny>) -> PyResult<Covariance> {
    if let Ok(s) = cov.extract::<String>() {
        return match s.as_str() {
            "unscaled" => Ok(Covariance::Unscaled),
            other => Err(PyValueError::new_err(format!(
                "cov must be a bool or \"unscaled\", not {other:?}"
            ))),
        };
    }
    if cov.is_instance_of::<PyBool>() || cov.extract::<bool>().is_ok() {
        return Ok(if cov.extract::<bool>()? {
            Covariance::Scaled
        } else {
            Covariance::None
        });
    }
    Err(PyTypeError::new_err("cov must be a bool or \"unscaled\""))
}

/// The weights `w`: the name of a coordinate, a `DataArray`, or anything numpy takes.
fn weights(
    w: &Bound<'_, PyAny>,
    coord: impl Fn(&str) -> Option<annoray::Result<ndarray::ArrayD<f64>>>,
) -> PyResult<Vec<f64>> {
    if let Ok(s) = w.cast::<PyString>() {
        let name = s.to_str()?;
        let values = coord(name)
            .ok_or_else(|| PyValueError::new_err(format!("no coordinate {name:?}")))?
            .py_err()?;
        return Ok(values.iter().copied().collect());
    }
    if let Ok(d) = w.cast::<PyDataArray>() {
        return Ok(d.get().inner.to_f64().py_err()?.iter().copied().collect());
    }
    let np = w.py().import("numpy")?;
    np.call_method1("asarray", (w, "float64"))?
        .call_method0("ravel")?
        .call_method0("tolist")?
        .extract()
}

#[allow(clippy::too_many_arguments)]
fn polyfit_options(
    skipna: Option<bool>,
    rcond: Option<f64>,
    w: Option<Vec<f64>>,
    full: bool,
    cov: &Bound<'_, PyAny>,
) -> PyResult<PolyfitOptions> {
    let mut o = PolyfitOptions::new().full(full).cov(covariance(cov)?);
    if let Some(s) = skipna {
        o = o.skipna(s);
    }
    if let Some(r) = rcond {
        o = o.rcond(r);
    }
    if let Some(w) = w {
        o = o.weights(w);
    }
    Ok(o)
}

#[pymethods]
impl PyDataArray {
    /// The derivative along the one-dimensional coordinate `coord` (numpy's `gradient`;
    /// `edge_order` 1 or 2 at the ends), in the data's unit per coordinate unit.
    ///
    /// `datetime_unit` sets the step of a time coordinate: a numpy code (`"W"`, `"D"`, `"h"`,
    /// `"m"` (minute; the parameter takes times only), `"s"`, `"ms"`, `"us"`, `"ns"`, `"ps"`,
    /// `"fs"`, `"as"`; `"Y"`/`"M"` have no fixed length and raise `ValueError`), a unit
    /// string such as `"min"`, a `Unit`, or an astropy unit. The default is seconds (xarray:
    /// the coordinate's resolution, nanoseconds); the result carries the unit.
    #[pyo3(signature = (coord, edge_order=1, datetime_unit=None))]
    fn differentiate(
        &self,
        py: Python<'_>,
        coord: &str,
        edge_order: usize,
        datetime_unit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataArray>> {
        let t = time_unit(datetime_unit)?;
        let out = py.detach(|| self.inner.differentiate(coord, edge_order, t.as_ref()));
        new_da(py, out.py_err()?)
    }

    /// The trapezoidal integral along one coordinate or several in turn, in the data's unit
    /// times the coordinate's; data in °C or dB raise `UnitError`.
    ///
    /// `datetime_unit` sets the step of a time coordinate: a numpy code (`"W"`, `"D"`, `"h"`,
    /// `"m"` (minute; the parameter takes times only), `"s"`, `"ms"`, `"us"`, `"ns"`, `"ps"`,
    /// `"fs"`, `"as"`; `"Y"`/`"M"` have no fixed length and raise `ValueError`), a unit
    /// string such as `"min"`, a `Unit`, or an astropy unit. The default is seconds (xarray:
    /// the coordinate's resolution, nanoseconds); the result carries the unit.
    #[pyo3(signature = (coord, datetime_unit=None))]
    fn integrate(
        &self,
        py: Python<'_>,
        coord: &Bound<'_, PyAny>,
        datetime_unit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataArray>> {
        let names = coord_names(coord)?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let t = time_unit(datetime_unit)?;
        let out = py.detach(|| self.inner.integrate(&refs, t.as_ref()));
        new_da(py, out.py_err()?)
    }

    /// The running trapezoidal integral from zero along one coordinate or several in turn;
    /// units as for `integrate`.
    ///
    /// `datetime_unit` sets the step of a time coordinate: a numpy code (`"W"`, `"D"`, `"h"`,
    /// `"m"` (minute; the parameter takes times only), `"s"`, `"ms"`, `"us"`, `"ns"`, `"ps"`,
    /// `"fs"`, `"as"`; `"Y"`/`"M"` have no fixed length and raise `ValueError`), a unit
    /// string such as `"min"`, a `Unit`, or an astropy unit. The default is seconds (xarray:
    /// the coordinate's resolution, nanoseconds); the result carries the unit.
    #[pyo3(signature = (coord, datetime_unit=None))]
    fn cumulative_integrate(
        &self,
        py: Python<'_>,
        coord: &Bound<'_, PyAny>,
        datetime_unit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataArray>> {
        let names = coord_names(coord)?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let t = time_unit(datetime_unit)?;
        let out = py.detach(|| self.inner.cumulative_integrate(&refs, t.as_ref()));
        new_da(py, out.py_err()?)
    }

    /// A least-squares polynomial fit along `dim` (a dataset of `polyfit_coefficients`,
    /// and with `full`/`cov` the residuals, rank, singular values, covariance).
    /// Coefficients carry no unit; their `x_units`/`y_units` attributes record the units,
    /// which `annoray.polyval` restores. Time coordinates are fitted in seconds since
    /// 1970-01-01 (xarray: nanoseconds).
    #[pyo3(signature = (dim, deg, skipna=None, rcond=None, w=None, full=false, cov=None))]
    #[allow(clippy::too_many_arguments)]
    fn polyfit(
        &self,
        py: Python<'_>,
        dim: &str,
        deg: usize,
        skipna: Option<bool>,
        rcond: Option<f64>,
        w: Option<&Bound<'_, PyAny>>,
        full: bool,
        cov: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataset>> {
        let w = match w.filter(|w| !w.is_none()) {
            Some(w) => Some(weights(w, |name| {
                self.inner.coords().get(name).map(annoray::Variable::to_f64)
            })?),
            None => None,
        };
        let no = PyBool::new(py, false);
        let opts = polyfit_options(skipna, rcond, w, full, cov.unwrap_or(no.as_any()))?;
        let out = py.detach(|| self.inner.polyfit(dim, deg, &opts));
        new_ds(py, out.py_err()?)
    }
}

#[pymethods]
impl PyDataset {
    /// `DataArray.differentiate` of every data variable along the coordinate's dimension
    /// (same `datetime_unit`).
    #[pyo3(signature = (coord, edge_order=1, datetime_unit=None))]
    fn differentiate(
        &self,
        py: Python<'_>,
        coord: &str,
        edge_order: usize,
        datetime_unit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataset>> {
        let t = time_unit(datetime_unit)?;
        let out = py.detach(|| self.inner.differentiate(coord, edge_order, t.as_ref()));
        new_ds(py, out.py_err()?)
    }

    /// `DataArray.integrate` of every data variable along the coordinates' dimensions (same
    /// `datetime_unit`).
    #[pyo3(signature = (coord, datetime_unit=None))]
    fn integrate(
        &self,
        py: Python<'_>,
        coord: &Bound<'_, PyAny>,
        datetime_unit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataset>> {
        let names = coord_names(coord)?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let t = time_unit(datetime_unit)?;
        let out = py.detach(|| self.inner.integrate(&refs, t.as_ref()));
        new_ds(py, out.py_err()?)
    }

    /// `DataArray.cumulative_integrate` of every data variable (same `datetime_unit`).
    #[pyo3(signature = (coord, datetime_unit=None))]
    fn cumulative_integrate(
        &self,
        py: Python<'_>,
        coord: &Bound<'_, PyAny>,
        datetime_unit: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataset>> {
        let names = coord_names(coord)?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let t = time_unit(datetime_unit)?;
        let out = py.detach(|| self.inner.cumulative_integrate(&refs, t.as_ref()));
        new_ds(py, out.py_err()?)
    }

    /// `DataArray.polyfit` of every data variable along `dim`, as
    /// `<name>_polyfit_coefficients` (and the rest).
    #[pyo3(signature = (dim, deg, skipna=None, rcond=None, w=None, full=false, cov=None))]
    #[allow(clippy::too_many_arguments)]
    fn polyfit(
        &self,
        py: Python<'_>,
        dim: &str,
        deg: usize,
        skipna: Option<bool>,
        rcond: Option<f64>,
        w: Option<&Bound<'_, PyAny>>,
        full: bool,
        cov: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataset>> {
        let w = match w.filter(|w| !w.is_none()) {
            Some(w) => Some(weights(w, |name| {
                self.inner.coords().get(name).map(annoray::Variable::to_f64)
            })?),
            None => None,
        };
        let no = PyBool::new(py, false);
        let opts = polyfit_options(skipna, rcond, w, full, cov.unwrap_or(no.as_any()))?;
        let out = py.detach(|| self.inner.polyfit(dim, deg, &opts));
        new_ds(py, out.py_err()?)
    }
}

/// Evaluate the polynomials of `coeffs` (along `degree_dim`, as `polyfit` gives them) at
/// `coord`, restoring the units `polyfit` recorded.
#[pyfunction]
#[pyo3(signature = (coord, coeffs, degree_dim="degree"))]
pub(crate) fn polyval(
    py: Python<'_>,
    coord: &Bound<'_, PyDataArray>,
    coeffs: &Bound<'_, PyAny>,
    degree_dim: &str,
) -> PyResult<Py<PyAny>> {
    let x = &coord.get().inner;
    if let Ok(da) = coeffs.cast::<PyDataArray>() {
        let c = &da.get().inner;
        let out = py.detach(|| annoray::polyval(x, c, degree_dim));
        return Ok(new_da(py, out.py_err()?)?.into_any());
    }
    let ds = &coeffs.cast::<PyDataset>()?.get().inner;
    let out = py.detach(|| annoray::polyval_dataset(x, ds, degree_dim));
    Ok(new_ds(py, out.py_err()?)?.into_any())
}
