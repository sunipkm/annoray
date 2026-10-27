//! Units and quantities: [`ArUnit`] for `annoray.Unit`, [`ArQuantity`] for `annoray.Quantity`.

use annoray::Unit;
use annoray::num_complex::Complex;
use annoray::units::{Dyn, Kind, Quantity, Scalar};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyFloat, PyString, PyType};

use crate::compat::annoray_module;

/// `annoray.UnitError` (the `annoray` package's, a `ValueError`) with `msg`: for unit errors
/// of annoray in Rust (`UnitError`, or a `CoreError` about units).
#[must_use]
pub fn unit_error(py: Python<'_>, msg: impl std::fmt::Display) -> PyErr {
    let class = annoray_module(py)
        .and_then(|m| m.getattr("UnitError"))
        .and_then(|c| c.cast_into::<PyType>().map_err(PyErr::from));
    match class {
        Ok(c) => PyErr::from_type(c, msg.to_string()),
        Err(e) => e,
    }
}

/// `ValueError` with `msg`: for other errors of annoray in Rust (`CoreError`, I/O).
#[must_use]
pub fn value_error(msg: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.to_string())
}

/// Whether `obj` is an instance of `annoray.<class>` (the class of the imported package).
pub(crate) fn is_annoray(obj: &Bound<'_, PyAny>, class: &str) -> PyResult<bool> {
    obj.is_instance(&annoray_module(obj.py())?.getattr(class)?)
}

/// `TypeError`: `obj` is not what `expected` names.
pub(crate) fn not_a(obj: &Bound<'_, PyAny>, expected: &str) -> PyErr {
    let found = obj
        .get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |n| n.to_string());
    PyTypeError::new_err(format!("expected {expected}, not {found}"))
}

fn parse_unit(py: Python<'_>, s: &str) -> PyResult<Unit> {
    s.parse::<Unit>().map_err(|e| unit_error(py, e))
}

/// A unit at the boundary with Python: `annoray.Unit` in Python.
///
/// From Python it takes an `annoray.Unit` or a unit string (`"mm"`, `"W m-2 sr-1"`); to
/// Python it is an `annoray.Unit`, spelled as the Rust unit displays.
#[derive(Debug, Clone, PartialEq)]
pub struct ArUnit(pub Unit);

impl<'py> IntoPyObject<'py> for ArUnit {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        annoray_module(py)?
            .getattr("Unit")?
            .call1((self.0.to_string(),))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ArUnit {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = obj.to_owned();
        let py = obj.py();
        if let Ok(s) = obj.cast::<PyString>() {
            return Ok(Self(parse_unit(py, s.to_str()?)?));
        }
        if is_annoray(&obj, "Unit")? {
            return Ok(Self(parse_unit(py, &obj.str()?.to_string_lossy())?));
        }
        Err(not_a(&obj, "an annoray.Unit or a unit string"))
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for annoray::num_complex::Complex<f64> {}
}

/// The number of an [`ArQuantity`]: `f64` (a real quantity) or `Complex<f64>`.
pub trait Magnitude: Scalar + sealed::Sealed {
    /// The number from a quantity's Python magnitude.
    ///
    /// # Errors
    /// `TypeError` for a complex magnitude where a real one is needed, or not a number.
    fn from_py(m: &Bound<'_, PyAny>) -> PyResult<Self>;

    /// The number as Python's `float` or `complex`.
    fn to_py(self, py: Python<'_>) -> Bound<'_, PyAny>;
}

impl Magnitude for f64 {
    fn from_py(m: &Bound<'_, PyAny>) -> PyResult<Self> {
        if m.is_instance_of::<PyComplex>() {
            return Err(PyTypeError::new_err(
                "a complex quantity where a real one is expected (take \
                 ArQuantity<_, Complex<f64>>, or .real / abs() in Python)",
            ));
        }
        m.extract()
    }
    fn to_py(self, py: Python<'_>) -> Bound<'_, PyAny> {
        PyFloat::new(py, self).into_any()
    }
}

impl Magnitude for Complex<f64> {
    fn from_py(m: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(c) = m.cast::<PyComplex>() {
            return Ok(Complex::new(c.real(), c.imag()));
        }
        Ok(Complex::new(m.extract()?, 0.0))
    }
    fn to_py(self, py: Python<'_>) -> Bound<'_, PyAny> {
        PyComplex::from_doubles(py, self.re, self.im).into_any()
    }
}

/// A quantity at the boundary with Python: `annoray.Quantity` in Python.
///
/// `ArQuantity<K, T>` holds a [`Quantity<K, T>`]: `K` is its dimension (`Dyn`, checked at run
/// time, or a typed one such as `Length`, checked when converted from Python) and `T` its
/// number, `f64` or `Complex<f64>`. From Python it takes an `annoray.Quantity` or a unit
/// string (one of that unit: `"mm"` is 1 mm); a real quantity becomes a complex one, not the
/// reverse.
#[derive(Debug, Clone, PartialEq)]
pub struct ArQuantity<K: Kind = Dyn, T: Magnitude = f64>(pub Quantity<K, T>);

impl<'py, K: Kind, T: Magnitude> IntoPyObject<'py> for ArQuantity<K, T> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (value, unit) = self.0.into_parts();
        annoray_module(py)?
            .getattr("Quantity")?
            .call1((value.to_py(py), unit.to_string()))
    }
}

impl<'a, 'py, K: Kind, T: Magnitude> FromPyObject<'a, 'py> for ArQuantity<K, T> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = obj.to_owned();
        let py = obj.py();
        let (value, unit) = if let Ok(s) = obj.cast::<PyString>() {
            (T::from_py(&PyFloat::new(py, 1.0))?, s.to_str()?.to_owned())
        } else if is_annoray(&obj, "Quantity")? {
            (
                T::from_py(&obj.getattr("magnitude")?)?,
                obj.getattr("units")?.extract::<String>()?,
            )
        } else {
            return Err(not_a(&obj, "an annoray.Quantity or a unit string"));
        };
        let unit = parse_unit(py, &unit)?;
        Ok(Self(
            Quantity::new(value, unit).map_err(|e| unit_error(py, e))?,
        ))
    }
}
