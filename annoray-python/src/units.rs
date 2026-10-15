//! `Unit` and `Quantity`: annoray's Rust unit system, directly from Python.
//!
//! Units are parsed and checked once, when made. Quantities follow the same rules as data
//! arrays: adding or subtracting needs compatible units in the same scale (`1 km + 1 m` is
//! an error: convert one first), multiplying and dividing combine units, and temperatures
//! know absolute from difference. Comparisons convert compatible units first.

// Quantity operators return `Result` (unit errors are values, never panics), which the
// `arithmetic_side_effects` lint cannot tell.
#![allow(clippy::arithmetic_side_effects)]

use std::hash::{Hash, Hasher};

use annoray::units::{BaseDim, Dimension};
use annoray::{Quantity, Unit, UnitError as RsUnitError};
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyString};

use crate::convert::{OrPy, UnitError};

/// A physical dimension: the exponents of the base quantities (the seven SI ones and angle).
/// `Dimension(length=1, time=-1)` is a velocity; dimensions multiply, divide and take
/// integer powers. `annoray.dims` names the common ones, and `annoray.dimension` declares
/// your own.
#[pyclass(
    frozen,
    name = "Dimension",
    module = "annoray",
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyDimension {
    pub(crate) inner: Dimension,
}

/// Python names of the base quantities, in `BaseDim::ALL` order.
const BASE_NAMES: [&str; 8] = [
    "mass",
    "length",
    "time",
    "current",
    "temperature",
    "amount",
    "luminosity",
    "angle",
];

#[pymethods]
impl PyDimension {
    /// `Dimension(mass=0, length=0, time=0, current=0, temperature=0, amount=0,
    /// luminosity=0, angle=0)`.
    #[new]
    #[pyo3(signature = (*, mass=0, length=0, time=0, current=0, temperature=0, amount=0,
        luminosity=0, angle=0))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        mass: i8,
        length: i8,
        time: i8,
        current: i8,
        temperature: i8,
        amount: i8,
        luminosity: i8,
        angle: i8,
    ) -> Self {
        Self {
            inner: Dimension::from_exponents([
                mass,
                length,
                time,
                current,
                temperature,
                amount,
                luminosity,
                angle,
            ]),
        }
    }

    /// Base quantity → exponent, for the non-zero ones.
    #[getter]
    fn exponents(&self) -> std::collections::BTreeMap<&'static str, i8> {
        BASE_NAMES
            .iter()
            .zip(BaseDim::ALL)
            .map(|(n, b)| (*n, self.inner.exponent(b)))
            .filter(|(_, e)| *e != 0)
            .collect()
    }

    /// Whether every exponent is zero.
    #[getter]
    fn dimensionless(&self) -> bool {
        self.inner.is_none()
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let Some(o) = dimension_of(other)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            inner: self.inner.checked_mul(o).map_err(rs_err)?,
        }))
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let Some(o) = dimension_of(other)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            inner: self.inner.checked_div(o).map_err(rs_err)?,
        }))
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        // `1 / Time` is a frequency.
        if other.extract::<f64>().is_ok_and(|v| v == 1.0) {
            return Ok(Some(Self {
                inner: Dimension::NONE.checked_div(self.inner).map_err(rs_err)?,
            }));
        }
        let Some(o) = dimension_of(other)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            inner: o.checked_div(self.inner).map_err(rs_err)?,
        }))
    }

    fn __pow__(&self, n: i32, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.powi(n).map_err(rs_err)?,
        })
    }

    /// Canonical form: `m s-1` (`1` when dimensionless).
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        let args: Vec<String> = self
            .exponents()
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        format!("Dimension({})", args.join(", "))
    }
}

/// A dimension from a `Dimension`, or an object with a `dimension` attribute holding one
/// (named dimension classes). `None` for anything else.
pub(crate) fn dimension_of(o: &Bound<'_, PyAny>) -> PyResult<Option<Dimension>> {
    if let Ok(d) = o.cast::<PyDimension>() {
        return Ok(Some(d.get().inner));
    }
    if let Ok(d) = o.getattr("dimension")
        && let Ok(d) = d.cast::<PyDimension>()
    {
        return Ok(Some(d.get().inner));
    }
    Ok(None)
}

/// A unit of measurement (`Unit("m s-1")`, `Unit("degC")`), parsed and checked when made.
/// Units multiply, divide and take integer powers; `2.5 * Unit("km")` is a `Quantity`.
#[pyclass(
    frozen,
    name = "Unit",
    module = "annoray",
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct PyUnit {
    pub(crate) inner: Unit,
}

impl Hash for PyUnit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// A unit from a string or a `Unit`.
pub(crate) fn unit_from_py(u: &Bound<'_, PyAny>) -> PyResult<Unit> {
    if let Ok(u) = u.cast::<PyUnit>() {
        return Ok(u.get().inner.clone());
    }
    if let Ok(s) = u.cast::<PyString>() {
        return s.to_str()?.parse::<Unit>().py_err();
    }
    Err(PyTypeError::new_err(
        "expected a unit (a string or annoray.Unit)",
    ))
}

fn rs_err(e: RsUnitError) -> PyErr {
    crate::convert::unit_error(&e)
}

#[pymethods]
impl PyUnit {
    #[new]
    fn py_new(unit: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: unit_from_py(unit)?,
        })
    }

    /// Whether values convert between this unit and `other` (same dimension).
    fn is_convertible(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.is_convertible(&unit_from_py(other)?))
    }

    /// The unit's dimension.
    #[getter]
    fn dimension(&self) -> PyDimension {
        PyDimension {
            inner: self.inner.dim(),
        }
    }

    /// Whether this unit has no dimension (`1`, `%`, `rad` is an angle and is not).
    #[getter]
    fn dimensionless(&self) -> bool {
        self.inner.is_dimensionless()
    }

    /// The unit in canonical form (prime-factorised scale, base dimensions).
    #[getter]
    fn canonical(&self) -> String {
        self.inner.canonical()
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(u) = unit_from_py(other) {
            return Ok(Py::new(
                py,
                Self {
                    inner: self.inner.mul(&u).map_err(rs_err)?,
                },
            )?
            .into_any());
        }
        // A number times a unit is a quantity.
        match other.extract::<f64>() {
            Ok(v) => Ok(Py::new(py, PyQuantity::new(v, self.inner.clone())?)?.into_any()),
            Err(_) => Ok(py.NotImplemented()),
        }
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__mul__(py, other)
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match unit_from_py(other) {
            Ok(u) => Ok(Py::new(
                py,
                Self {
                    inner: self.inner.div(&u).map_err(rs_err)?,
                },
            )?
            .into_any()),
            Err(_) => Ok(py.NotImplemented()),
        }
    }

    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(u) = unit_from_py(other) {
            return Ok(Py::new(
                py,
                Self {
                    inner: u.div(&self.inner).map_err(rs_err)?,
                },
            )?
            .into_any());
        }
        // A number over a unit is a quantity in the reciprocal unit.
        match other.extract::<f64>() {
            Ok(v) => Ok(
                Py::new(py, PyQuantity::new(v, self.inner.recip().map_err(rs_err)?)?)?.into_any(),
            ),
            Err(_) => Ok(py.NotImplemented()),
        }
    }

    fn __pow__(&self, n: i32, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.powi(n).map_err(rs_err)?,
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Unit({:?})", self.inner.to_string())
    }
}

/// A number with a unit (`Quantity(2.5, "km")`, or `2.5 * Unit("km")`): annoray's Rust
/// `Quantity`, with its dimension checked at run time.
///
/// Arithmetic checks units as data arrays do: `+` and `-` need the same unit (convert with
/// `.to()` first; adding a plain number works only for dimensionless quantities), `*` and
/// `/` combine units, `**` takes integer powers. Comparisons convert compatible units, so
/// `Quantity(1, "km") == Quantity(1000, "m")`; comparing incompatible units is unequal, and
/// ordering them is an error. Quantities work as operands of data arrays and as labels in
/// `sel`.
#[pyclass(frozen, name = "Quantity", module = "annoray", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyQuantity {
    pub(crate) inner: Quantity,
}

impl PyQuantity {
    pub(crate) fn new(value: f64, unit: Unit) -> PyResult<Self> {
        Ok(Self {
            inner: Quantity::new(value, unit).map_err(rs_err)?,
        })
    }

    /// The other operand: a quantity, or a plain number (dimensionless).
    fn operand(other: &Bound<'_, PyAny>) -> Option<Quantity> {
        if let Ok(q) = other.cast::<PyQuantity>() {
            return Some(q.get().inner.clone());
        }
        if other.is_instance_of::<pyo3::types::PyBool>() {
            return None;
        }
        let v = other.extract::<f64>().ok()?;
        Quantity::new(v, Unit::one()).ok()
    }
}

fn wrap(py: Python<'_>, q: Option<Result<Quantity, RsUnitError>>) -> PyResult<Py<PyAny>> {
    match q {
        Some(q) => Ok(Py::new(
            py,
            PyQuantity {
                inner: q.map_err(rs_err)?,
            },
        )?
        .into_any()),
        None => Ok(py.NotImplemented()),
    }
}

#[pymethods]
impl PyQuantity {
    /// `Quantity[Length]`: the dimension, for annotations (checked by `annoray.dims.checked`).
    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, pyo3::types::PyType>,
        item: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cls.py()
            .import("types")?
            .getattr("GenericAlias")?
            .call1((cls, item))
    }

    /// `Quantity(magnitude, units=None)`: dimensionless without `units`.
    #[new]
    #[pyo3(signature = (magnitude, units=None))]
    fn py_new(magnitude: f64, units: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let unit = match units {
            Some(u) => unit_from_py(u)?,
            None => Unit::one(),
        };
        Self::new(magnitude, unit)
    }

    /// The number, in this quantity's unit.
    #[getter]
    fn magnitude(&self) -> f64 {
        self.inner.value()
    }

    /// The quantity's dimension.
    #[getter]
    fn dimension(&self) -> PyDimension {
        PyDimension {
            inner: self.inner.unit().dim(),
        }
    }

    /// The unit, as a string (as `DataArray.units`).
    #[getter]
    fn units(&self) -> String {
        self.inner.unit().to_string()
    }

    /// The unit.
    #[getter]
    fn unit(&self) -> PyUnit {
        PyUnit {
            inner: self.inner.unit().clone(),
        }
    }

    /// The same quantity in another (compatible) unit.
    fn to(&self, units: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.to(unit_from_py(units)?).map_err(rs_err)?,
        })
    }

    fn __add__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(py, Self::operand(o).map(|b| &self.inner + &b))
    }
    fn __radd__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(py, Self::operand(o).map(|a| &a + &self.inner))
    }
    fn __sub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(py, Self::operand(o).map(|b| &self.inner - &b))
    }
    fn __rsub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(py, Self::operand(o).map(|a| &a - &self.inner))
    }
    fn __mul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(u) = o.cast::<PyUnit>() {
            let unit = self.inner.unit().mul(&u.get().inner).map_err(rs_err)?;
            return Ok(Py::new(py, Self::new(self.inner.value(), unit)?)?.into_any());
        }
        wrap(py, Self::operand(o).map(|b| &self.inner * &b))
    }
    fn __rmul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(py, Self::operand(o).map(|a| &a * &self.inner))
    }
    fn __truediv__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(u) = o.cast::<PyUnit>() {
            let unit = self.inner.unit().div(&u.get().inner).map_err(rs_err)?;
            return Ok(Py::new(py, Self::new(self.inner.value(), unit)?)?.into_any());
        }
        wrap(py, Self::operand(o).map(|b| &self.inner / &b))
    }
    fn __rtruediv__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(py, Self::operand(o).map(|a| &a / &self.inner))
    }

    fn __pow__(&self, n: i32, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.powi(n).map_err(rs_err)?,
        })
    }

    fn __neg__(&self) -> Self {
        Self {
            inner: -&self.inner,
        }
    }

    fn __pos__(&self) -> Self {
        self.clone()
    }

    fn __abs__(&self) -> Self {
        if self.inner.value() < 0.0 {
            self.__neg__()
        } else {
            self.clone()
        }
    }

    /// Only dimensionless quantities are plain numbers (`Quantity(5, "%")` is 0.05).
    fn __float__(&self) -> PyResult<f64> {
        self.inner.as_number().map_err(rs_err)
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(b) = Self::operand(other) else {
            return Ok(py.NotImplemented());
        };
        let r = match self.inner.compare(&b) {
            Some(o) => op.matches(o),
            None if !self.inner.unit().is_convertible(b.unit()) => match op {
                CompareOp::Eq => false,
                CompareOp::Ne => true,
                _ => {
                    return Err(UnitError::new_err(format!(
                        "incompatible units: {} and {}",
                        self.inner.unit(),
                        b.unit()
                    )));
                }
            },
            // NaN.
            None => matches!(op, CompareOp::Ne),
        };
        Ok(r.into_pyobject(py)?.to_owned().into_any().unbind())
    }

    /// `f"{q:.2f}"` formats the magnitude and appends the unit: `"2.50 km"`.
    fn __format__(&self, py: Python<'_>, spec: &str) -> PyResult<String> {
        let m: String = PyFloat::new(py, self.inner.value())
            .call_method1("__format__", (spec,))?
            .extract()?;
        Ok(format!("{m} {}", self.inner.unit()))
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        self.__format__(py, "")
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let m: String = PyFloat::new(py, self.inner.value()).repr()?.extract()?;
        Ok(format!(
            "Quantity({m}, {:?})",
            self.inner.unit().to_string()
        ))
    }
}
