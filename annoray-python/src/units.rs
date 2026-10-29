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

use annoray::num_complex::Complex;
use annoray::units::{BaseDim, Dimension, UnitKind};
use annoray::{Quantity, Unit, UnitError as RsUnitError};
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyFloat, PyString};

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

/// A unit from a string, a `Unit`, or an astropy unit.
pub(crate) fn unit_from_py(u: &Bound<'_, PyAny>) -> PyResult<Unit> {
    if let Ok(u) = u.cast::<PyUnit>() {
        return Ok(u.get().inner.clone());
    }
    if let Ok(s) = u.cast::<PyString>() {
        return s.to_str()?.parse::<Unit>().py_err();
    }
    if is_astropy(u)? {
        let converted = u
            .py()
            .import("annoray._astropy")?
            .call_method1("unit_from_astropy", (u,))?;
        return Ok(converted.cast::<PyUnit>()?.get().inner.clone());
    }
    Err(PyTypeError::new_err(
        "expected a unit (a string, annoray.Unit or astropy unit)",
    ))
}

/// `annoray._astropy.to_astropy(o)`.
pub(crate) fn to_astropy<'py>(o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    o.py()
        .import("annoray._astropy")?
        .call_method1("to_astropy", (o,))
}

/// Whether `o` is an astropy object (by its type's module; astropy is not imported).
pub(crate) fn is_astropy(o: &Bound<'_, PyAny>) -> PyResult<bool> {
    let module: String = o.get_type().getattr("__module__")?.extract()?;
    Ok(module == "astropy" || module.starts_with("astropy."))
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

    /// As an `astropy.units.UnitBase` (needs astropy). Names are kept where astropy reads
    /// them as the same unit; otherwise the unit is composed from SI base units.
    fn to_astropy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_astropy(slf.as_any())
    }

    /// The unit in canonical form (prime-factorised scale, base dimensions).
    #[getter]
    fn canonical(&self) -> String {
        self.inner.canonical()
    }

    /// `"linear"`, `"affine"` (°C), `"epoch"` (`days since …`) or `"log"` (dB).
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            UnitKind::Affine { .. } => "affine",
            UnitKind::Epoch(_) => "epoch",
            UnitKind::Log(_) => "log",
            _ => "linear",
        }
    }

    /// The factor to coherent SI base units (and radians): `value_SI = (value + offset) ·
    /// si_factor`.
    #[getter]
    fn si_factor(&self) -> f64 {
        self.inner.scale().to_f64()
    }

    /// The zero offset of an affine unit, in its own scale (273.15 for °C); else `None`.
    #[getter]
    fn offset(&self) -> Option<f64> {
        match self.inner.kind() {
            UnitKind::Affine { offset } => Some(offset.to_f64()),
            _ => None,
        }
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
        // A number (real or complex) times a unit is a quantity.
        match PyQuantity::from_number(other, self.inner.clone()) {
            Ok(q) => Ok(Py::new(py, q)?.into_any()),
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
        let recip = self.inner.recip().map_err(rs_err)?;
        match PyQuantity::from_number(other, recip) {
            Ok(q) => Ok(Py::new(py, q)?.into_any()),
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

/// A quantity's number: real, or complex.
pub(crate) type ComplexQuantity = Quantity<annoray::units::Dyn, Complex<f64>>;

/// The quantity a Python `Quantity` holds.
#[derive(Clone, Debug)]
pub(crate) enum Q {
    Real(Quantity),
    Complex(ComplexQuantity),
}

impl Q {
    pub(crate) fn unit(&self) -> &Unit {
        match self {
            Self::Real(q) => q.unit(),
            Self::Complex(q) => q.unit(),
        }
    }

    /// As a complex quantity (a real one with no imaginary part).
    fn widened(&self) -> ComplexQuantity {
        match self {
            Self::Real(q) => (*q).into(),
            Self::Complex(q) => *q,
        }
    }

    /// The number, as a Python `float` or `complex`.
    pub(crate) fn value_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match self {
            Self::Real(q) => PyFloat::new(py, q.value()).into_any(),
            Self::Complex(q) => PyComplex::from_doubles(py, q.value().re, q.value().im).into_any(),
        }
    }

    /// The same quantity in another unit.
    pub(crate) fn to(&self, unit: Unit) -> Result<Self, RsUnitError> {
        Ok(match self {
            Self::Real(q) => Self::Real(q.to(unit)?),
            Self::Complex(q) => Self::Complex(q.to(unit)?),
        })
    }

    /// `a op b`, real when both are, complex otherwise (as Python's numbers mix).
    fn binary(
        a: &Self,
        b: &Self,
        real: impl Fn(&Quantity, &Quantity) -> Result<Quantity, RsUnitError>,
        complex: impl Fn(&ComplexQuantity, &ComplexQuantity) -> Result<ComplexQuantity, RsUnitError>,
    ) -> Result<Self, RsUnitError> {
        match (a, b) {
            (Self::Real(x), Self::Real(y)) => real(x, y).map(Self::Real),
            _ => complex(&a.widened(), &b.widened()).map(Self::Complex),
        }
    }
}

/// A Python number as a complex one: `complex`, numpy's complex scalars, or anything with
/// `__complex__`; `None` otherwise.
fn complex_of(v: &Bound<'_, PyAny>) -> Option<Complex<f64>> {
    if let Ok(c) = v.cast::<PyComplex>() {
        return Some(Complex::new(c.real(), c.imag()));
    }
    let c = v.call_method0("__complex__").ok()?;
    let c = c.cast::<PyComplex>().ok()?;
    Some(Complex::new(c.real(), c.imag()))
}

/// A number with a unit (`Quantity(2.5, "km")`, `Quantity(3+4j, "V")`, or `2.5 * Unit("km")`):
/// annoray's Rust `Quantity`, with its dimension checked at run time; the number is a `float`
/// or a `complex`.
///
/// Arithmetic checks units as data arrays do: `+` and `-` need the same unit (convert with
/// `.to()` first; adding a plain number works only for dimensionless quantities), `*` and
/// `/` combine units, `**` takes integer powers; a real and a complex operand give a complex
/// result. Comparisons convert compatible units, so `Quantity(1, "km") == Quantity(1000, "m")`;
/// comparing incompatible units is unequal, and ordering them is an error. Complex quantities
/// are compared for equality only (they have no order), and a unit with an offset (°C) does not
/// convert them. Quantities work as operands of data arrays and as labels in `sel`.
#[pyclass(frozen, name = "Quantity", module = "annoray", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyQuantity {
    pub(crate) inner: Q,
}

impl PyQuantity {
    pub(crate) fn new(value: f64, unit: Unit) -> PyResult<Self> {
        Ok(Self {
            inner: Q::Real(Quantity::new(value, unit).map_err(rs_err)?),
        })
    }

    pub(crate) fn new_complex(value: Complex<f64>, unit: Unit) -> PyResult<Self> {
        Ok(Self {
            inner: Q::Complex(ComplexQuantity::new(value, unit).map_err(rs_err)?),
        })
    }

    /// `value` (a real or complex Python number) in `unit`.
    pub(crate) fn from_number(value: &Bound<'_, PyAny>, unit: Unit) -> PyResult<Self> {
        if let Ok(v) = value.extract::<f64>() {
            return Self::new(v, unit);
        }
        match complex_of(value) {
            Some(c) => Self::new_complex(c, unit),
            None => Err(PyTypeError::new_err(format!(
                "a quantity's magnitude is a real or complex number, not {}",
                value.get_type().name()?
            ))),
        }
    }

    /// The other operand: a quantity, or a plain number (dimensionless), real or complex.
    fn operand(other: &Bound<'_, PyAny>) -> Option<Q> {
        if let Ok(q) = other.cast::<PyQuantity>() {
            return Some(q.get().inner.clone());
        }
        if other.is_instance_of::<pyo3::types::PyBool>() {
            return None;
        }
        if let Ok(v) = other.extract::<f64>() {
            return Quantity::new(v, Unit::one()).ok().map(Q::Real);
        }
        ComplexQuantity::new(complex_of(other)?, Unit::one())
            .ok()
            .map(Q::Complex)
    }

    fn of(inner: Q) -> Self {
        Self { inner }
    }
}

fn wrap(py: Python<'_>, q: Option<Result<Q, RsUnitError>>) -> PyResult<Py<PyAny>> {
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

    /// `Quantity(magnitude, units=None)`: dimensionless without `units`; the magnitude is a
    /// real or complex number.
    #[new]
    #[pyo3(signature = (magnitude, units=None))]
    fn py_new(magnitude: &Bound<'_, PyAny>, units: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if units.is_none() && is_astropy(magnitude)? {
            // An astropy Quantity: its value and unit.
            let (value, unit): (Bound<'_, PyAny>, Bound<'_, PyAny>) = magnitude
                .py()
                .import("annoray._astropy")?
                .call_method1("quantity_parts", (magnitude,))?
                .extract()?;
            return Self::from_number(&value, unit_from_py(&unit)?);
        }
        let unit = match units {
            Some(u) => unit_from_py(u)?,
            None => Unit::one(),
        };
        Self::from_number(magnitude, unit)
    }

    /// As an `astropy.units.Quantity` (needs astropy).
    fn to_astropy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        to_astropy(slf.as_any())
    }

    /// The number, in this quantity's unit: a `float`, or a `complex`.
    #[getter]
    fn magnitude<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.value_py(py)
    }

    /// The real part, in the same unit.
    #[getter]
    fn real(&self) -> Self {
        match &self.inner {
            Q::Real(q) => Self::of(Q::Real(*q)),
            Q::Complex(q) => Self::of(Q::Real(q.re().dynamic())),
        }
    }

    /// The imaginary part, in the same unit (0 for a real quantity).
    #[getter]
    fn imag(&self) -> PyResult<Self> {
        Ok(match &self.inner {
            Q::Real(q) => Self::new(0.0, q.unit().clone())?,
            Q::Complex(q) => Self::of(Q::Real(q.im().dynamic())),
        })
    }

    /// The complex conjugate (a real quantity is its own).
    fn conjugate(&self) -> Self {
        match &self.inner {
            Q::Real(_) => self.clone(),
            Q::Complex(q) => Self::of(Q::Complex(q.conj())),
        }
    }

    /// The complex conjugate, as `conjugate()` (numpy's spelling).
    fn conj(&self) -> Self {
        self.conjugate()
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

    /// The same quantity in another (compatible) unit; for a complex quantity, both parts
    /// scale, and a unit with an offset (°C) is an error.
    fn to(&self, units: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::of(
            self.inner.to(unit_from_py(units)?).map_err(rs_err)?,
        ))
    }

    fn __add__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(
            py,
            Self::operand(o).map(|b| Q::binary(&self.inner, &b, |x, y| x + y, |x, y| x + y)),
        )
    }
    fn __radd__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(
            py,
            Self::operand(o).map(|a| Q::binary(&a, &self.inner, |x, y| x + y, |x, y| x + y)),
        )
    }
    fn __sub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(
            py,
            Self::operand(o).map(|b| Q::binary(&self.inner, &b, |x, y| x - y, |x, y| x - y)),
        )
    }
    fn __rsub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(
            py,
            Self::operand(o).map(|a| Q::binary(&a, &self.inner, |x, y| x - y, |x, y| x - y)),
        )
    }
    fn __mul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(u) = o.cast::<PyUnit>() {
            let unit = self.inner.unit().mul(&u.get().inner).map_err(rs_err)?;
            return Ok(Py::new(py, self.with_unit(unit)?)?.into_any());
        }
        wrap(
            py,
            Self::operand(o).map(|b| Q::binary(&self.inner, &b, |x, y| x * y, |x, y| x * y)),
        )
    }
    fn __rmul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(
            py,
            Self::operand(o).map(|a| Q::binary(&a, &self.inner, |x, y| x * y, |x, y| x * y)),
        )
    }
    fn __truediv__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(u) = o.cast::<PyUnit>() {
            let unit = self.inner.unit().div(&u.get().inner).map_err(rs_err)?;
            return Ok(Py::new(py, self.with_unit(unit)?)?.into_any());
        }
        wrap(
            py,
            Self::operand(o).map(|b| Q::binary(&self.inner, &b, |x, y| x / y, |x, y| x / y)),
        )
    }
    fn __rtruediv__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        wrap(
            py,
            Self::operand(o).map(|a| Q::binary(&a, &self.inner, |x, y| x / y, |x, y| x / y)),
        )
    }

    fn __pow__(&self, n: i32, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self::of(match &self.inner {
            Q::Real(q) => Q::Real(q.powi(n).map_err(rs_err)?),
            Q::Complex(q) => Q::Complex(q.powi(n).map_err(rs_err)?),
        }))
    }

    fn __neg__(&self) -> Self {
        Self::of(match &self.inner {
            Q::Real(q) => Q::Real(-q),
            Q::Complex(q) => Q::Complex(-q),
        })
    }

    fn __pos__(&self) -> Self {
        self.clone()
    }

    /// The magnitude: a real quantity in the same unit (for a complex one, its modulus).
    fn __abs__(&self) -> Self {
        match &self.inner {
            Q::Real(q) if q.value() < 0.0 => self.__neg__(),
            Q::Real(_) => self.clone(),
            Q::Complex(q) => Self::of(Q::Real(q.norm())),
        }
    }

    /// Only dimensionless quantities are plain numbers (`Quantity(5, "%")` is 0.05); a complex
    /// one is not a `float` (as `float(1j)` is an error).
    fn __float__(&self) -> PyResult<f64> {
        match &self.inner {
            Q::Real(q) => q.as_number().map_err(rs_err),
            Q::Complex(_) => Err(PyTypeError::new_err(
                "a complex quantity is not a float: take .real, .imag or abs() first",
            )),
        }
    }

    /// Only dimensionless quantities are plain numbers.
    fn __complex__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyComplex>> {
        let c = match &self.inner {
            Q::Real(q) => Complex::new(q.as_number().map_err(rs_err)?, 0.0),
            Q::Complex(q) => q.as_number().map_err(rs_err)?,
        };
        Ok(PyComplex::from_doubles(py, c.re, c.im))
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Some(b) = Self::operand(other) else {
            return Ok(py.NotImplemented());
        };
        let convertible = self.inner.unit().is_convertible(b.unit());
        let r = match (&self.inner, &b) {
            (Q::Real(x), Q::Real(y)) => match x.compare(y) {
                Some(o) => op.matches(o),
                None if !convertible => match op {
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
            },
            // Complex numbers have no order: equality only (after converting compatible units).
            (x, y) => match op {
                CompareOp::Eq => x.widened() == y.widened(),
                CompareOp::Ne => x.widened() != y.widened(),
                _ => {
                    return Err(PyTypeError::new_err(
                        "complex quantities have no order: compare .real, .imag or abs()",
                    ));
                }
            },
        };
        Ok(r.into_pyobject(py)?.to_owned().into_any().unbind())
    }

    /// `f"{q:.2f}"` formats the magnitude and appends the unit: `"2.50 km"`, `"(3+4j) V"`.
    fn __format__(&self, py: Python<'_>, spec: &str) -> PyResult<String> {
        let m: String = self
            .inner
            .value_py(py)
            .call_method1("__format__", (spec,))?
            .extract()?;
        Ok(format!("{m} {}", self.inner.unit()))
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        self.__format__(py, "")
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let m: String = self.inner.value_py(py).repr()?.extract()?;
        Ok(format!(
            "Quantity({m}, {:?})",
            self.inner.unit().to_string()
        ))
    }
}

impl PyQuantity {
    /// The same number in `unit`.
    fn with_unit(&self, unit: Unit) -> PyResult<Self> {
        match &self.inner {
            Q::Real(q) => Self::new(q.value(), unit),
            Q::Complex(q) => Self::new_complex(q.value(), unit),
        }
    }
}

/// Set the cap on interned units (default 65,536): the units quantities hold are kept once per
/// spelling in a process-wide table that only grows, and past the cap a quantity in a new unit
/// is a `UnitError`. Units already interned stay; applications that read untrusted files may
/// lower it.
#[pyfunction]
pub(crate) fn set_intern_cap(cap: usize) {
    annoray::units::set_intern_cap(cap);
}

/// The cap on interned units.
#[pyfunction]
pub(crate) fn intern_cap() -> usize {
    annoray::units::intern_cap()
}

/// The number of units interned so far.
#[pyfunction]
pub(crate) fn interned() -> usize {
    annoray::units::interned()
}
