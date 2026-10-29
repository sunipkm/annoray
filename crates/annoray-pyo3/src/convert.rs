//! Conversions between annoray's values and attributes and numpy / Python objects, the pieces
//! the exchange format ([`parts`](crate::parts)) is built from. They need no `annoray` package
//! at run time.

use annoray::{AttrValue, Attrs, Duration, Instant, Name, Owned, TypedArray};
use ndarray::{ArrayD, IxDyn};
use numpy::{Complex32, Complex64, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ----- attributes -----

/// An attribute as Python: a `str`, a list of `str`, or numpy (a scalar for one value, as
/// netCDF4-python gives). A string array of one element is a plain `str`, like a single
/// number is a scalar, so `['Hα']` and `'Hα'` read the same; two or more strings are a list.
///
/// # Errors
/// `ValueError` for an attribute of a type annoray adds later (the type is non-exhaustive).
pub fn attr_to_py<'py>(py: Python<'py>, a: &AttrValue) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! num {
        ($v:expr) => {{
            let arr = PyArrayDyn::from_owned_array(
                py,
                ArrayD::from_shape_vec(IxDyn(&[$v.len()]), $v.clone()).map_err(value_error)?,
            );
            if $v.len() == 1 {
                arr.get_item(0)?
            } else {
                arr.into_any()
            }
        }};
    }
    Ok(match a {
        AttrValue::Str(s) => PyString::new(py, s).into_any(),
        // netCDF4-python and xarray give one string for a one-element NC_STRING array.
        AttrValue::Strs(v) => match v.as_slice() {
            [s] => PyString::new(py, s).into_any(),
            _ => PyList::new(py, v)?.into_any(),
        },
        AttrValue::I8(v) => num!(v),
        AttrValue::I16(v) => num!(v),
        AttrValue::I32(v) => num!(v),
        AttrValue::I64(v) => num!(v),
        AttrValue::U8(v) => num!(v),
        AttrValue::U16(v) => num!(v),
        AttrValue::U32(v) => num!(v),
        AttrValue::U64(v) => num!(v),
        AttrValue::F32(v) => num!(v),
        AttrValue::F64(v) => num!(v),
        other => return Err(value_error(format!("unsupported attribute {other:?}"))),
    })
}

/// A Python attribute value: `str`, `bytes` (as UTF-8 text), a sequence of `str`, or a number
/// or array of numbers (booleans as `int8`, as netCDF stores them).
///
/// # Errors
/// `ValueError` for other values.
pub fn attr_from_py(v: &Bound<'_, PyAny>) -> PyResult<AttrValue> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(AttrValue::Str(s));
    }
    if let Ok(b) = v.cast::<PyBytes>() {
        return Ok(AttrValue::Str(
            String::from_utf8_lossy(b.as_bytes()).into_owned(),
        ));
    }
    // A sequence of `str`; a numpy array only of text (an empty numeric array would extract
    // as an empty list of strings, losing its type).
    let numeric_array = match v.getattr("dtype") {
        Ok(d) => !matches!(
            d.getattr("kind")?.extract::<String>()?.as_str(),
            "U" | "S" | "O"
        ),
        Err(_) => false,
    };
    if !numeric_array && let Ok(v) = v.extract::<Vec<String>>() {
        return Ok(AttrValue::Strs(v));
    }
    // numpy scalars and arrays (and Python numbers, via numpy).
    let np = v.py().import("numpy")?;
    let arr = np
        .call_method1("atleast_1d", (v,))?
        .call_method1("ravel", ())?;
    let kind: String = arr.getattr("dtype")?.getattr("name")?.extract()?;
    macro_rules! vec_of {
        ($t:ty, $var:ident) => {
            AttrValue::$var(arr.call_method0("tolist")?.extract::<Vec<$t>>()?)
        };
    }
    Ok(match kind.as_str() {
        "int8" => vec_of!(i8, I8),
        "int16" => vec_of!(i16, I16),
        "int32" => vec_of!(i32, I32),
        "int64" => vec_of!(i64, I64),
        "uint8" => vec_of!(u8, U8),
        "uint16" => vec_of!(u16, U16),
        "uint32" => vec_of!(u32, U32),
        "uint64" => vec_of!(u64, U64),
        "float32" => vec_of!(f32, F32),
        "float64" => vec_of!(f64, F64),
        "bool" => AttrValue::I8(
            arr.call_method0("tolist")?
                .extract::<Vec<bool>>()?
                .into_iter()
                .map(i8::from)
                .collect(),
        ),
        other => return Err(value_error(format!("unsupported attribute type {other}"))),
    })
}

/// Attributes as a Python dict.
///
/// # Errors
/// As for [`attr_to_py`].
pub fn attrs_to_py<'py>(py: Python<'py>, attrs: &Attrs) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in attrs {
        d.set_item(k.as_str(), attr_to_py(py, v)?)?;
    }
    Ok(d)
}

/// Attributes from a Python dict of `str` keys.
///
/// # Errors
/// `TypeError` if `d` is not a dict; otherwise as for [`attr_from_py`].
pub fn attrs_from_py(d: &Bound<'_, PyAny>) -> PyResult<Attrs> {
    let d = d
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("attrs must be a dict"))?;
    let mut out = Attrs::new();
    for (k, v) in d.iter() {
        out.insert(
            Name::from(k.extract::<String>()?.as_str()),
            attr_from_py(&v)?,
        );
    }
    Ok(out)
}

// ----- values -----

/// `i64` nanosecond counts viewed as numpy's `dtype` (`"datetime64[ns]"` or
/// `"timedelta64[ns]"`): moved, not copied.
///
/// # Errors
/// As numpy's `view`.
pub fn nanos_into_numpy<'py>(
    py: Python<'py>,
    ns: ArrayD<i64>,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    PyArrayDyn::from_owned_array(py, ns)
        .into_any()
        .call_method1("view", (dtype,))
}

/// Values as numpy. Numbers move into the array without a copy; text becomes an object
/// array of `str`, and date-times and durations `datetime64[ns]` and `timedelta64[ns]` (one
/// copy, to `i64`).
///
/// # Errors
/// `TypeError` for CF dates, which numpy has no type for, and for element types annoray adds
/// later.
pub fn into_numpy(py: Python<'_>, t: TypedArray<Owned>) -> PyResult<Bound<'_, PyAny>> {
    macro_rules! np {
        ($a:expr) => {
            PyArrayDyn::from_owned_array(py, $a).into_any()
        };
    }
    Ok(match t {
        TypedArray::F32(a) => np!(a),
        TypedArray::F64(a) => np!(a),
        TypedArray::I8(a) => np!(a),
        TypedArray::I16(a) => np!(a),
        TypedArray::I32(a) => np!(a),
        TypedArray::I64(a) => np!(a),
        TypedArray::U8(a) => np!(a),
        TypedArray::U16(a) => np!(a),
        TypedArray::U32(a) => np!(a),
        TypedArray::U64(a) => np!(a),
        TypedArray::Bool(a) => np!(a),
        TypedArray::C64(a) => np!(a),
        TypedArray::C128(a) => np!(a),
        TypedArray::Str(a) => {
            let objs = a.mapv(|s| PyString::new(py, &s).into_any().unbind());
            PyArrayDyn::from_owned_object_array(py, objs).into_any()
        }
        TypedArray::DateTime(a) => {
            nanos_into_numpy(py, a.mapv(Instant::unix_nanos), "datetime64[ns]")?
        }
        TypedArray::TimeDelta(a) => {
            nanos_into_numpy(py, a.mapv(Duration::nanos), "timedelta64[ns]")?
        }
        other => {
            return Err(PyTypeError::new_err(format!(
                "values of type {} have no numpy type",
                other.dtype()
            )));
        }
    })
}

/// A numpy array of numbers, booleans or complex numbers (any layout and byte order) as
/// typed values: one copy.
///
/// # Errors
/// `TypeError` for other element types.
pub fn numbers_from_numpy(arr: &Bound<'_, PyAny>) -> PyResult<TypedArray<Owned>> {
    let np = arr.py().import("numpy")?;
    // Native byte order (a view when it already is).
    let native = arr.getattr("dtype")?.call_method1("newbyteorder", ("=",))?;
    let arr = np.call_method1("asarray", (arr, native))?;
    macro_rules! try_as {
        ($($t:ty => $var:ident),*) => {
            $(if let Ok(a) = arr.cast::<PyArrayDyn<$t>>() {
                return Ok(TypedArray::$var(a.readonly().as_array().to_owned()));
            })*
        };
    }
    try_as!(f64 => F64, f32 => F32, i64 => I64, i32 => I32, i16 => I16, i8 => I8,
        u64 => U64, u32 => U32, u16 => U16, u8 => U8, bool => Bool,
        Complex64 => C128, Complex32 => C64);
    Err(PyTypeError::new_err(format!(
        "unsupported array type {}",
        arr.getattr("dtype")?.str()?
    )))
}

/// `i64` counts of a `datetime64` / `timedelta64` array in numpy's `unit`
/// (`"datetime64[ns]"`, `"timedelta64[ns]"`, …), `i64::MIN` for NaT: one copy.
///
/// # Errors
/// As numpy's `astype`, for an array that is not of dates or durations.
pub fn nanos_from_numpy(arr: &Bound<'_, PyAny>, unit: &str) -> PyResult<ArrayD<i64>> {
    let ns = arr
        .call_method1("astype", (unit,))?
        .call_method1("view", ("int64",))?;
    Ok(ns
        .cast::<PyArrayDyn<i64>>()?
        .readonly()
        .as_array()
        .to_owned())
}
