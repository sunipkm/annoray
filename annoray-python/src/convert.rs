//! Conversions between Python (numpy, cftime, pint-like quantities) and annoray values, and
//! the mapping of errors to Python exceptions.

use annoray::ndarray::{ArrayD, IxDyn};
use annoray::units::Calendar;
use annoray::{
    AttrValue, CfDate, CoreError, DType, Indexer, Label, LabelIndexer, Method, Owned,
    PySlice as PySliceIx, TypedArray, Unit,
};
use annoray::{Duration, Instant, TimeZone};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PySlice, PyString, PyTuple};

pub(crate) fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

pub(crate) fn io_error(e: impl std::fmt::Display) -> PyErr {
    PyOSError::new_err(e.to_string())
}

// ----- attributes -----

/// A 1-D numeric attribute as numpy: a scalar for one value (as netCDF4-python gives), an
/// array otherwise.
pub(crate) fn attr_to_py(py: Python<'_>, a: &AttrValue) -> PyResult<Py<PyAny>> {
    macro_rules! num {
        ($v:expr) => {{
            let arr = PyArrayDyn::from_owned_array(
                py,
                ArrayD::from_shape_vec(IxDyn(&[$v.len()]), $v.clone()).map_err(value_error)?,
            );
            if $v.len() == 1 {
                arr.get_item(0)?.unbind()
            } else {
                arr.into_any().unbind()
            }
        }};
    }
    Ok(match a {
        AttrValue::Str(s) => PyString::new(py, s).into_any().unbind(),
        AttrValue::Strs(v) => PyList::new(py, v)?.into_any().unbind(),
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

pub(crate) fn attrs_to_py<'py>(
    py: Python<'py>,
    attrs: &annoray::Attrs,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in attrs {
        d.set_item(k.as_str(), attr_to_py(py, v)?)?;
    }
    Ok(d)
}

pub(crate) fn attr_from_py(v: &Bound<'_, PyAny>) -> PyResult<AttrValue> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(AttrValue::Str(s));
    }
    if let Ok(b) = v.extract::<Vec<u8>>().and_then(|b| {
        if v.is_instance_of::<pyo3::types::PyBytes>() {
            Ok(b)
        } else {
            Err(PyValueError::new_err(""))
        }
    }) {
        return Ok(AttrValue::Str(String::from_utf8_lossy(&b).into_owned()));
    }
    if let Ok(v) = v.extract::<Vec<String>>() {
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

// ----- errors -----

pyo3::create_exception!(
    annoray,
    UnitError,
    PyValueError,
    "Incompatible or unknown units (e.g. adding metres to seconds)."
);

/// An annoray error as the Python exception xarray would raise (or [`UnitError`]).
pub(crate) fn core_error(e: CoreError) -> PyErr {
    let msg = e.to_string();
    match e {
        CoreError::Unit(u) => unit_error(&u),
        CoreError::NoSuchVariable(_) | CoreError::LabelNotFound { .. } => PyKeyError::new_err(msg),
        CoreError::Index(_) => PyIndexError::new_err(msg),
        CoreError::Read(_) => PyOSError::new_err(msg),
        CoreError::UnsupportedDType { .. } | CoreError::DTypeMismatch { .. } => {
            PyTypeError::new_err(msg)
        }
        _ => PyValueError::new_err(msg),
    }
}

/// Map a `Result` from annoray to a Python exception.
pub(crate) trait OrPy<T> {
    fn py_err(self) -> PyResult<T>;
}

impl<T> OrPy<T> for Result<T, CoreError> {
    fn py_err(self) -> PyResult<T> {
        self.map_err(core_error)
    }
}

impl<T> OrPy<T> for Result<T, annoray::UnitError> {
    fn py_err(self) -> PyResult<T> {
        self.map_err(|e| unit_error(&e))
    }
}

/// A unit error as `annoray.UnitError`.
pub(crate) fn unit_error(e: &annoray::UnitError) -> PyErr {
    UnitError::new_err(e.to_string())
}

// ----- values: Python → Rust -----

/// A pint-like quantity (`.magnitude` and `.units`): its magnitude and unit string.
pub(crate) fn quantity_parts<'py>(
    v: &Bound<'py, PyAny>,
) -> PyResult<Option<(Bound<'py, PyAny>, String)>> {
    if let Ok(q) = v.cast::<crate::units::PyQuantity>() {
        let q = &q.get().inner;
        let m = pyo3::types::PyFloat::new(v.py(), q.value()).into_any();
        return Ok(Some((m, q.unit().to_string())));
    }
    if !(v.hasattr("magnitude")? && v.hasattr("units")?) {
        return Ok(None);
    }
    let units = v.getattr("units")?;
    // pint's abbreviated form ("m / s"); plain strings as they are.
    let text: String = match units.call_method1("__format__", ("~",)) {
        Ok(s) => s.extract()?,
        Err(_) => units.str()?.extract()?,
    };
    Ok(Some((v.getattr("magnitude")?, text)))
}

fn numpy(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("numpy")
}

fn dtype_attr(arr: &Bound<'_, PyAny>, attr: &str) -> PyResult<String> {
    arr.getattr("dtype")?.getattr(attr)?.extract()
}

fn numbers_from(arr: &Bound<'_, PyAny>) -> PyResult<TypedArray<Owned>> {
    macro_rules! try_as {
        ($($t:ty => $var:ident),*) => {
            $(if let Ok(a) = arr.cast::<PyArrayDyn<$t>>() {
                return Ok(TypedArray::$var(a.readonly().as_array().to_owned()));
            })*
        };
    }
    try_as!(f64 => F64, f32 => F32, i64 => I64, i32 => I32, i16 => I16, i8 => I8,
        u64 => U64, u32 => U32, u16 => U16, u8 => U8, bool => Bool);
    Err(PyTypeError::new_err(format!(
        "unsupported array type {}",
        dtype_attr(arr, "str")?
    )))
}

fn ns_array(arr: &Bound<'_, PyAny>, unit: &str) -> PyResult<ArrayD<i64>> {
    let ns = arr
        .call_method1("astype", (unit,))?
        .call_method1("view", ("int64",))?;
    Ok(ns
        .cast::<PyArrayDyn<i64>>()?
        .readonly()
        .as_array()
        .to_owned())
}

/// numpy's `datetime64[ns]` values (`i64::MIN` is NaT, as in an [`Instant`]).
fn datetimes(ns: &ArrayD<i64>) -> ArrayD<Instant> {
    ns.mapv(Instant::from_unix_nanos)
}

/// numpy's `timedelta64[ns]` values (`i64::MIN` is NaT, as in a [`Duration`]).
fn timedeltas(ns: &ArrayD<i64>) -> ArrayD<Duration> {
    ns.mapv(Duration::from_nanos)
}

/// A cftime date (anything with `calendar`, `year`, …, `microsecond`).
pub(crate) fn cfdate_from_py(d: &Bound<'_, PyAny>) -> PyResult<CfDate> {
    let calendar: Calendar = d
        .getattr("calendar")?
        .extract::<String>()?
        .parse::<Calendar>()
        .map_err(value_error)?;
    let get = |k: &str| -> PyResult<i64> { d.getattr(k)?.extract() };
    let small = |k: &str| -> PyResult<u8> { u8::try_from(get(k)?).map_err(value_error) };
    let nanos = u32::try_from(get("microsecond")?.saturating_mul(1000)).map_err(value_error)?;
    let (h, mi, sec) = (small("hour")?, small("minute")?, small("second")?);
    CfDate::from_ymd(calendar, get("year")?, small("month")?, small("day")?)
        .and_then(|x| x.and_hms_nano(h, mi, sec, nanos))
        .py_err()
}

fn objects_from(arr: &Bound<'_, PyAny>) -> PyResult<TypedArray<Owned>> {
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let flat: Vec<Bound<'_, PyAny>> = arr
        .call_method0("ravel")?
        .call_method0("tolist")?
        .extract()?;
    fn shaped<T>(shape: &[usize], v: Vec<T>) -> PyResult<ArrayD<T>> {
        ArrayD::from_shape_vec(IxDyn(shape), v).map_err(value_error)
    }
    if flat.iter().all(|x| x.is_instance_of::<PyString>()) {
        let v = flat
            .iter()
            .map(|x| x.extract())
            .collect::<PyResult<Vec<String>>>()?;
        return Ok(TypedArray::Str(shaped(&shape, v)?));
    }
    if flat.iter().all(|x| x.is_instance_of::<PyBytes>()) {
        let v = flat
            .iter()
            .map(|x| Ok(String::from_utf8_lossy(&x.extract::<Vec<u8>>()?).into_owned()))
            .collect::<PyResult<Vec<String>>>()?;
        return Ok(TypedArray::Str(shaped(&shape, v)?));
    }
    if !flat.is_empty() && flat.iter().all(|x| x.hasattr("calendar").unwrap_or(false)) {
        let v = flat
            .iter()
            .map(cfdate_from_py)
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(TypedArray::CfDate(shaped(&shape, v)?));
    }
    // datetime.datetime, pandas timestamps, …: whatever numpy can turn into datetime64.
    let np = numpy(arr.py())?;
    if let Ok(dt) = np.call_method1("asarray", (arr, "datetime64[ns]")) {
        return Ok(TypedArray::DateTime(datetimes(&ns_array(
            &dt,
            "datetime64[ns]",
        )?)));
    }
    Err(PyTypeError::new_err(
        "object arrays must hold only strings, bytes, cftime dates or datetimes",
    ))
}

/// Any array-like (numpy array, list, scalar) as typed values.
pub(crate) fn typed_from_py(obj: &Bound<'_, PyAny>) -> PyResult<TypedArray<Owned>> {
    Ok(typed_zone_from_py(obj)?.0)
}

/// Any array-like as typed values, with the time zone that times with one share (whenever's
/// date-times, timezone-aware `datetime`s and pandas times; see `annoray._time`).
pub(crate) fn typed_zone_from_py(
    obj: &Bound<'_, PyAny>,
) -> PyResult<(TypedArray<Owned>, Option<TimeZone>)> {
    let np = numpy(obj.py())?;
    let plain = obj.is_instance(&np.getattr("ndarray")?)? && dtype_attr(obj, "kind")? != "O";
    let (arr, zone) = if plain {
        (obj.clone(), None)
    } else {
        let (arr, zone): (Bound<'_, PyAny>, Option<String>) = obj
            .py()
            .import("annoray._time")?
            .getattr("coerce")?
            .call1((obj,))?
            .extract()?;
        let zone = zone.map(|z| z.parse::<TimeZone>()).transpose().py_err()?;
        (arr, zone)
    };
    Ok((typed_of_array(&np, arr)?, zone))
}

/// The typed values of a numpy array (or anything `numpy.asarray` accepts).
fn typed_of_array(np: &Bound<'_, PyModule>, obj: Bound<'_, PyAny>) -> PyResult<TypedArray<Owned>> {
    let mut arr = np.call_method1("asarray", (&obj,))?;
    let kind = dtype_attr(&arr, "kind")?;
    match kind.as_str() {
        "f" | "i" | "u" | "b" => {
            // Native byte order, and float16 widened (annoray has no half floats).
            if dtype_attr(&arr, "str")? == "<f2" || dtype_attr(&arr, "str")? == ">f2" {
                arr = arr.call_method1("astype", ("float32",))?;
            }
            let native = arr.getattr("dtype")?.call_method1("newbyteorder", ("=",))?;
            // (Not `ascontiguousarray`: it turns scalars into 1-element arrays.)
            let kw = PyDict::new(arr.py());
            kw.set_item("dtype", native)?;
            kw.set_item("order", "C")?;
            arr = np.getattr("asarray")?.call((arr,), Some(&kw))?;
            numbers_from(&arr)
        }
        "M" => Ok(TypedArray::DateTime(datetimes(&ns_array(
            &arr,
            "datetime64[ns]",
        )?))),
        "m" => Ok(TypedArray::TimeDelta(timedeltas(&ns_array(
            &arr,
            "timedelta64[ns]",
        )?))),
        "U" | "S" | "O" => objects_from(&arr),
        _ => Err(PyTypeError::new_err(format!(
            "unsupported array type {}",
            dtype_attr(&arr, "str")?
        ))),
    }
}

/// A unit string as a [`Unit`].
pub(crate) fn parse_unit(s: &str) -> PyResult<Unit> {
    s.parse::<Unit>().py_err()
}

// ----- values: Rust → Python -----

fn cftime_to_py<'py>(py: Python<'py>, d: &CfDate) -> PyResult<Bound<'py, PyAny>> {
    if d.is_nat() {
        return Ok(py.None().into_bound(py));
    }
    let kw = PyDict::new(py);
    kw.set_item("calendar", d.calendar().name())?;
    py.import("cftime")?.getattr("datetime")?.call(
        (
            d.year(),
            d.month(),
            d.day(),
            d.hour(),
            d.minute(),
            d.second(),
            d.nanosecond() / 1000,
        ),
        Some(&kw),
    )
}

fn ns_to_numpy<'py>(py: Python<'py>, ns: ArrayD<i64>, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
    PyArrayDyn::from_owned_array(py, ns)
        .into_any()
        .call_method1("view", (dtype,))
}

/// The width numpy's fixed-width text type needs for `a`: the most characters (code
/// points) in an element, at least 1 (as numpy gives `<U1` for empty strings).
pub(crate) fn text_width(a: &ndarray::ArrayViewD<'_, String>) -> usize {
    a.iter()
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(0)
        .max(1)
}

/// Text as numpy's fixed-width `U{n}` (n the longest element, in characters), as numpy
/// and xarray make from Python strings: written as UTF-32 code units, then viewed.
pub(crate) fn text_to_py<'py>(
    py: Python<'py>,
    a: &ndarray::ArrayViewD<'_, String>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = text_width(a);
    let mut units = Vec::with_capacity(a.len().saturating_mul(n));
    for s in a {
        let start = units.len();
        units.extend(s.chars().map(u32::from));
        units.resize(start.saturating_add(n), 0);
    }
    let mut shape = a.shape().to_vec();
    shape.push(n);
    let flat = ArrayD::from_shape_vec(IxDyn(&shape), units).map_err(value_error)?;
    let arr = PyArrayDyn::from_owned_array(py, flat).into_any();
    let dims = PyTuple::new(py, a.shape())?;
    arr.call_method1("view", (format!("U{n}"),))?
        .call_method1("reshape", (dims,))
}

/// Owned values as a numpy array (moved, not copied, for numbers).
pub(crate) fn typed_to_py(py: Python<'_>, t: TypedArray<Owned>) -> PyResult<Bound<'_, PyAny>> {
    macro_rules! np {
        ($a:expr) => {
            Ok(PyArrayDyn::from_owned_array(py, $a).into_any())
        };
    }
    match t {
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
        TypedArray::Str(a) => text_to_py(py, &a.view()),
        TypedArray::DateTime(a) => ns_to_numpy(py, a.mapv(Instant::unix_nanos), "datetime64[ns]"),
        TypedArray::TimeDelta(a) => ns_to_numpy(py, a.mapv(Duration::nanos), "timedelta64[ns]"),
        TypedArray::CfDate(a) => {
            let mut objs = Vec::with_capacity(a.len());
            for d in &a {
                objs.push(cftime_to_py(py, d)?.unbind());
            }
            let arr = ArrayD::from_shape_vec(IxDyn(a.shape()), objs).map_err(value_error)?;
            Ok(PyArrayDyn::from_owned_object_array(py, arr).into_any())
        }
        other => Err(PyTypeError::new_err(format!(
            "values of type {} cannot be converted",
            other.dtype()
        ))),
    }
}

/// numpy's name for an element type.
pub(crate) fn numpy_dtype_name(d: DType) -> &'static str {
    match d {
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::I8 => "int8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::U16 => "uint16",
        DType::U32 => "uint32",
        DType::U64 => "uint64",
        DType::Bool => "bool",
        DType::DateTime => "datetime64[ns]",
        DType::TimeDelta => "timedelta64[ns]",
        _ => "object",
    }
}

// ----- labels and indexers -----

/// A Python value as a coordinate label.
pub(crate) fn label_from_py(v: &Bound<'_, PyAny>) -> PyResult<Label> {
    if let Some((mag, unit)) = quantity_parts(v)? {
        return Ok(Label::Quantity(mag.extract()?, parse_unit(&unit)?));
    }
    if v.is_instance_of::<PyBool>() {
        return Ok(Label::Int(i64::from(v.extract::<bool>()?)));
    }
    if v.is_instance_of::<PyString>() {
        return Ok(Label::Str(v.extract()?));
    }
    if v.hasattr("calendar")? && v.hasattr("year")? {
        return Ok(Label::CfDate(cfdate_from_py(v)?));
    }
    let np = numpy(v.py())?;
    // Times with a zone (whenever, aware `datetime`s) as UTC `datetime64[ns]`.
    let (arr, _zone): (Bound<'_, PyAny>, Option<String>) = v
        .py()
        .import("annoray._time")?
        .getattr("coerce")?
        .call1((v,))?
        .extract()?;
    if arr.getattr("ndim")?.extract::<usize>()? != 0 {
        return Err(PyTypeError::new_err("expected a single label"));
    }
    match dtype_attr(&arr, "kind")?.as_str() {
        "i" | "u" => Ok(Label::Int(arr.call_method0("item")?.extract()?)),
        "f" => Ok(Label::Float(arr.call_method0("item")?.extract()?)),
        "M" => {
            let ns = ns_array(&arr, "datetime64[ns]")?;
            Ok(Label::DateTime(
                datetimes(&ns)
                    .iter()
                    .next()
                    .copied()
                    .unwrap_or(Instant::NAT),
            ))
        }
        "m" => {
            let ns = ns_array(&arr, "timedelta64[ns]")?;
            Ok(Label::TimeDelta(
                timedeltas(&ns)
                    .iter()
                    .next()
                    .copied()
                    .unwrap_or(Duration::NAT),
            ))
        }
        "O" => {
            // datetime.datetime and friends.
            let dt = np.call_method1("asarray", (v, "datetime64[ns]"))?;
            let ns = ns_array(&dt, "datetime64[ns]")?;
            Ok(Label::DateTime(
                datetimes(&ns)
                    .iter()
                    .next()
                    .copied()
                    .unwrap_or(Instant::NAT),
            ))
        }
        k => Err(PyTypeError::new_err(format!(
            "unsupported label type (kind {k:?})"
        ))),
    }
}

/// What `isel` was given for one dimension.
pub(crate) enum PosSel {
    /// An index, a slice or `...`.
    Basic(Indexer),
    /// Integer positions (from a list, integer array or boolean mask).
    Take(Vec<isize>),
}

fn opt_isize(v: &Bound<'_, PyAny>) -> PyResult<Option<isize>> {
    if v.is_none() {
        Ok(None)
    } else {
        Ok(Some(v.extract()?))
    }
}

pub(crate) fn pos_from_py(v: &Bound<'_, PyAny>) -> PyResult<PosSel> {
    if let Ok(s) = v.cast::<PySlice>() {
        return Ok(PosSel::Basic(Indexer::Slice(PySliceIx::new(
            opt_isize(&s.getattr("start")?)?,
            opt_isize(&s.getattr("stop")?)?,
            opt_isize(&s.getattr("step")?)?,
        ))));
    }
    if v.is(v.py().Ellipsis()) {
        return Ok(PosSel::Basic(Indexer::Ellipsis));
    }
    let np = numpy(v.py())?;
    let arr = np.call_method1("asarray", (v,))?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    let kind = dtype_attr(&arr, "kind")?;
    match (ndim, kind.as_str()) {
        (0, "i" | "u") => Ok(PosSel::Basic(Indexer::Int(
            arr.call_method0("item")?.extract()?,
        ))),
        (1, "b") => Ok(PosSel::Take(
            np.call_method1("flatnonzero", (arr,))?
                .call_method0("tolist")?
                .extract()?,
        )),
        (1, "i" | "u") => Ok(PosSel::Take(arr.call_method0("tolist")?.extract()?)),
        // An empty list.
        (1, "f") if arr.len()? == 0 => Ok(PosSel::Take(Vec::new())),
        _ => Err(PyIndexError::new_err(
            "indexers must be integers, slices, 1-D integer arrays or 1-D boolean masks",
        )),
    }
}

/// What `sel` was given for one dimension.
pub(crate) enum LabelSel {
    One(Box<LabelIndexer>),
    Many(Vec<LabelIndexer>),
}

pub(crate) fn label_sel_from_py(
    v: &Bound<'_, PyAny>,
    method: Method,
    tolerance: Option<&Label>,
) -> PyResult<LabelSel> {
    let scalar = |l: Label| LabelIndexer::Scalar {
        label: l,
        method,
        tolerance: tolerance.cloned(),
    };
    if let Ok(s) = v.cast::<PySlice>() {
        let bound = |k: &str| -> PyResult<Option<Label>> {
            let b = s.getattr(k)?;
            if b.is_none() {
                Ok(None)
            } else {
                label_from_py(&b).map(Some)
            }
        };
        return Ok(LabelSel::One(Box::new(LabelIndexer::Slice {
            start: bound("start")?,
            stop: bound("stop")?,
            step: opt_isize(&s.getattr("step")?)?,
        })));
    }
    let is_list = v.is_instance_of::<PyList>()
        || v.is_instance_of::<PyTuple>()
        || (v.hasattr("ndim")?
            && v.getattr("ndim")?.extract::<usize>()? >= 1
            && quantity_parts(v)?.is_none());
    if is_list {
        let items: Vec<Bound<'_, PyAny>> = if v.hasattr("tolist")? {
            // numpy arrays: elements as numpy scalars keep datetime64 precision.
            v.try_iter()?.collect::<PyResult<_>>()?
        } else {
            v.extract()?
        };
        return Ok(LabelSel::Many(
            items
                .iter()
                .map(|x| label_from_py(x).map(scalar))
                .collect::<PyResult<_>>()?,
        ));
    }
    Ok(LabelSel::One(Box::new(scalar(label_from_py(v)?))))
}

pub(crate) fn method_from_py(m: Option<&str>) -> PyResult<Method> {
    Ok(match m {
        None => Method::Exact,
        Some("nearest") => Method::Nearest,
        Some("pad" | "ffill") => Method::Pad,
        Some("backfill" | "bfill") => Method::Backfill,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "method must be nearest, pad/ffill or backfill/bfill, not {other:?}"
            )));
        }
    })
}
