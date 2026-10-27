//! The exchange format ("parts") between annoray values in Rust and the `annoray` Python
//! package: plain Python dicts of numpy arrays and strings, which any extension module can
//! build and read without access to another module's Rust types.
//!
//! The same code runs on both sides: in `annoray._annoray` (whose `_array_from_parts`,
//! `_array_to_parts` and their dataset and tree counterparts call it) and in every extension
//! that links this crate. [`PROTOCOL`](crate::PROTOCOL) names the format; both sides check
//! that they agree before exchanging anything.
//!
//! # The format (protocol 1)
//!
//! A **variable** is a dict:
//!
//! | key | value |
//! |---|---|
//! | `"dims"` | tuple of `str` |
//! | `"kind"` | `"number"`, `"text"`, `"datetime"`, `"timedelta"` or `"cfdate"` |
//! | `"values"` | `number`: a numpy array of a numeric, `bool` or complex type; `text`: an object array of `str`; `datetime` / `timedelta`: `datetime64[ns]` / `timedelta64[ns]`; `cfdate`: a dict of arrays `year` (`int64`, `i64::MIN` for NaT), `month`, `day`, `hour`, `minute`, `second` (`uint8`) and `nanosecond` (`uint32`) |
//! | `"calendar"` | the CF calendar of `cfdate` values, else `None` |
//! | `"units"` | the unit as written, or `None` |
//! | `"time_zone"` | the zone of `datetime` values, or `None` |
//! | `"attrs"` | dict of `str` → `str`, list of `str`, or numeric numpy array / scalar |
//!
//! A **data array** is `{"name", "variable", "coords"}` (`coords`: name → variable), a
//! **dataset** `{"data_vars", "coords", "attrs"}`, and a **tree**
//! `{"name", "dataset", "children"}` (`children`: name → tree, recursively). Encodings
//! (how values are stored on disk) are not part of the format.

use annoray::units::{Calendar, CfDate};
use annoray::{
    DType, DataArray, DataTree, Dataset, Duration, Instant, Name, Owned, TimeZone, TypedArray,
    Unit, Variable,
};
use ndarray::{ArrayD, IxDyn};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::convert::{
    attrs_from_py, attrs_to_py, into_numpy, nanos_from_numpy, numbers_from_numpy,
};

/// Values a variable lends instead of a copy (a read-only numpy view of its data), when it
/// can: given the variable, `Some(array)` or `None` to copy.
pub type Lend<'a, 'py> = &'a dyn Fn(&Variable) -> PyResult<Option<Bound<'py, PyAny>>>;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A required entry of a parts dict.
fn item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("annoray parts: missing {key:?}")))
}

/// An optional string entry (`None` or absent for none).
fn opt_str(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

fn dict<'py>(v: &Bound<'py, PyAny>, what: &str) -> PyResult<Bound<'py, PyDict>> {
    v.cast::<PyDict>()
        .cloned()
        .map_err(|_| PyTypeError::new_err(format!("annoray parts: {what} must be a dict")))
}

// ----- values -----

/// A one-byte field of a CF date (month, day, hour, minute or second).
type Field = fn(&CfDate) -> u8;

/// `values` as numpy (moved, not copied, for numbers), with the kind and the calendar.
fn values_into_py<'py>(
    py: Python<'py>,
    t: TypedArray<Owned>,
) -> PyResult<(&'static str, Bound<'py, PyAny>, Option<String>)> {
    Ok(match t {
        TypedArray::CfDate(a) => {
            let mut calendar = None;
            for d in a.iter().filter(|d| !d.is_nat()) {
                match calendar {
                    None => calendar = Some(d.calendar()),
                    Some(c) if c != d.calendar() => {
                        return Err(value_error("CF dates of more than one calendar"));
                    }
                    Some(_) => {}
                }
            }
            let fields = PyDict::new(py);
            let years = a.mapv(|d| if d.is_nat() { i64::MIN } else { d.year() });
            fields.set_item("year", PyArrayDyn::from_owned_array(py, years))?;
            let parts: [(&str, Field); 5] = [
                ("month", CfDate::month),
                ("day", CfDate::day),
                ("hour", CfDate::hour),
                ("minute", CfDate::minute),
                ("second", CfDate::second),
            ];
            for (k, f) in parts {
                fields.set_item(k, PyArrayDyn::from_owned_array(py, a.map(f)))?;
            }
            let ns = a.mapv(|d| d.nanosecond());
            fields.set_item("nanosecond", PyArrayDyn::from_owned_array(py, ns))?;
            (
                "cfdate",
                fields.into_any(),
                Some(calendar.unwrap_or(Calendar::Standard).name().to_owned()),
            )
        }
        t => (kind_of(t.dtype()), into_numpy(py, t)?, None),
    })
}

/// One field array of CF dates, as `T`.
fn field<T: numpy::Element + Copy>(fields: &Bound<'_, PyDict>, key: &str) -> PyResult<ArrayD<T>> {
    let a = item(fields, key)?;
    let dtype = numpy::dtype::<T>(a.py());
    let a = a
        .py()
        .import("numpy")?
        .call_method1("asarray", (a, dtype))?;
    Ok(a.cast::<PyArrayDyn<T>>()?.readonly().as_array().to_owned())
}

fn values_from_py(
    kind: &str,
    values: &Bound<'_, PyAny>,
    calendar: Option<&str>,
) -> PyResult<TypedArray<Owned>> {
    Ok(match kind {
        "number" => numbers_from_numpy(values)?,
        "text" => {
            let np = values.py().import("numpy")?;
            let objs = np.call_method1("asarray", (values, "object"))?;
            let objs = objs.cast::<PyArrayDyn<Py<PyAny>>>()?.readonly();
            let objs = objs.as_array();
            let shape = objs.shape().to_vec();
            let text = objs
                .iter()
                .map(|o| o.extract::<String>(values.py()))
                .collect::<PyResult<Vec<_>>>()?;
            TypedArray::Str(ArrayD::from_shape_vec(IxDyn(&shape), text).map_err(value_error)?)
        }
        "datetime" => TypedArray::DateTime(
            nanos_from_numpy(values, "datetime64[ns]")?.mapv(Instant::from_unix_nanos),
        ),
        "timedelta" => TypedArray::TimeDelta(
            nanos_from_numpy(values, "timedelta64[ns]")?.mapv(Duration::from_nanos),
        ),
        "cfdate" => {
            let calendar: Calendar = calendar
                .ok_or_else(|| value_error("annoray parts: cfdate values without a calendar"))?
                .parse()
                .map_err(value_error)?;
            let fields = dict(values, "cfdate values")?;
            let years = field::<i64>(&fields, "year")?;
            let [month, day, hour, minute, second] =
                ["month", "day", "hour", "minute", "second"].map(|k| field::<u8>(&fields, k));
            let (month, day, hour, minute, second) = (month?, day?, hour?, minute?, second?);
            let ns = field::<u32>(&fields, "nanosecond")?;
            let shape = years.shape().to_vec();
            let same = [&month, &day, &hour, &minute, &second]
                .iter()
                .all(|a| a.shape() == shape);
            if !same || ns.shape() != shape {
                return Err(value_error(
                    "annoray parts: cfdate fields of different shapes",
                ));
            }
            // All seven fields in one pass (in the same, logical, order).
            let dates = years
                .iter()
                .zip(month.iter().zip(day.iter()))
                .zip(hour.iter().zip(minute.iter()))
                .zip(second.iter().zip(ns.iter()))
                .map(|(((y, (mo, d)), (h, mi)), (s, n))| {
                    if *y == i64::MIN {
                        return Ok(CfDate::NAT);
                    }
                    CfDate::from_ymd(calendar, *y, *mo, *d)
                        .and_then(|x| x.and_hms_nano(*h, *mi, *s, *n))
                        .map_err(value_error)
                })
                .collect::<PyResult<Vec<_>>>()?;
            TypedArray::CfDate(ArrayD::from_shape_vec(IxDyn(&shape), dates).map_err(value_error)?)
        }
        other => {
            return Err(value_error(format!(
                "annoray parts: unknown kind {other:?}"
            )));
        }
    })
}

/// Whether a variable of `dtype` can lend its values (numbers, date-times, durations).
#[must_use]
pub fn lends(dtype: DType) -> bool {
    !matches!(dtype, DType::Str | DType::CfDate)
}

// ----- variables -----

fn metadata<'py>(py: Python<'py>, var: &Variable) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "dims",
        PyTuple::new(py, var.dims().iter().map(Name::as_str))?,
    )?;
    d.set_item("units", var.unit().map(ToString::to_string))?;
    d.set_item("time_zone", var.time_zone().map(|z| z.name().to_owned()))?;
    d.set_item("attrs", attrs_to_py(py, var.attrs())?)?;
    Ok(d)
}

fn kind_of(dtype: DType) -> &'static str {
    match dtype {
        DType::Str => "text",
        DType::DateTime => "datetime",
        DType::TimeDelta => "timedelta",
        DType::CfDate => "cfdate",
        _ => "number",
    }
}

/// A variable's parts, its values moved into numpy (numbers without a copy; lazily read
/// values are read).
///
/// # Errors
/// If the values cannot be read or exchanged.
pub fn variable_into_parts(py: Python<'_>, var: Variable) -> PyResult<Bound<'_, PyDict>> {
    let d = metadata(py, &var)?;
    let (kind, values, calendar) = values_into_py(py, var.into_typed().map_err(value_error)?)?;
    d.set_item("kind", kind)?;
    d.set_item("values", values)?;
    d.set_item("calendar", calendar)?;
    Ok(d)
}

/// A variable's parts: its values lent by `lend` (see [`Lend`]) or else copied.
///
/// # Errors
/// As for [`variable_into_parts`].
pub fn variable_to_parts<'py>(
    py: Python<'py>,
    var: &Variable,
    lend: Lend<'_, 'py>,
) -> PyResult<Bound<'py, PyDict>> {
    if lends(var.dtype())
        && let Some(values) = lend(var)?
    {
        let d = metadata(py, var)?;
        d.set_item("kind", kind_of(var.dtype()))?;
        d.set_item("values", values)?;
        d.set_item("calendar", None::<String>)?;
        return Ok(d);
    }
    variable_into_parts(py, var.clone())
}

/// A variable from its parts: its values copied once.
///
/// # Errors
/// `TypeError` / `ValueError` for parts that do not follow the format, or values,
/// dimensions, units or zones annoray refuses.
pub fn variable_from_parts(parts: &Bound<'_, PyAny>) -> PyResult<Variable> {
    let d = dict(parts, "a variable")?;
    let dims: Vec<String> = item(&d, "dims")?.extract()?;
    let kind: String = item(&d, "kind")?.extract()?;
    let calendar = opt_str(&d, "calendar")?;
    let values = values_from_py(&kind, &item(&d, "values")?, calendar.as_deref())?;
    let mut var = Variable::from_typed(dims, values).map_err(value_error)?;
    if let Some(u) = opt_str(&d, "units")? {
        var = var.with_unit(u.parse::<Unit>().map_err(value_error)?);
    }
    if let Some(z) = opt_str(&d, "time_zone")? {
        let tz = z.parse::<TimeZone>().map_err(value_error)?;
        var.set_time_zone(Some(tz)).map_err(value_error)?;
    }
    if let Some(a) = d.get_item("attrs")? {
        *var.attrs_mut() = attrs_from_py(&a)?;
    }
    Ok(var)
}

fn variables_from_parts(d: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<(Name, Variable)>> {
    dict(d, what)?
        .iter()
        .map(|(k, v)| {
            Ok((
                Name::from(k.extract::<String>()?.as_str()),
                variable_from_parts(&v)?,
            ))
        })
        .collect()
}

// ----- data arrays, datasets, trees -----

/// A data array's parts, its values moved (see [`variable_into_parts`]).
///
/// # Errors
/// As for [`variable_into_parts`].
pub fn dataarray_into_parts(py: Python<'_>, da: DataArray) -> PyResult<Bound<'_, PyDict>> {
    let (name, var, coords) = da.into_parts();
    let d = PyDict::new(py);
    d.set_item("name", name.as_ref().map(Name::as_str))?;
    d.set_item("variable", variable_into_parts(py, var)?)?;
    let c = PyDict::new(py);
    for (k, v) in coords {
        c.set_item(k.as_str(), variable_into_parts(py, v)?)?;
    }
    d.set_item("coords", c)?;
    Ok(d)
}

/// A data array's parts, its values lent by `lend` or copied.
///
/// # Errors
/// As for [`variable_into_parts`].
pub fn dataarray_to_parts<'py>(
    py: Python<'py>,
    da: &DataArray,
    lend: Lend<'_, 'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("name", da.name().map(Name::as_str))?;
    d.set_item("variable", variable_to_parts(py, da.variable(), lend)?)?;
    let c = PyDict::new(py);
    for (k, v) in da.coords() {
        c.set_item(k.as_str(), variable_to_parts(py, v, lend)?)?;
    }
    d.set_item("coords", c)?;
    Ok(d)
}

/// A data array from its parts.
///
/// # Errors
/// As for [`variable_from_parts`], and coordinates that do not fit the data.
pub fn dataarray_from_parts(parts: &Bound<'_, PyAny>) -> PyResult<DataArray> {
    let d = dict(parts, "a data array")?;
    let mut da = DataArray::new(variable_from_parts(&item(&d, "variable")?)?);
    if let Some(n) = opt_str(&d, "name")? {
        da = da.with_name(n.as_str());
    }
    for (k, v) in variables_from_parts(&item(&d, "coords")?, "coords")? {
        da = da.with_coord(k, v).map_err(value_error)?;
    }
    Ok(da)
}

/// A dataset's parts, its values moved.
///
/// # Errors
/// As for [`variable_into_parts`].
pub fn dataset_into_parts(py: Python<'_>, mut ds: Dataset) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("attrs", attrs_to_py(py, ds.attrs())?)?;
    let (vars, coords): (Vec<Name>, Vec<Name>) = (
        ds.data_vars().keys().cloned().collect(),
        ds.coords().keys().cloned().collect(),
    );
    for (key, names) in [("data_vars", vars), ("coords", coords)] {
        let out = PyDict::new(py);
        for k in names {
            let v = ds.remove(k.as_str()).map_err(value_error)?;
            out.set_item(k.as_str(), variable_into_parts(py, v)?)?;
        }
        d.set_item(key, out)?;
    }
    Ok(d)
}

/// A dataset's parts, its values lent by `lend` or copied.
///
/// # Errors
/// As for [`variable_into_parts`].
pub fn dataset_to_parts<'py>(
    py: Python<'py>,
    ds: &Dataset,
    lend: Lend<'_, 'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("attrs", attrs_to_py(py, ds.attrs())?)?;
    for (key, vars) in [("data_vars", ds.data_vars()), ("coords", ds.coords())] {
        let out = PyDict::new(py);
        for (k, v) in vars {
            out.set_item(k.as_str(), variable_to_parts(py, v, lend)?)?;
        }
        d.set_item(key, out)?;
    }
    Ok(d)
}

/// A dataset from its parts.
///
/// # Errors
/// As for [`variable_from_parts`], and variables whose dimensions disagree.
pub fn dataset_from_parts(parts: &Bound<'_, PyAny>) -> PyResult<Dataset> {
    let d = dict(parts, "a dataset")?;
    let mut ds = Dataset::new();
    for (k, v) in variables_from_parts(&item(&d, "coords")?, "coords")? {
        ds.add_coord(k, v).map_err(value_error)?;
    }
    for (k, v) in variables_from_parts(&item(&d, "data_vars")?, "data_vars")? {
        ds.add_var(k, v).map_err(value_error)?;
    }
    if let Some(a) = d.get_item("attrs")? {
        *ds.attrs_mut() = attrs_from_py(&a)?;
    }
    Ok(ds)
}

/// A tree's parts, its values moved; `name` is the root's name.
///
/// # Errors
/// As for [`variable_into_parts`].
pub fn tree_into_parts<'py>(
    py: Python<'py>,
    tree: DataTree,
    name: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let (ds, children) = tree.into_parts();
    let d = PyDict::new(py);
    d.set_item("name", name)?;
    d.set_item("dataset", dataset_into_parts(py, ds)?)?;
    let c = PyDict::new(py);
    for (k, child) in children {
        c.set_item(k.as_str(), tree_into_parts(py, child, Some(k.as_str()))?)?;
    }
    d.set_item("children", c)?;
    Ok(d)
}

/// A tree from its parts, and its root's name.
///
/// # Errors
/// As for [`dataset_from_parts`], and children that disagree with the coordinates they
/// inherit.
pub fn tree_from_parts(parts: &Bound<'_, PyAny>) -> PyResult<(DataTree, Option<String>)> {
    let d = dict(parts, "a tree")?;
    let mut tree = DataTree::new(dataset_from_parts(&item(&d, "dataset")?)?);
    for (k, child) in dict(&item(&d, "children")?, "children")?.iter() {
        let (child, _) = tree_from_parts(&child)?;
        tree.add_child(k.extract::<String>()?.as_str(), child)
            .map_err(value_error)?;
    }
    Ok((tree, opt_str(&d, "name")?))
}
