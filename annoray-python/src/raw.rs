//! The xarray backend's Rust side: raw (not CF-decoded) file access, and writing xarray
//! datasets.
//!
//! - [`RawGroup`]: a file opened lazily with our pure-Rust readers, serving **raw** (not
//!   CF-decoded) variables slice by slice. The Python side wraps it as an xarray backend, so
//!   xarray's own decoder runs on top and results match xarray's other engines.
//! - [`write`]: build an annoray `Dataset` from variables handed over by Python and write it
//!   with our CF encoder and pure-Rust writers.
//!
//! The GIL is released while files are read and written.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{NaiveDateTime, TimeDelta};
use ndarray::{ArrayD, IxDyn};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::convert::{attr_from_py, attrs_to_py, io_error, value_error};
use annoray::io::{Format, RawDataset, WriteOptions, default_backend};
use annoray::units::{Calendar, CfDate};
use annoray::{
    BloscCodec, BloscShuffle, Compression, DType, Dataset, NaT, Owned, TypedArray, Variable,
};
use pyo3::types::{PyDict, PyString, PyTuple};

// ----- Rust → Python -----

/// Raw values as numpy (strings as an object array).
fn array_to_py(py: Python<'_>, t: TypedArray<Owned>) -> PyResult<Py<PyAny>> {
    macro_rules! np {
        ($a:expr) => {
            PyArrayDyn::from_owned_array(py, $a).into_any().unbind()
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
        TypedArray::Str(a) => {
            let objs = a.mapv(|s| PyString::new(py, &s).into_any().unbind());
            PyArrayDyn::from_owned_object_array(py, objs)
                .into_any()
                .unbind()
        }
        other => {
            return Err(value_error(format!(
                "raw data of type {} (expected stored values)",
                other.dtype()
            )));
        }
    })
}

/// numpy's name for a stored element type (`S1` for netCDF characters, `O` for strings).
fn numpy_dtype(d: DType, is_char: bool) -> &'static str {
    match d {
        DType::U8 if is_char => "S1",
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
        _ => "object",
    }
}

fn compression_name(c: Compression) -> String {
    match c {
        Compression::Zstd { .. } => "zstd".into(),
        Compression::Lz4 => "lz4".into(),
        Compression::Blosc { codec, .. } => match codec {
            BloscCodec::BloscLz => "blosc_lz".into(),
            BloscCodec::Lz4 => "blosc_lz4".into(),
            BloscCodec::Lz4Hc => "blosc_lz4hc".into(),
            BloscCodec::Zlib => "blosc_zlib".into(),
            BloscCodec::Zstd => "blosc_zstd".into(),
            _ => "blosc".into(),
        },
        _ => "other".into(),
    }
}

/// A group of a file opened lazily, serving raw variables.
#[pyclass(frozen, module = "annoray._annoray")]
pub(crate) struct RawGroup {
    root: Arc<RawDataset<'static>>,
    path: String,
}

impl RawGroup {
    fn group(&self) -> PyResult<&RawDataset<'static>> {
        self.root.group(&self.path).map_err(value_error)
    }

    fn var(&self, name: &str) -> PyResult<&annoray::io::RawVariable<'static>> {
        self.group()?
            .vars
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("no variable {name:?}")))
    }
}

#[pymethods]
impl RawGroup {
    /// Open a netCDF-3, netCDF-4 or HDF5 file (memory-mapped; data read on demand).
    #[staticmethod]
    #[pyo3(signature = (path, group=None))]
    fn open(py: Python<'_>, path: PathBuf, group: Option<String>) -> PyResult<Self> {
        let raw = py
            .detach(|| default_backend().read_lazy(&path))
            .map_err(|e| io_error(format!("{}: {e}", path.display())))?;
        let g = Self {
            root: Arc::new(raw),
            path: group.unwrap_or_default(),
        };
        g.group()?;
        Ok(g)
    }

    /// Dimensions: name → (length, unlimited).
    #[getter]
    fn dims<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.group()?.dims {
            d.set_item(k.as_str(), (v.len, v.unlimited))?;
        }
        Ok(d)
    }

    /// Group attributes.
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        attrs_to_py(py, &self.group()?.attrs)
    }

    /// Variable names, in file order.
    #[getter]
    fn variables(&self) -> PyResult<Vec<String>> {
        Ok(self.group()?.vars.keys().map(ToString::to_string).collect())
    }

    /// Child group names.
    #[getter]
    fn groups(&self) -> PyResult<Vec<String>> {
        Ok(self
            .group()?
            .groups
            .keys()
            .map(ToString::to_string)
            .collect())
    }

    /// The `/`-separated path of this group (`""` for the root).
    #[getter]
    fn path(&self) -> String {
        self.path.clone()
    }

    /// Variables the reader could not represent, with the reason.
    #[getter]
    fn skipped(&self) -> PyResult<Vec<(String, String)>> {
        Ok(self
            .group()?
            .skipped
            .iter()
            .map(|(n, w)| (n.to_string(), w.clone()))
            .collect())
    }

    /// A child group.
    fn child(&self, name: &str) -> PyResult<Self> {
        let path = format!("{}/{name}", self.path);
        let g = Self {
            root: self.root.clone(),
            path,
        };
        g.group()?;
        Ok(g)
    }

    /// A variable's dims, shape, numpy dtype, attributes and storage details.
    fn info<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let v = self.var(name)?;
        let d = PyDict::new(py);
        d.set_item(
            "dims",
            PyTuple::new(py, v.dims.iter().map(|x| x.to_string()))?,
        )?;
        d.set_item("shape", PyTuple::new(py, v.data.shape())?)?;
        d.set_item("dtype", numpy_dtype(v.data.dtype(), v.is_char))?;
        d.set_item("attrs", attrs_to_py(py, &v.attrs)?)?;
        d.set_item("chunks", v.storage.chunks.clone())?;
        d.set_item("deflate", v.storage.deflate)?;
        d.set_item("shuffle", v.storage.shuffle)?;
        d.set_item("compression", v.storage.compression.map(compression_name))?;
        Ok(d)
    }

    /// Read the box `[start, start + count)` of a variable (only that part is read).
    fn read(
        &self,
        py: Python<'_>,
        name: &str,
        start: Vec<usize>,
        count: Vec<usize>,
    ) -> PyResult<Py<PyAny>> {
        let v = self.var(name)?;
        let data = &v.data;
        let t = py
            .detach(|| data.read_slice(&start, &count).map_err(|e| e.to_string()))
            .map_err(io_error)?;
        array_to_py(py, t)
    }
}

// ----- Python → Rust (writing) -----

fn numeric_from_py(data: &Bound<'_, PyAny>) -> PyResult<TypedArray<Owned>> {
    macro_rules! try_as {
        ($($t:ty => $var:ident),*) => {
            $(if let Ok(a) = data.cast::<PyArrayDyn<$t>>() {
                return Ok(TypedArray::$var(a.readonly().as_array().to_owned()));
            })*
        };
    }
    try_as!(f64 => F64, f32 => F32, i64 => I64, i32 => I32, i16 => I16, i8 => I8,
        u64 => U64, u32 => U32, u16 => U16, u8 => U8, bool => Bool);
    Err(value_error(
        "unsupported array type (expected numbers or booleans)",
    ))
}

fn i64_array(data: &Bound<'_, PyAny>) -> PyResult<ArrayD<i64>> {
    Ok(data
        .cast::<PyArrayDyn<i64>>()
        .map_err(|_| value_error("expected int64 values"))?
        .readonly()
        .as_array()
        .to_owned())
}

/// A variable's values from what the Python side prepared.
fn values_from_py(v: &Bound<'_, PyDict>) -> PyResult<TypedArray<Owned>> {
    let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
        v.get_item(k)?
            .ok_or_else(|| PyValueError::new_err(format!("missing {k:?}")))
    };
    let kind: String = get("kind")?.extract()?;
    let data = get("data")?;
    Ok(match kind.as_str() {
        "num" => numeric_from_py(&data)?,
        // Nanoseconds since the epoch; i64::MIN is NaT.
        "datetime" => {
            let epoch = chrono::DateTime::UNIX_EPOCH.naive_utc();
            TypedArray::DateTime(i64_array(&data)?.mapv(|ns| {
                if ns == i64::MIN {
                    NaiveDateTime::NAT
                } else {
                    epoch
                        .checked_add_signed(TimeDelta::nanoseconds(ns))
                        .unwrap_or(NaiveDateTime::NAT)
                }
            }))
        }
        "timedelta" => TypedArray::TimeDelta(i64_array(&data)?.mapv(|ns| {
            if ns == i64::MIN {
                TimeDelta::NAT
            } else {
                TimeDelta::nanoseconds(ns)
            }
        })),
        "str" => {
            let flat: Vec<String> = data.extract()?;
            let shape: Vec<usize> = get("shape")?.extract()?;
            TypedArray::Str(ArrayD::from_shape_vec(IxDyn(&shape), flat).map_err(value_error)?)
        }
        // Components of cftime dates (year, month, day, hour, minute, second, microsecond),
        // -1 years for missing values.
        "cftime" => {
            let calendar: Calendar = get("calendar")?
                .extract::<String>()?
                .parse()
                .map_err(value_error)?;
            let parts: Vec<Vec<i64>> = data.extract()?;
            let shape: Vec<usize> = get("shape")?.extract()?;
            let n = parts.first().map_or(0, Vec::len);
            let mut dates = Vec::with_capacity(n);
            for i in 0..n {
                let p = |k: usize| parts.get(k).and_then(|c| c.get(i)).copied().unwrap_or(0);
                let small = |k: usize| u8::try_from(p(k)).map_err(value_error);
                if p(7) != 0 {
                    dates.push(CfDate::NAT);
                    continue;
                }
                let d = CfDate::from_ymd(calendar, p(0), small(1)?, small(2)?)
                    .and_then(|d| {
                        d.and_hms_nano(
                            u8::try_from(p(3)).unwrap_or(0),
                            u8::try_from(p(4)).unwrap_or(0),
                            u8::try_from(p(5)).unwrap_or(0),
                            u32::try_from(p(6).saturating_mul(1000)).unwrap_or(0),
                        )
                    })
                    .map_err(value_error)?;
                dates.push(d);
            }
            TypedArray::CfDate(ArrayD::from_shape_vec(IxDyn(&shape), dates).map_err(value_error)?)
        }
        other => return Err(value_error(format!("unknown value kind {other:?}"))),
    })
}

pub(crate) fn dtype_from_name(name: &str) -> Option<DType> {
    Some(match name {
        "float32" => DType::F32,
        "float64" => DType::F64,
        "int8" => DType::I8,
        "int16" => DType::I16,
        "int32" => DType::I32,
        "int64" => DType::I64,
        "uint8" => DType::U8,
        "uint16" => DType::U16,
        "uint32" => DType::U32,
        "uint64" => DType::U64,
        "bool" => DType::Bool,
        "str" | "object" => DType::Str,
        // numpy's fixed-width text (`<U3` is named "str96").
        t if t.starts_with("str") => DType::Str,
        _ => return None,
    })
}

/// Our `Encoding` as xarray's encoding dict (netCDF4-python names); settings left at their
/// defaults are omitted.
pub(crate) fn encoding_to_py<'py>(
    py: Python<'py>,
    enc: &annoray::Encoding,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    if let Some(t) = enc.dtype {
        d.set_item("dtype", numpy_dtype(t, enc.char_dim.is_some()))?;
    }
    for (key, slot) in [
        ("_FillValue", &enc.fill_value),
        ("missing_value", &enc.missing_value),
        ("scale_factor", &enc.scale_factor),
        ("add_offset", &enc.add_offset),
    ] {
        if let Some(v) = slot {
            d.set_item(key, crate::convert::attr_to_py(py, v)?)?;
        }
    }
    if enc.unsigned {
        d.set_item("_Unsigned", "true")?;
    }
    if let Some(u) = &enc.time_units {
        d.set_item("units", u)?;
    }
    if let Some(c) = &enc.calendar {
        d.set_item("calendar", c)?;
    }
    if let Some(c) = &enc.char_dim {
        d.set_item("char_dim_name", c.as_str())?;
    }
    if let Some(c) = &enc.chunks {
        d.set_item("chunksizes", PyTuple::new(py, c)?)?;
    }
    if let Some(level) = enc.deflate {
        d.set_item("zlib", true)?;
        d.set_item("complevel", level)?;
    }
    if enc.shuffle {
        d.set_item("shuffle", true)?;
    }
    if let Some(c) = enc.compression {
        d.set_item("compression", compression_name(c))?;
        match c {
            Compression::Zstd { level } => d.set_item("complevel", level)?,
            Compression::Blosc { level, .. } => d.set_item("complevel", level)?,
            _ => {}
        }
    }
    Ok(d)
}

/// Reset the setting under xarray's encoding key `key` to its default.
pub(crate) fn clear_encoding_key(enc: &mut annoray::Encoding, key: &str) -> PyResult<()> {
    match key {
        "dtype" => enc.dtype = None,
        "_FillValue" => enc.fill_value = None,
        "missing_value" => enc.missing_value = None,
        "scale_factor" => enc.scale_factor = None,
        "add_offset" => enc.add_offset = None,
        "_Unsigned" => enc.unsigned = false,
        "units" => enc.time_units = None,
        "calendar" => enc.calendar = None,
        "char_dim_name" => enc.char_dim = None,
        "chunksizes" => enc.chunks = None,
        "zlib" => enc.deflate = None,
        "shuffle" => enc.shuffle = false,
        "compression" => enc.compression = None,
        // Read back with the codec it qualifies.
        "complevel" => {}
        other => {
            return Err(value_error(format!("unknown encoding key {other:?}")));
        }
    }
    Ok(())
}

/// xarray's encoding dict (netCDF4-python names) as our `Encoding`.
pub(crate) fn encoding_from_py(e: &Bound<'_, PyDict>) -> PyResult<annoray::Encoding> {
    let mut enc = annoray::Encoding::default();
    update_encoding(&mut enc, e)?;
    Ok(enc)
}

/// Apply the settings in `e` (netCDF4-python names) to `enc`; settings not given are kept.
pub(crate) fn update_encoding(enc: &mut annoray::Encoding, e: &Bound<'_, PyDict>) -> PyResult<()> {
    if let Some(d) = e.get_item("dtype")? {
        enc.dtype = dtype_from_name(&d.extract::<String>()?);
    }
    for (key, slot) in [
        ("_FillValue", &mut enc.fill_value),
        ("missing_value", &mut enc.missing_value),
        ("scale_factor", &mut enc.scale_factor),
        ("add_offset", &mut enc.add_offset),
    ] {
        if let Some(v) = e.get_item(key)?.filter(|v| !v.is_none()) {
            *slot = Some(attr_from_py(&v)?);
        }
    }
    if let Some(u) = e.get_item("units")? {
        enc.time_units = Some(u.extract()?);
    }
    if let Some(c) = e.get_item("calendar")? {
        enc.calendar = Some(c.extract()?);
    }
    if let Some(c) = e.get_item("chunksizes")?.filter(|v| !v.is_none()) {
        enc.chunks = Some(c.extract()?);
    }
    let level = e
        .get_item("complevel")?
        .map(|l| l.extract::<u8>())
        .transpose()?
        .unwrap_or(4);
    match e.get_item("zlib")?.map(|z| z.is_truthy()).transpose()? {
        Some(true) => enc.deflate = Some(level),
        Some(false) => enc.deflate = None,
        None => {}
    }
    if let Some(u) = e.get_item("_Unsigned")? {
        enc.unsigned = u.extract::<String>().is_ok_and(|s| s == "true");
    }
    if let Some(c) = e.get_item("char_dim_name")?.filter(|v| !v.is_none()) {
        enc.char_dim = Some(c.extract::<String>()?.into());
    }
    if let Some(s) = e.get_item("shuffle")? {
        enc.shuffle = s.is_truthy()?;
    }
    if let Some(c) = e.get_item("compression")?.filter(|v| !v.is_none()) {
        let blosc = |codec| Compression::Blosc {
            codec,
            level: level.min(9),
            shuffle: BloscShuffle::Byte,
        };
        match c.extract::<String>()?.as_str() {
            "zlib" => enc.deflate = Some(level),
            "zstd" => {
                enc.compression = Some(Compression::Zstd {
                    level: i32::from(level),
                })
            }
            "lz4" => enc.compression = Some(Compression::Lz4),
            "blosc_lz4" => enc.compression = Some(blosc(BloscCodec::Lz4)),
            "blosc_lz4hc" => enc.compression = Some(blosc(BloscCodec::Lz4Hc)),
            "blosc_zlib" => enc.compression = Some(blosc(BloscCodec::Zlib)),
            "blosc_zstd" => enc.compression = Some(blosc(BloscCodec::Zstd)),
            other => {
                return Err(value_error(format!(
                    "compression {other:?} cannot be written"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn format_from_name(name: &str) -> PyResult<Format> {
    Ok(match name.to_ascii_uppercase().as_str() {
        "NETCDF4" => Format::Netcdf4,
        "NETCDF4_CLASSIC" => Format::Netcdf4Classic,
        "NETCDF3_CLASSIC" => Format::Netcdf3Classic,
        "NETCDF3_64BIT" | "NETCDF3_64BIT_OFFSET" => Format::Netcdf3Offset64,
        "NETCDF3_64BIT_DATA" => Format::Netcdf3Data64,
        other => return Err(value_error(format!("unknown format {other:?}"))),
    })
}

/// Write variables prepared by `annoray.to_netcdf` with annoray's CF encoder and writers.
#[pyfunction]
#[pyo3(signature = (path, format, attrs, variables, unlimited_dims))]
pub(crate) fn write(
    py: Python<'_>,
    path: PathBuf,
    format: &str,
    attrs: &Bound<'_, PyDict>,
    variables: Vec<Bound<'_, PyDict>>,
    unlimited_dims: Vec<String>,
) -> PyResult<()> {
    let mut ds = Dataset::new();
    for (k, v) in attrs.iter() {
        ds.attrs_mut()
            .insert(k.extract::<String>()?.into(), attr_from_py(&v)?);
    }
    for v in &variables {
        let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
            v.get_item(k)?
                .ok_or_else(|| PyValueError::new_err(format!("missing {k:?}")))
        };
        let name: String = get("name")?.extract()?;
        let dims: Vec<String> = get("dims")?.extract()?;
        let mut var = Variable::from_typed(dims, values_from_py(v)?)
            .map_err(value_error)?
            .with_encoding(encoding_from_py(get("encoding")?.cast::<PyDict>()?)?);
        for (k, a) in get("attrs")?.cast::<PyDict>()?.iter() {
            var.attrs_mut()
                .insert(k.extract::<String>()?.into(), attr_from_py(&a)?);
        }
        if get("coord")?.is_truthy()? {
            ds.add_coord(name, var).map_err(value_error)?;
        } else {
            ds.add_var(name, var).map_err(value_error)?;
        }
    }
    ds.encoding_mut().unlimited_dims = unlimited_dims.into_iter().map(Into::into).collect();
    let opts = WriteOptions::format(format_from_name(format)?);
    py.detach(|| annoray::io::to_netcdf_with_backend(default_backend(), &ds, &path, &opts))
        .map_err(|e| io_error(format!("{}: {e}", path.display())))
}
