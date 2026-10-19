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
    if enc.float16 {
        d.set_item("dtype", "float16")?;
    } else if let (Some(t), Some(e)) = (enc.dtype, &enc.enumeration) {
        // An enumeration, as xarray keeps it: in the dtype's metadata.
        let members = PyDict::new(py);
        for (n, v) in &e.members {
            members.set_item(n, v)?;
        }
        let meta = PyDict::new(py);
        meta.set_item("enum", members)?;
        meta.set_item("enum_name", e.name.as_deref())?;
        let kw = PyDict::new(py);
        kw.set_item("metadata", meta)?;
        let dtype = py
            .import("numpy")?
            .getattr("dtype")?
            .call((numpy_dtype(t, false),), Some(&kw))?;
        d.set_item("dtype", dtype)?;
    } else if let Some(t) = enc.dtype {
        d.set_item("dtype", numpy_dtype(t, enc.char_dim.is_some()))?;
    }
    for (key, value) in &enc.cf_related {
        d.set_item(key, value)?;
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
            Compression::Blosc { level, shuffle, .. } => {
                d.set_item("complevel", level)?;
                d.set_item(
                    "blosc_shuffle",
                    match shuffle {
                        BloscShuffle::None => 0,
                        BloscShuffle::Bit => 2,
                        _ => 1,
                    },
                )?;
            }
            _ => {}
        }
    }
    if enc.fletcher32 {
        d.set_item("fletcher32", true)?;
    }
    if enc.contiguous {
        d.set_item("contiguous", true)?;
    }
    if enc.big_endian {
        d.set_item("endian", "big")?;
    }
    if enc.no_fill_value {
        d.set_item("_FillValue", py.None())?;
    }
    if let Some(q) = enc.quantize {
        d.set_item("significant_digits", q.nsd)?;
        d.set_item("quantize_mode", quantize_mode_name(q.mode))?;
    }
    if let Some(l) = enc.least_significant_digit {
        d.set_item("least_significant_digit", l)?;
    }
    if let Some(c) = &enc.ragged {
        d.set_item("ragged_counts", c.as_str())?;
    }
    Ok(d)
}

fn quantize_mode_name(m: annoray::cf::QuantizeMode) -> &'static str {
    match m {
        annoray::cf::QuantizeMode::GranularBitRound => "GranularBitRound",
        annoray::cf::QuantizeMode::BitRound => "BitRound",
        _ => "BitGroom",
    }
}

/// Encoding keys (xarray's, netCDF4-python names) a netCDF write understands.
pub(crate) const NETCDF_ENCODING_KEYS: &[&str] = &[
    "dtype",
    "_FillValue",
    "missing_value",
    "scale_factor",
    "add_offset",
    "units",
    "calendar",
    "_Unsigned",
    "char_dim_name",
    "chunksizes",
    "contiguous",
    "zlib",
    "complevel",
    "shuffle",
    "fletcher32",
    "compression",
    "blosc_shuffle",
    "szip_coding",
    "szip_pixels_per_block",
    "endian",
    "least_significant_digit",
    "significant_digits",
    "quantize_mode",
    // annoray: write this CF ragged array's values as a variable-length variable, with the
    // lengths in the named count variable.
    "ragged_counts",
    // CF attributes naming other variables, moved to the encoding by
    // `decode_coords="all"` (written back as attributes).
    "bounds",
    "grid_mapping",
    "climatology",
    "geometry",
    "node_coordinates",
    "node_count",
    "part_node_count",
    "interior_ring",
    "cell_measures",
    "formula_terms",
];

/// Refuse keys of an explicit `encoding=` that are unknown, or that `format` cannot honour
/// (as xarray refuses unknown ones; netCDF4-python would silently drop the rest).
pub(crate) fn check_encoding(
    var: &str,
    e: &Bound<'_, PyDict>,
    valid: &[&str],
    format: Option<Format>,
) -> PyResult<()> {
    let mut unknown: Vec<String> = Vec::new();
    for k in e.keys() {
        let k: String = k.extract()?;
        if !valid.contains(&k.as_str()) {
            unknown.push(k);
        }
    }
    if !unknown.is_empty() {
        return Err(value_error(format!(
            "unexpected encoding parameters for variable {var:?}: {unknown:?}; valid ones are \
             {valid:?}"
        )));
    }
    let on = |k: &str| -> PyResult<bool> {
        Ok(match e.get_item(k)? {
            None => false,
            Some(v) if v.is_none() => false,
            Some(v) => v.is_truthy()?,
        })
    };
    if on("contiguous")? {
        for k in ["zlib", "shuffle", "fletcher32", "compression", "chunksizes"] {
            if on(k)? {
                return Err(value_error(format!(
                    "variable {var:?}: contiguous storage cannot be combined with {k} \
                     (compression, checksums and chunk shapes need chunked storage)"
                )));
            }
        }
    }
    if format.is_some_and(Format::is_netcdf3) {
        for k in [
            "zlib",
            "shuffle",
            "fletcher32",
            "compression",
            "chunksizes",
            "contiguous",
            "szip_coding",
        ] {
            if on(k)? {
                return Err(value_error(format!(
                    "variable {var:?}: {k} needs a netCDF-4 file (netCDF-3 has no chunking, \
                     compression or checksums)"
                )));
            }
        }
        if let Some(v) = e.get_item("endian")?
            && v.extract::<String>().is_ok_and(|s| s == "little")
        {
            return Err(value_error(format!(
                "variable {var:?}: netCDF-3 stores numbers big-endian"
            )));
        }
    }
    Ok(())
}

/// Reset the setting under xarray's encoding key `key` to its default.
pub(crate) fn clear_encoding_key(enc: &mut annoray::Encoding, key: &str) -> PyResult<()> {
    match key {
        "dtype" => {
            enc.dtype = None;
            enc.enumeration = None;
        }
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
        "fletcher32" => enc.fletcher32 = false,
        "contiguous" => enc.contiguous = false,
        "endian" => enc.big_endian = false,
        "significant_digits" | "quantize_mode" => enc.quantize = None,
        "least_significant_digit" => enc.least_significant_digit = None,
        "ragged_counts" => enc.ragged = None,
        k if annoray::cf::CF_RELATED.contains(&k) => enc.cf_related.retain(|(a, _)| a != k),
        // Read back with the codec they qualify.
        "complevel" | "blosc_shuffle" | "szip_coding" | "szip_pixels_per_block" => {}
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
    for key in annoray::cf::CF_RELATED {
        if let Some(v) = e.get_item(key)? {
            enc.cf_related.retain(|(a, _)| a != key);
            if !v.is_none() {
                enc.cf_related.push((key.to_owned(), v.extract()?));
            }
        }
    }
    if let Some(d) = e.get_item("dtype")? {
        // A name, or a numpy dtype (with an enumeration in its metadata, as xarray has it).
        let (name, enumeration) = match d.extract::<String>() {
            Ok(name) => (name, None),
            Err(_) => {
                let name: String = d.getattr("name")?.extract()?;
                let meta = d.getattr("metadata").ok().filter(|m| !m.is_none());
                let members = meta
                    .as_ref()
                    .map(|m| m.get_item("enum"))
                    .and_then(Result::ok)
                    .filter(|m| !m.is_none());
                let enumeration = match members {
                    Some(m) => {
                        let mut out = Vec::new();
                        for item in m.call_method0("items")?.try_iter()? {
                            let (n, v): (String, i128) = item?.extract()?;
                            out.push((n, v));
                        }
                        let type_name = meta
                            .as_ref()
                            .map(|x| x.get_item("enum_name"))
                            .and_then(Result::ok)
                            .filter(|x| !x.is_none())
                            .map(|x| x.extract::<String>())
                            .transpose()?;
                        Some(annoray::io::EnumType {
                            name: type_name,
                            members: out,
                        })
                    }
                    None => None,
                };
                (name, enumeration)
            }
        };
        if enumeration.is_some() && !name.contains("int") {
            return Err(value_error(format!(
                "an enumeration's dtype must be an integer type, not {name}"
            )));
        }
        enc.enumeration = enumeration;
        // No float16 in memory: the values stay f32/f64 and are stored as halves.
        enc.float16 = name == "float16";
        enc.dtype = if enc.float16 {
            None
        } else {
            dtype_from_name(&name)
        };
    }
    // `_FillValue: None` asks for no fill value at all (xarray's meaning).
    match e.get_item("_FillValue")? {
        Some(v) if v.is_none() => {
            enc.fill_value = None;
            enc.no_fill_value = true;
        }
        Some(v) => {
            enc.fill_value = Some(attr_from_py(&v)?);
            enc.no_fill_value = false;
        }
        None => {}
    }
    for (key, slot) in [
        ("missing_value", &mut enc.missing_value),
        ("scale_factor", &mut enc.scale_factor),
        ("add_offset", &mut enc.add_offset),
    ] {
        if let Some(v) = e.get_item(key)?.filter(|v| !v.is_none()) {
            *slot = Some(attr_from_py(&v)?);
        }
    }
    if let Some(f) = e.get_item("fletcher32")? {
        enc.fletcher32 = f.is_truthy()?;
    }
    if let Some(c) = e.get_item("contiguous")? {
        enc.contiguous = c.is_truthy()?;
    } else {
        // Asking for chunked storage (compression, checksums, a chunk shape) drops the
        // contiguous layout a file was read with.
        for k in ["zlib", "shuffle", "fletcher32", "compression", "chunksizes"] {
            if e.get_item(k)?
                .is_some_and(|v| !v.is_none() && v.is_truthy().unwrap_or(false))
            {
                enc.contiguous = false;
            }
        }
    }
    if let Some(x) = e.get_item("endian")?.filter(|v| !v.is_none()) {
        enc.big_endian = match x.extract::<String>()?.as_str() {
            "big" => true,
            "little" => false,
            "native" => cfg!(target_endian = "big"),
            other => {
                return Err(value_error(format!(
                    "endian must be \"native\", \"little\" or \"big\", not {other:?}"
                )));
            }
        };
    }
    if let Some(l) = e
        .get_item("least_significant_digit")?
        .filter(|v| !v.is_none())
    {
        enc.least_significant_digit = Some(l.extract()?);
    }
    if let Some(r) = e.get_item("ragged_counts")? {
        enc.ragged = if r.is_none() {
            None
        } else {
            Some(r.extract::<String>()?.into())
        };
    }
    let nsd = e
        .get_item("significant_digits")?
        .filter(|v| !v.is_none())
        .map(|v| v.extract::<u8>())
        .transpose()?;
    let mode = e.get_item("quantize_mode")?.filter(|v| !v.is_none());
    if let Some(nsd) = nsd {
        let mode = match mode.map(|m| m.extract::<String>()).transpose()?.as_deref() {
            None | Some("BitGroom") => annoray::cf::QuantizeMode::BitGroom,
            Some("GranularBitRound") => annoray::cf::QuantizeMode::GranularBitRound,
            Some("BitRound") => annoray::cf::QuantizeMode::BitRound,
            Some(other) => {
                return Err(value_error(format!(
                    "quantize_mode must be \"BitGroom\", \"GranularBitRound\" or \"BitRound\", \
                     not {other:?}"
                )));
            }
        };
        let q = annoray::cf::Quantize { mode, nsd };
        if !q.valid_for(52) {
            return Err(value_error(format!(
                "significant_digits={nsd} is out of range for {}",
                quantize_mode_name(mode)
            )));
        }
        enc.quantize = Some(q);
    } else if mode.is_some() {
        return Err(value_error("quantize_mode needs significant_digits"));
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
    let blosc_shuffle = match e
        .get_item("blosc_shuffle")?
        .filter(|v| !v.is_none())
        .map(|v| v.extract::<u8>())
        .transpose()?
    {
        None | Some(1) => BloscShuffle::Byte,
        Some(0) => BloscShuffle::None,
        Some(2) => BloscShuffle::Bit,
        Some(other) => {
            return Err(value_error(format!(
                "blosc_shuffle must be 0 (none), 1 (byte) or 2 (bit), not {other}"
            )));
        }
    };
    if let Some(c) = e.get_item("compression")?.filter(|v| !v.is_none()) {
        let blosc = |codec| Compression::Blosc {
            codec,
            level: level.min(9),
            shuffle: blosc_shuffle,
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
/// Validate the explicit `encoding` of variable `var` for a netCDF write in `format`.
#[pyfunction]
#[pyo3(name = "_check_encoding")]
pub(crate) fn check_encoding_py(
    var: &str,
    encoding: &Bound<'_, PyDict>,
    format: &str,
) -> PyResult<()> {
    check_encoding(
        var,
        encoding,
        NETCDF_ENCODING_KEYS,
        Some(format_from_name(format)?),
    )
}

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
