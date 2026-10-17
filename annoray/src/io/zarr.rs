//! Zarr v2 and v3 stores (local directories), through [`annoray_zarr`], as xarray's zarr
//! backend lays them out.
//!
//! - Dimensions are the v2 `_ARRAY_DIMENSIONS` attribute or v3 `dimension_names`; arrays
//!   without them are skipped (with a warning), as xarray refuses them.
//! - `_FillValue` follows xarray: in v2 it is the array's `fill_value`; in v3 it is an
//!   attribute (base64 of a little-endian `f64` for floats) and the array's `fill_value` is NaN
//!   for floats and zero otherwise.
//! - Reading is lazy and chunk-aware (only the chunks a selection touches are decoded);
//!   writing goes chunk by chunk, so lazy data is never loaded whole. Variables keep their
//!   chunk shape from the encoding (`chunks`); otherwise chunks of about 1 MiB are chosen.
//!   Groups map to subgroups. Consolidated metadata is written by default (xarray and
//!   zarr-python then open the store with one read).
//! - CF decoding and encoding (times, packing, units) are the netCDF ones: this module only
//!   moves the raw data model in and out.

use std::path::Path;
use std::sync::Arc;

pub use annoray_zarr::ZarrFormat;
use annoray_zarr::{
    Array, ArrayBuilder, ArrayData, Compressor, DType as ZType, Fill, Group, Map, Node, Number,
    Store, Value,
};

use super::raw::{IoError, RawData, RawDataset, RawDim, RawVariable, Result, Storage};
use crate::{AttrValue, Attrs, CoreError, DType, LazySource, Name, Owned, TypedArray};

fn err(e: annoray_zarr::Error) -> IoError {
    match e {
        annoray_zarr::Error::Unsupported(m) => IoError::Unsupported(m),
        annoray_zarr::Error::NotFound(m) => IoError::NoSuchGroup(m),
        other => IoError::Backend {
            backend: "zarr",
            message: other.to_string(),
        },
    }
}

fn bad(message: impl Into<String>) -> IoError {
    IoError::Backend {
        backend: "zarr",
        message: message.into(),
    }
}

/// Whether `path` is a Zarr store (a directory with v2 or v3 root metadata).
#[must_use]
pub fn is_zarr(path: &Path) -> bool {
    path.is_dir()
        && ["zarr.json", ".zgroup", ".zarray"]
            .iter()
            .any(|f| path.join(f).is_file())
}

// ----- attributes -----

fn attr_from_json(v: &Value) -> Option<AttrValue> {
    let num = |n: &Number| -> Option<AttrValue> {
        n.as_i64()
            .map(|i| AttrValue::I64(vec![i]))
            .or_else(|| n.as_u64().map(|u| AttrValue::U64(vec![u])))
            .or_else(|| n.as_f64().map(|f| AttrValue::F64(vec![f])))
    };
    Some(match v {
        Value::String(s) => AttrValue::Str(s.clone()),
        Value::Number(n) => num(n)?,
        Value::Bool(b) => AttrValue::I8(vec![i8::from(*b)]),
        Value::Array(items) if items.iter().all(Value::is_string) => AttrValue::Strs(
            items
                .iter()
                .filter_map(|x| x.as_str().map(str::to_owned))
                .collect(),
        ),
        Value::Array(items) if items.iter().all(|x| x.as_i64().is_some()) => {
            AttrValue::I64(items.iter().filter_map(Value::as_i64).collect())
        }
        Value::Array(items)
            if items
                .iter()
                .all(|x| x.is_number() || x.as_str().is_some_and(is_special)) =>
        {
            AttrValue::F64(
                items
                    .iter()
                    .map(|x| {
                        x.as_f64()
                            .unwrap_or_else(|| special(x.as_str().unwrap_or("")))
                    })
                    .collect(),
            )
        }
        Value::Null => return None,
        // Objects and mixed lists: kept as JSON text.
        other => AttrValue::Str(other.to_string()),
    })
}

fn is_special(s: &str) -> bool {
    matches!(s, "NaN" | "Infinity" | "-Infinity")
}

fn special(s: &str) -> f64 {
    match s {
        "Infinity" => f64::INFINITY,
        "-Infinity" => f64::NEG_INFINITY,
        _ => f64::NAN,
    }
}

fn attr_to_json(a: &AttrValue) -> Value {
    fn f(x: f64) -> Value {
        if x.is_nan() {
            Value::String("NaN".into())
        } else if x.is_infinite() {
            Value::String(if x > 0.0 { "Infinity" } else { "-Infinity" }.into())
        } else {
            Number::from_f64(x).map_or(Value::Null, Value::Number)
        }
    }
    fn list<T: Into<Value> + Clone>(v: &[T]) -> Value {
        match v {
            [one] => one.clone().into(),
            _ => Value::Array(v.iter().cloned().map(Into::into).collect()),
        }
    }
    match a {
        AttrValue::Str(s) => Value::String(s.clone()),
        AttrValue::Strs(v) => Value::Array(v.iter().cloned().map(Value::String).collect()),
        AttrValue::I8(v) => list(v),
        AttrValue::I16(v) => list(v),
        AttrValue::I32(v) => list(v),
        AttrValue::I64(v) => list(v),
        AttrValue::U8(v) => list(v),
        AttrValue::U16(v) => list(v),
        AttrValue::U32(v) => list(v),
        AttrValue::U64(v) => list(v),
        AttrValue::F32(v) => match v.as_slice() {
            [one] => f(f64::from(*one)),
            _ => Value::Array(v.iter().map(|x| f(f64::from(*x))).collect()),
        },
        AttrValue::F64(v) => match v.as_slice() {
            [one] => f(*one),
            _ => Value::Array(v.iter().map(|x| f(*x)).collect()),
        },
    }
}

fn attrs_from(m: &Map<String, Value>) -> Attrs {
    m.iter()
        .filter_map(|(k, v)| attr_from_json(v).map(|a| (Name::from(k.as_str()), a)))
        .collect()
}

fn attrs_to(a: &Attrs) -> Map<String, Value> {
    a.iter()
        .map(|(k, v)| (k.to_string(), attr_to_json(v)))
        .collect()
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// The shift of 6-bit digit `i` (of 4) in a 24-bit group.
fn shift(i: usize) -> u32 {
    18u32.saturating_sub(6u32.saturating_mul(u32::try_from(i).unwrap_or(0)))
}

/// Standard base64 (RFC 4648), as xarray encodes v3 float fill values.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let digits: Vec<u32> = s
        .bytes()
        .filter(|b| *b != b'=')
        .map(|c| {
            BASE64
                .iter()
                .position(|a| *a == c)
                .and_then(|p| u32::try_from(p).ok())
        })
        .collect::<Option<_>>()?;
    let mut out = Vec::new();
    for chunk in digits.chunks(4) {
        let acc = chunk
            .iter()
            .enumerate()
            .fold(0u32, |a, (i, d)| a | d.wrapping_shl(shift(i)));
        out.extend(
            acc.to_be_bytes()
                .iter()
                .skip(1)
                .take(chunk.len().saturating_sub(1)),
        );
    }
    Some(out)
}

fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let acc = chunk.iter().enumerate().fold(0u32, |a, (i, b)| {
            a | u32::from(*b).wrapping_shl(
                16u32.saturating_sub(8u32.saturating_mul(u32::try_from(i).unwrap_or(0))),
            )
        });
        for i in 0..4 {
            if i <= chunk.len() {
                let idx = usize::try_from(acc.wrapping_shr(shift(i)) & 63).unwrap_or(0);
                out.push(char::from(BASE64.get(idx).copied().unwrap_or(b'A')));
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// A fill value as an attribute of the variable's type.
fn fill_attr(fill: &Fill, d: DType) -> Option<AttrValue> {
    #[allow(clippy::cast_possible_truncation)]
    Some(match (fill, d) {
        (Fill::Float(v), DType::F32) => AttrValue::F32(vec![*v as f32]),
        (Fill::Float(v), _) => AttrValue::F64(vec![*v]),
        (Fill::Int(v), DType::I8) => AttrValue::I8(vec![i8::try_from(*v).ok()?]),
        (Fill::Int(v), DType::I16) => AttrValue::I16(vec![i16::try_from(*v).ok()?]),
        (Fill::Int(v), DType::I32) => AttrValue::I32(vec![i32::try_from(*v).ok()?]),
        (Fill::Int(v), DType::I64) => AttrValue::I64(vec![i64::try_from(*v).ok()?]),
        (Fill::Int(v), DType::U8) => AttrValue::U8(vec![u8::try_from(*v).ok()?]),
        (Fill::Int(v), DType::U16) => AttrValue::U16(vec![u16::try_from(*v).ok()?]),
        (Fill::Int(v), DType::U32) => AttrValue::U32(vec![u32::try_from(*v).ok()?]),
        (Fill::Int(v), DType::U64) => AttrValue::U64(vec![u64::try_from(*v).ok()?]),
        (Fill::Str(s), _) => AttrValue::Str(s.clone()),
        _ => return None,
    })
}

/// A `_FillValue` attribute as a Zarr fill value for `d`.
fn fill_of(a: &AttrValue, d: ZType) -> Option<Fill> {
    let first_f = || -> Option<f64> {
        Some(match a {
            AttrValue::F32(v) => f64::from(*v.first()?),
            AttrValue::F64(v) => *v.first()?,
            _ => crate::cf::num::attr_f64(a)?,
        })
    };
    Some(match d {
        ZType::F32 | ZType::F64 => Fill::Float(first_f()?),
        ZType::Bool => Fill::Bool(first_f()? != 0.0),
        ZType::String | ZType::FixedBytes(_) | ZType::FixedUnicode(_) => match a {
            AttrValue::Str(s) => Fill::Str(s.clone()),
            _ => return None,
        },
        _ => {
            let v = first_f()?;
            #[allow(clippy::cast_possible_truncation)]
            Fill::Int(if v.fract() == 0.0 {
                v as i128
            } else {
                return None;
            })
        }
    })
}

// ----- reading -----

fn dtype_of(t: ZType) -> DType {
    match t {
        ZType::Bool => DType::Bool,
        ZType::I8 => DType::I8,
        ZType::I16 => DType::I16,
        ZType::I32 => DType::I32,
        ZType::I64 => DType::I64,
        ZType::U8 => DType::U8,
        ZType::U16 => DType::U16,
        ZType::U32 => DType::U32,
        ZType::U64 => DType::U64,
        ZType::F32 => DType::F32,
        ZType::F64 => DType::F64,
        _ => DType::Str,
    }
}

fn typed(d: ArrayData) -> crate::Result<TypedArray<Owned>> {
    Ok(match d {
        ArrayData::Bool(a) => TypedArray::Bool(a),
        ArrayData::I8(a) => TypedArray::I8(a),
        ArrayData::I16(a) => TypedArray::I16(a),
        ArrayData::I32(a) => TypedArray::I32(a),
        ArrayData::I64(a) => TypedArray::I64(a),
        ArrayData::U8(a) => TypedArray::U8(a),
        ArrayData::U16(a) => TypedArray::U16(a),
        ArrayData::U32(a) => TypedArray::U32(a),
        ArrayData::U64(a) => TypedArray::U64(a),
        ArrayData::F32(a) => TypedArray::F32(a),
        ArrayData::F64(a) => TypedArray::F64(a),
        ArrayData::Str(a) => TypedArray::Str(a),
        other => {
            return Err(CoreError::Read(format!(
                "zarr values of type {:?}",
                other.dtype()
            )));
        }
    })
}

/// One array of a store, read region by region.
#[derive(Debug)]
struct ZarrSource {
    array: Array,
    dtype: DType,
}

impl LazySource for ZarrSource {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        let size = *self.array.chunks().get(axis)?;
        (size > 0).then_some((size, 0))
    }

    fn read_slice(&self, start: &[usize], count: &[usize]) -> crate::Result<TypedArray<Owned>> {
        let data = self
            .array
            .read_region(start, count)
            .map_err(|e| CoreError::Read(format!("{}: {e}", self.array.path())))?;
        typed(data)
    }
}

fn read_group(g: &Group, lazy: bool) -> Result<RawDataset<'static>> {
    let mut ds = RawDataset {
        attrs: attrs_from(g.attrs()),
        ..RawDataset::default()
    };
    for (name, node) in g.members().map_err(err)? {
        match node {
            Node::Group(sub) => {
                ds.groups
                    .insert(Name::from(name.as_str()), read_group(&sub, lazy)?);
            }
            Node::Array(a) => {
                let Some(dims) = a.dimension_names().map(<[String]>::to_vec) else {
                    ds.skipped.push((
                        Name::from(name.as_str()),
                        "no dimension names (_ARRAY_DIMENSIONS / dimension_names)".into(),
                    ));
                    continue;
                };
                if dims.len() != a.shape().len() || dims.iter().any(String::is_empty) {
                    ds.skipped.push((
                        Name::from(name.as_str()),
                        "dimension names do not match the shape".into(),
                    ));
                    continue;
                }
                for (d, n) in dims.iter().zip(a.shape()) {
                    match ds.dims.get(d.as_str()) {
                        Some(existing) if existing.len != *n => {
                            return Err(bad(format!(
                                "dimension {d:?} has lengths {} and {n}",
                                existing.len
                            )));
                        }
                        Some(_) => {}
                        None => {
                            ds.dims.insert(
                                Name::from(d.as_str()),
                                RawDim {
                                    len: *n,
                                    unlimited: false,
                                },
                            );
                        }
                    }
                }
                let dtype = dtype_of(a.dtype());
                let mut attrs = attrs_from(a.attrs());
                match a.format() {
                    ZarrFormat::V2 => {
                        if let Some(f) = a.fill_value()
                            && dtype != DType::Str
                            && let Some(v) = fill_attr(f, dtype)
                        {
                            attrs.insert(Name::from("_FillValue"), v);
                        }
                    }
                    ZarrFormat::V3 => {
                        // xarray stores float fill values as base64 of a little-endian f64.
                        if let Some(AttrValue::Str(s)) = attrs.get("_FillValue")
                            && matches!(dtype, DType::F32 | DType::F64)
                        {
                            let v = base64_decode(s)
                                .and_then(|b| b.try_into().ok())
                                .map(f64::from_le_bytes);
                            if let Some(v) = v {
                                #[allow(clippy::cast_possible_truncation)]
                                let attr = if dtype == DType::F32 {
                                    AttrValue::F32(vec![v as f32])
                                } else {
                                    AttrValue::F64(vec![v])
                                };
                                attrs.insert(Name::from("_FillValue"), attr);
                            }
                        }
                    }
                }
                let storage = Storage {
                    chunks: Some(a.chunks().to_vec()),
                    ..Storage::default()
                };
                let source = ZarrSource { array: a, dtype };
                let data = if lazy {
                    RawData::Lazy(Arc::new(source))
                } else {
                    let start = vec![0; source.shape().len()];
                    RawData::Loaded(source.read_slice(&start, source.shape())?)
                };
                ds.vars.insert(
                    Name::from(name.as_str()),
                    RawVariable {
                        dims: dims.iter().map(|d| Name::from(d.as_str())).collect(),
                        data,
                        is_char: false,
                        attrs,
                        storage,
                    },
                );
            }
        }
    }
    Ok(ds)
}

/// Read a store into the raw data model (`lazy`: data read on demand).
///
/// # Errors
/// [`IoError`] for unreadable stores.
pub fn read(path: &Path, lazy: bool) -> Result<RawDataset<'static>> {
    let store = Store::new(path);
    let root = Group::open(&store, "").map_err(err)?;
    read_group(&root, lazy)
}

// ----- writing -----

/// Options for writing a Zarr store.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ZarrOptions {
    /// v2 or v3 (the default, as zarr-python 3).
    pub format: ZarrFormat,
    /// The compressor (default: Zstandard when compiled in, else gzip).
    pub compressor: Option<Compressor>,
    /// Skip consolidated metadata.
    pub no_consolidated: bool,
}

impl ZarrOptions {
    /// Options for a format.
    #[must_use]
    pub fn format(format: ZarrFormat) -> Self {
        Self {
            format,
            ..Self::default()
        }
    }

    /// Compress with `c`.
    #[must_use]
    pub fn compressor(mut self, c: Compressor) -> Self {
        self.compressor = Some(c);
        self
    }

    /// Whether to write consolidated metadata (default yes).
    #[must_use]
    pub fn consolidated(mut self, yes: bool) -> Self {
        self.no_consolidated = !yes;
        self
    }
}

fn ztype(d: DType) -> Result<ZType> {
    Ok(match d {
        DType::Bool => ZType::Bool,
        DType::I8 => ZType::I8,
        DType::I16 => ZType::I16,
        DType::I32 => ZType::I32,
        DType::I64 => ZType::I64,
        DType::U8 => ZType::U8,
        DType::U16 => ZType::U16,
        DType::U32 => ZType::U32,
        DType::U64 => ZType::U64,
        DType::F32 => ZType::F32,
        DType::F64 => ZType::F64,
        DType::Str => ZType::String,
        other => {
            return Err(IoError::Unsupported(format!(
                "{other} values in zarr (encode times first)"
            )));
        }
    })
}

fn zdata(t: TypedArray<Owned>) -> Result<ArrayData> {
    Ok(match t {
        TypedArray::Bool(a) => ArrayData::Bool(a),
        TypedArray::I8(a) => ArrayData::I8(a),
        TypedArray::I16(a) => ArrayData::I16(a),
        TypedArray::I32(a) => ArrayData::I32(a),
        TypedArray::I64(a) => ArrayData::I64(a),
        TypedArray::U8(a) => ArrayData::U8(a),
        TypedArray::U16(a) => ArrayData::U16(a),
        TypedArray::U32(a) => ArrayData::U32(a),
        TypedArray::U64(a) => ArrayData::U64(a),
        TypedArray::F32(a) => ArrayData::F32(a),
        TypedArray::F64(a) => ArrayData::F64(a),
        TypedArray::Str(a) => ArrayData::Str(a),
        other => {
            return Err(IoError::Unsupported(format!(
                "{} values in zarr",
                other.dtype()
            )));
        }
    })
}

/// Chunks of about 1 MiB: halve the longest axis until the chunk fits.
fn default_chunks(shape: &[usize], size: usize) -> Vec<usize> {
    const TARGET: usize = 1 << 20;
    let mut c: Vec<usize> = shape.iter().map(|n| (*n).max(1)).collect();
    loop {
        let bytes = c.iter().try_fold(size.max(1), |a, n| a.checked_mul(*n));
        if bytes.is_some_and(|b| b <= TARGET) {
            return c;
        }
        let Some((i, _)) = c
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 1)
            .max_by_key(|(_, n)| **n)
        else {
            return c;
        };
        if let Some(n) = c.get_mut(i) {
            *n = n.div_ceil(2);
        }
    }
}

fn element_size(d: ZType) -> usize {
    d.size().unwrap_or(16)
}

fn write_var(store: &Store, path: &str, v: &RawVariable, opts: &ZarrOptions) -> Result<()> {
    let ztype = ztype(v.data.dtype())?;
    let shape = v.data.shape().to_vec();
    let chunks = v
        .storage
        .chunks
        .clone()
        .filter(|c| c.len() == shape.len())
        .unwrap_or_else(|| default_chunks(&shape, element_size(ztype)));
    let mut attrs = v.attrs.clone();
    let fill_attr_v = attrs.shift_remove("_FillValue");
    let mut b = ArrayBuilder::new(ztype, &shape, &chunks)
        .dimension_names(v.dims.iter().map(ToString::to_string).collect())
        .compressor(opts.compressor.unwrap_or_default());
    match opts.format {
        ZarrFormat::V2 => {
            if let Some(f) = fill_attr_v.as_ref().and_then(|a| fill_of(a, ztype)) {
                b = b.fill_value(f);
            }
        }
        ZarrFormat::V3 => {
            // The array's own fill: NaN for floats, as xarray (missing chunks read as NaN).
            if matches!(ztype, ZType::F32 | ZType::F64) {
                b = b.fill_value(Fill::Float(f64::NAN));
            }
            if let Some(a) = &fill_attr_v {
                let encoded = match (ztype, fill_of(a, ztype)) {
                    (ZType::F32 | ZType::F64, Some(Fill::Float(x))) => {
                        AttrValue::Str(base64_encode(&x.to_le_bytes()))
                    }
                    _ => a.clone(),
                };
                attrs.insert(Name::from("_FillValue"), encoded);
            }
        }
    }
    let array = b
        .attrs(attrs_to(&attrs))
        .create(store, path, opts.format)
        .map_err(err)?;
    // Chunk by chunk: lazy data is read one chunk's region at a time.
    let grid = array.grid();
    if grid.contains(&0) {
        return Ok(());
    }
    let total = grid.iter().product::<usize>();
    let one = |k: usize| -> Result<()> {
        let mut idx = vec![0; grid.len()];
        let mut rest = k;
        for (slot, g) in idx.iter_mut().zip(&grid).rev() {
            *slot = rest.checked_rem(*g).unwrap_or(0);
            rest = rest.checked_div(*g).unwrap_or(0);
        }
        let (start, count) = array.chunk_region(&idx);
        let part = match &v.data {
            RawData::Lazy(s) => s.read_slice(&start, &count)?,
            RawData::Loaded(a) => super::raw::slice_of(&a.view(), &start, &count),
            RawData::Borrowed(a) => super::raw::slice_of(a, &start, &count),
        };
        array.write_chunk(&idx, &zdata(part)?).map_err(err)
    };
    // Chunks are encoded in parallel, a few per thread at a time (lazy data is read one
    // chunk's region at a time, so memory stays bounded).
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let batch = rayon::current_num_threads().max(1).saturating_mul(4);
        let mut k = 0;
        while k < total {
            let end = k.saturating_add(batch).min(total);
            (k..end).into_par_iter().try_for_each(one)?;
            k = end;
        }
    }
    #[cfg(not(feature = "rayon"))]
    for k in 0..total {
        one(k)?;
    }
    Ok(())
}

fn write_group(store: &Store, path: &str, ds: &RawDataset, opts: &ZarrOptions) -> Result<Group> {
    let g = Group::create(store, path, opts.format, attrs_to(&ds.attrs)).map_err(err)?;
    for (name, v) in &ds.vars {
        write_var(store, &annoray_join(path, name), v, opts)?;
    }
    for (name, sub) in &ds.groups {
        write_group(store, &annoray_join(path, name), sub, opts)?;
    }
    Ok(g)
}

fn annoray_join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}/{name}")
    }
}

/// Write the raw data model as a store at `path` (replacing it).
///
/// # Errors
/// [`IoError`] for types Zarr cannot hold or I/O failures.
pub fn write(ds: &RawDataset, path: &Path, opts: &ZarrOptions) -> Result<()> {
    if path.exists() {
        if !is_zarr(path) && path.read_dir().is_ok_and(|mut d| d.next().is_some()) {
            return Err(bad(format!(
                "{} exists and is not a zarr store; not replacing it",
                path.display()
            )));
        }
        std::fs::remove_dir_all(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    }
    let store = Store::new(path);
    let root = write_group(&store, "", ds, opts)?;
    if !opts.no_consolidated {
        root.consolidate().map_err(err)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips() {
        for bytes in [&b""[..], b"a", b"ab", b"abc", &f64::NAN.to_le_bytes()] {
            assert_eq!(base64_decode(&base64_encode(bytes)).unwrap(), bytes);
        }
        assert_eq!(base64_encode(&f64::NAN.to_le_bytes()), "AAAAAAAA+H8=");
        assert!(base64_decode("@@@@").is_none());
    }

    #[test]
    fn chunks_are_about_a_mebibyte() {
        assert_eq!(default_chunks(&[10, 10], 8), vec![10, 10]);
        let c = default_chunks(&[4000, 4000], 8);
        assert!(c.iter().product::<usize>() * 8 <= 1 << 20);
        assert_eq!(default_chunks(&[0], 8), vec![1]);
    }
}
