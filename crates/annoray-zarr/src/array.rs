//! Arrays: metadata, region reads and chunk writes.

use ndarray::Slice;
use serde_json::{Map, Value, json};

use crate::codec::{self, Codec, Sharding};
use crate::data::{ArrayData, DType, Endian, Fill};
use crate::store::{Store, join};
use crate::{Error, ZarrFormat, bad};

/// How chunk indices become keys.
#[derive(Debug, Clone, PartialEq)]
struct KeyEncoding {
    /// v3's default encoding (`c/0/1`) rather than v2's (`0.1`).
    prefixed: bool,
    separator: char,
}

/// An array in a store.
#[derive(Debug, Clone)]
pub struct Array {
    store: Store,
    path: String,
    format: ZarrFormat,
    shape: Vec<usize>,
    chunks: Vec<usize>,
    dtype: DType,
    fill: Option<Fill>,
    codecs: Vec<Codec>,
    key: KeyEncoding,
    dims: Option<Vec<String>>,
    attrs: Map<String, Value>,
}

fn usizes(v: Option<&Value>, what: &str) -> Result<Vec<usize>, Error> {
    v.and_then(Value::as_array)
        .and_then(|a| {
            a.iter()
                .map(|x| x.as_u64().and_then(|n| usize::try_from(n).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| bad(format!("{what} is not a list of sizes")))
}

pub(crate) fn json_of(bytes: &[u8], key: &str) -> Result<Value, Error> {
    serde_json::from_slice(bytes).map_err(|e| bad(format!("{key}: {e}")))
}

impl Array {
    /// Open the array at `path` (v2 `.zarray` or v3 `zarr.json`).
    ///
    /// # Errors
    /// [`Error::NotFound`] if there is no array; [`Error::Invalid`] for malformed metadata;
    /// [`Error::Unsupported`] for data types, codecs or chunk grids this crate does not read.
    pub fn open(store: &Store, path: &str) -> Result<Self, Error> {
        if let Some(b) = store.get(&join(path, "zarr.json"))? {
            let meta = json_of(&b, "zarr.json")?;
            if meta.get("node_type").and_then(Value::as_str) != Some("array") {
                return Err(Error::NotFound(format!("array {path:?}")));
            }
            return Self::from_v3(store, path, &meta);
        }
        let Some(b) = store.get(&join(path, ".zarray"))? else {
            return Err(Error::NotFound(format!("array {path:?}")));
        };
        let meta = json_of(&b, ".zarray")?;
        let attrs = match store.get(&join(path, ".zattrs"))? {
            Some(a) => match json_of(&a, ".zattrs")? {
                Value::Object(m) => m,
                _ => Map::new(),
            },
            None => Map::new(),
        };
        Self::from_v2(store, path, &meta, attrs)
    }

    fn from_v2(
        store: &Store,
        path: &str,
        meta: &Value,
        mut attrs: Map<String, Value>,
    ) -> Result<Self, Error> {
        let shape = usizes(meta.get("shape"), "shape")?;
        let chunks = usizes(meta.get("chunks"), "chunks")?;
        let dt = meta
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("dtype"))?;
        let (mut dtype, endian) = DType::from_v2(dt)?;
        let order_f = meta.get("order").and_then(Value::as_str) == Some("F");
        let filters = meta.get("filters").cloned().unwrap_or(Value::Null);
        let compressor = meta.get("compressor").cloned().unwrap_or(Value::Null);
        // `|O` holds text only with the vlen-utf8 filter.
        if dtype == DType::String
            && !filters.as_array().is_some_and(|f| {
                f.iter()
                    .any(|x| x.get("id").and_then(Value::as_str) == Some("vlen-utf8"))
            })
        {
            return Err(Error::Unsupported(
                "zarr v2 object arrays other than vlen-utf8 text".into(),
            ));
        }
        let codecs = codec::from_v2(dtype, endian, order_f, shape.len(), &filters, &compressor)?;
        if dtype == DType::FixedBytes(0) {
            dtype = DType::FixedBytes(1);
        }
        let fill = Fill::parse(dtype, meta.get("fill_value").unwrap_or(&Value::Null))?;
        let separator = match meta.get("dimension_separator").and_then(Value::as_str) {
            Some("/") => '/',
            _ => '.',
        };
        let dims = attrs
            .remove("_ARRAY_DIMENSIONS")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v).ok());
        let a = Self {
            store: store.clone(),
            path: path.to_owned(),
            format: ZarrFormat::V2,
            shape,
            chunks,
            dtype,
            fill,
            codecs,
            key: KeyEncoding {
                prefixed: false,
                separator,
            },
            dims,
            attrs,
        };
        a.check()?;
        Ok(a)
    }

    fn from_v3(store: &Store, path: &str, meta: &Value) -> Result<Self, Error> {
        let shape = usizes(meta.get("shape"), "shape")?;
        let grid = meta.get("chunk_grid").ok_or_else(|| bad("chunk_grid"))?;
        if grid.get("name").and_then(Value::as_str) != Some("regular") {
            return Err(Error::Unsupported(format!("chunk grid {grid}")));
        }
        let chunks = usizes(
            grid.get("configuration").and_then(|c| c.get("chunk_shape")),
            "chunk_shape",
        )?;
        let dtype = DType::from_v3(meta.get("data_type").ok_or_else(|| bad("data_type"))?)?;
        let codecs = codec::from_v3(
            meta.get("codecs").unwrap_or(&Value::Null),
            dtype,
            shape.len(),
        )?;
        let fill = Fill::parse(dtype, meta.get("fill_value").unwrap_or(&Value::Null))?;
        let enc = meta.get("chunk_key_encoding");
        let name = enc
            .and_then(|e| e.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("default");
        let sep = enc
            .and_then(|e| e.get("configuration"))
            .and_then(|c| c.get("separator"))
            .and_then(Value::as_str);
        let key = match name {
            "default" => KeyEncoding {
                prefixed: true,
                separator: if sep == Some(".") { '.' } else { '/' },
            },
            "v2" => KeyEncoding {
                prefixed: false,
                separator: if sep == Some("/") { '/' } else { '.' },
            },
            other => return Err(Error::Unsupported(format!("chunk key encoding {other:?}"))),
        };
        if meta
            .get("storage_transformers")
            .and_then(Value::as_array)
            .is_some_and(|t| !t.is_empty())
        {
            return Err(Error::Unsupported("storage transformers".into()));
        }
        let dims = meta
            .get("dimension_names")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().unwrap_or("").to_owned())
                    .collect()
            });
        let attrs = match meta.get("attributes") {
            Some(Value::Object(m)) => m.clone(),
            _ => Map::new(),
        };
        let a = Self {
            store: store.clone(),
            path: path.to_owned(),
            format: ZarrFormat::V3,
            shape,
            chunks,
            dtype,
            fill,
            codecs,
            key,
            dims,
            attrs,
        };
        a.check()?;
        Ok(a)
    }

    fn check(&self) -> Result<(), Error> {
        if self.chunks.len() != self.shape.len() || self.chunks.contains(&0) {
            return Err(bad(format!(
                "{}: chunk shape does not fit the array",
                self.path
            )));
        }
        if let Some(Codec::Sharding(s)) =
            self.codecs.iter().find(|c| matches!(c, Codec::Sharding(_)))
            && (s.chunk_shape.len() != self.chunks.len()
                || s.chunk_shape
                    .iter()
                    .zip(&self.chunks)
                    .any(|(i, c)| *i == 0 || c.checked_rem(*i) != Some(0)))
        {
            return Err(bad(format!(
                "{}: shard shape is not a multiple of the inner chunks",
                self.path
            )));
        }
        Ok(())
    }

    /// The array's path in the store.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The format.
    #[must_use]
    pub fn format(&self) -> ZarrFormat {
        self.format
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The chunk (or shard) shape.
    #[must_use]
    pub fn chunks(&self) -> &[usize] {
        &self.chunks
    }

    /// The element type as stored.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The fill value (`None`: unset in v2).
    #[must_use]
    pub fn fill_value(&self) -> Option<&Fill> {
        self.fill.as_ref()
    }

    /// Dimension names: v3 `dimension_names`, or the v2 `_ARRAY_DIMENSIONS` attribute (xarray's
    /// convention; it is not listed among [`attrs`](Self::attrs)).
    #[must_use]
    pub fn dimension_names(&self) -> Option<&[String]> {
        self.dims.as_deref()
    }

    /// User attributes.
    #[must_use]
    pub fn attrs(&self) -> &Map<String, Value> {
        &self.attrs
    }

    /// The codec pipeline.
    #[must_use]
    pub fn codecs(&self) -> &[Codec] {
        &self.codecs
    }

    fn fill_or_zero(&self) -> Fill {
        self.fill.clone().unwrap_or_else(|| Fill::zero(self.dtype))
    }

    /// The key of the chunk at grid position `idx`.
    #[must_use]
    pub fn chunk_key(&self, idx: &[usize]) -> String {
        let sep = self.key.separator.to_string();
        let parts: Vec<String> = idx.iter().map(ToString::to_string).collect();
        let tail = if self.key.prefixed {
            if parts.is_empty() {
                "c".to_owned()
            } else {
                format!("c{sep}{}", parts.join(&sep))
            }
        } else if parts.is_empty() {
            "0".to_owned()
        } else {
            parts.join(&sep)
        };
        join(&self.path, &tail)
    }

    /// The chunk grid's size along each axis.
    #[must_use]
    pub fn grid(&self) -> Vec<usize> {
        self.shape
            .iter()
            .zip(&self.chunks)
            .map(|(s, c)| s.div_ceil(*c))
            .collect()
    }

    /// The decoded chunk at `idx` (full chunk shape), or `None` if it is not stored.
    ///
    /// # Errors
    /// [`Error::Io`], codec errors.
    pub fn read_chunk(&self, idx: &[usize]) -> Result<Option<ArrayData>, Error> {
        let Some(bytes) = self.store.get(&self.chunk_key(idx))? else {
            return Ok(None);
        };
        let fill = self.fill_or_zero();
        codec::decode(&self.codecs, bytes, self.dtype, &self.chunks, &fill)
            .map(Some)
            .map_err(|e| match e {
                Error::Invalid(m) => bad(format!("{}: chunk {idx:?}: {m}", self.path)),
                other => other,
            })
    }

    /// Every value.
    ///
    /// # Errors
    /// As for [`read_region`](Self::read_region).
    pub fn read(&self) -> Result<ArrayData, Error> {
        self.read_region(&vec![0; self.shape.len()], &self.shape.clone())
    }

    /// The values in `start .. start + count`: only intersecting chunks are read; missing
    /// chunks give the fill value.
    ///
    /// # Errors
    /// [`Error::Invalid`] for a region outside the array; [`Error::Io`]; codec errors.
    pub fn read_region(&self, start: &[usize], count: &[usize]) -> Result<ArrayData, Error> {
        let ndim = self.shape.len();
        if start.len() != ndim || count.len() != ndim {
            return Err(bad("region rank differs from the array's"));
        }
        for ((s, c), n) in start.iter().zip(count).zip(&self.shape) {
            if s.checked_add(*c).is_none_or(|e| e > *n) {
                return Err(bad(format!(
                    "region {start:?}+{count:?} outside shape {:?}",
                    self.shape
                )));
            }
        }
        let fill = self.fill_or_zero();
        let mut out = ArrayData::filled(self.dtype, count, &fill)?;
        if count.contains(&0) {
            return Ok(out);
        }
        let lo: Vec<usize> = start
            .iter()
            .zip(&self.chunks)
            .map(|(s, c)| s.checked_div(*c).unwrap_or(0))
            .collect();
        let hi: Vec<usize> = start
            .iter()
            .zip(count)
            .zip(&self.chunks)
            .map(|((s, n), c)| {
                s.saturating_add(*n)
                    .saturating_sub(1)
                    .checked_div(*c)
                    .unwrap_or(0)
            })
            .collect();
        // The chunks the region touches, in C order.
        let mut todo = Vec::new();
        let mut idx = lo.clone();
        loop {
            todo.push(idx.clone());
            if !next(&mut idx, &lo, &hi) {
                break;
            }
        }
        let place = |out: &mut ArrayData, idx: &[usize], chunk: &ArrayData| -> Result<(), Error> {
            // Intersection of this chunk with the region, in both coordinate systems.
            let mut src = Vec::with_capacity(ndim);
            let mut dst = Vec::with_capacity(ndim);
            let mut len = Vec::with_capacity(ndim);
            for d in 0..ndim {
                let (Some(i), Some(c), Some(s), Some(n)) =
                    (idx.get(d), self.chunks.get(d), start.get(d), count.get(d))
                else {
                    return Err(bad("region"));
                };
                let c0 = i.saturating_mul(*c);
                let a = (*s).max(c0);
                let b = s.saturating_add(*n).min(c0.saturating_add(*c));
                src.push(a.saturating_sub(c0));
                dst.push(a.saturating_sub(*s));
                len.push(b.saturating_sub(a));
            }
            copy(out, &dst, chunk, &src, &len)
        };
        #[cfg(feature = "rayon")]
        if todo.len() > 1 {
            use rayon::prelude::*;
            // Decoded chunks in flight: up to four per thread, so large reads stay within
            // memory.
            let batch = rayon::current_num_threads().max(1).saturating_mul(4);
            for group in todo.chunks(batch) {
                let decoded = group
                    .par_iter()
                    .map(|i| self.read_chunk(i))
                    .collect::<Result<Vec<_>, Error>>()?;
                for (i, chunk) in group.iter().zip(decoded) {
                    if let Some(chunk) = chunk {
                        place(&mut out, i, &chunk)?;
                    }
                }
            }
            return Ok(out);
        }
        for i in &todo {
            if let Some(chunk) = self.read_chunk(i)? {
                place(&mut out, i, &chunk)?;
            }
        }
        Ok(out)
    }

    /// Write the chunk at grid position `idx` from `data`, the values of that chunk's part of
    /// the array (smaller than the chunk at the array's far edges; padded with the fill
    /// value). A chunk equal to the fill value everywhere is removed instead (zarr-python's
    /// default).
    ///
    /// # Errors
    /// [`Error::Invalid`] for a wrong shape or type; [`Error::Io`]; codec errors.
    pub fn write_chunk(&self, idx: &[usize], data: &ArrayData) -> Result<(), Error> {
        let ndim = self.shape.len();
        if idx.len() != ndim {
            return Err(bad("chunk index rank"));
        }
        let want: Vec<usize> = (0..ndim)
            .map(|d| {
                let (i, c, n) = (
                    idx.get(d).copied().unwrap_or(0),
                    self.chunks.get(d).copied().unwrap_or(1),
                    self.shape.get(d).copied().unwrap_or(0),
                );
                c.min(n.saturating_sub(i.saturating_mul(c)))
            })
            .collect();
        if data.shape() != want.as_slice() {
            return Err(bad(format!(
                "chunk {idx:?} needs shape {want:?}, got {:?}",
                data.shape()
            )));
        }
        let fill = self.fill_or_zero();
        let full = if want == self.chunks {
            convert(data, self.dtype)?
        } else {
            let mut f = ArrayData::filled(self.dtype, &self.chunks, &fill)?;
            assign(&mut f, &vec![0; ndim], &convert(data, self.dtype)?)?;
            f
        };
        let key = self.chunk_key(idx);
        if all_fill(&full, &fill) {
            return self.store.delete(&key);
        }
        let bytes = codec::encode(&self.codecs, full, self.dtype, &fill)?;
        self.store.set(&key, &bytes)
    }

    /// Write every chunk from the whole array's values.
    ///
    /// # Errors
    /// As for [`write_chunk`](Self::write_chunk).
    pub fn write(&self, data: &ArrayData) -> Result<(), Error> {
        if data.shape() != self.shape.as_slice() {
            return Err(bad(format!(
                "data shape {:?} differs from {:?}",
                data.shape(),
                self.shape
            )));
        }
        let grid = self.grid();
        if grid.contains(&0) {
            return Ok(());
        }
        let lo = vec![0; grid.len()];
        let hi: Vec<usize> = grid.iter().map(|g| g.saturating_sub(1)).collect();
        let mut idx = lo.clone();
        loop {
            let (start, count) = self.chunk_region(&idx);
            self.write_chunk(&idx, &extract(data, &start, &count)?)?;
            if !next(&mut idx, &lo, &hi) {
                break;
            }
        }
        Ok(())
    }

    /// The part of the array the chunk at `idx` covers: `(start, count)`.
    #[must_use]
    pub fn chunk_region(&self, idx: &[usize]) -> (Vec<usize>, Vec<usize>) {
        idx.iter()
            .zip(&self.chunks)
            .zip(&self.shape)
            .map(|((i, c), n)| {
                let s = i.saturating_mul(*c);
                (s, (*c).min(n.saturating_sub(s)))
            })
            .unzip()
    }

    /// Replace the user attributes (and rewrite the metadata).
    ///
    /// # Errors
    /// [`Error::Io`].
    pub fn set_attrs(&mut self, attrs: Map<String, Value>) -> Result<(), Error> {
        self.attrs = attrs;
        self.write_meta()
    }

    fn write_meta(&self) -> Result<(), Error> {
        match self.format {
            ZarrFormat::V2 => {
                let (filters, compressor) = codec::to_v2(
                    &self
                        .codecs
                        .iter()
                        .filter(|c| !matches!(c, Codec::Transpose(_)))
                        .cloned()
                        .collect::<Vec<_>>(),
                )?;
                let meta = json!({
                    "zarr_format": 2,
                    "shape": self.shape,
                    "chunks": self.chunks,
                    "dtype": self.dtype.to_v2(),
                    "fill_value": self.fill.as_ref().map_or(Value::Null, Fill::to_json),
                    "order": "C",
                    "filters": filters,
                    "compressor": compressor,
                    "dimension_separator": self.key.separator.to_string(),
                });
                let mut attrs = self.attrs.clone();
                if let Some(d) = &self.dims {
                    attrs.insert("_ARRAY_DIMENSIONS".into(), json!(d));
                }
                self.store
                    .set(&join(&self.path, ".zarray"), &pretty(&meta)?)?;
                self.store.set(
                    &join(&self.path, ".zattrs"),
                    &pretty(&Value::Object(attrs))?,
                )
            }
            ZarrFormat::V3 => {
                let mut meta = json!({
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": self.shape,
                    "data_type": self.dtype.to_v3(),
                    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": self.chunks}},
                    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                    "fill_value": self.fill_or_zero().to_json(),
                    "codecs": codec::to_v3(&self.codecs)?,
                    "attributes": Value::Object(self.attrs.clone()),
                    "storage_transformers": [],
                });
                if let (Some(d), Value::Object(m)) = (&self.dims, &mut meta) {
                    m.insert("dimension_names".into(), json!(d));
                }
                self.store
                    .set(&join(&self.path, "zarr.json"), &pretty(&meta)?)
            }
        }
    }
}

pub(crate) fn pretty(v: &Value) -> Result<Vec<u8>, Error> {
    serde_json::to_vec_pretty(v).map_err(|e| bad(e.to_string()))
}

/// Advance a C-order counter over `lo..=hi`; false once it wraps.
fn next(idx: &mut [usize], lo: &[usize], hi: &[usize]) -> bool {
    for d in (0..idx.len()).rev() {
        let (Some(i), Some(l), Some(h)) = (idx.get_mut(d), lo.get(d), hi.get(d)) else {
            return false;
        };
        if *i < *h {
            *i = i.saturating_add(1);
            return true;
        }
        *i = *l;
    }
    false
}

/// `dst[dst_start .. + len] = src[src_start .. + len]`.
pub(crate) fn copy(
    dst: &mut ArrayData,
    dst_start: &[usize],
    src: &ArrayData,
    src_start: &[usize],
    len: &[usize],
) -> Result<(), Error> {
    macro_rules! go {
        ($($v:ident),*) => {
            match (dst, src) {
                $((ArrayData::$v(d), ArrayData::$v(s)) => {
                    let sv = s.slice_each_axis(|ax| {
                        let i = ax.axis.index();
                        let a = src_start.get(i).copied().unwrap_or(0);
                        Slice::from(a..a.saturating_add(len.get(i).copied().unwrap_or(0)))
                    });
                    let mut dv = d.slice_each_axis_mut(|ax| {
                        let i = ax.axis.index();
                        let a = dst_start.get(i).copied().unwrap_or(0);
                        Slice::from(a..a.saturating_add(len.get(i).copied().unwrap_or(0)))
                    });
                    dv.assign(&sv);
                    Ok(())
                })*
                _ => Err(bad("element types differ")),
            }
        };
    }
    for (((ds, ss), l), (dn, sn)) in dst_start
        .iter()
        .zip(src_start)
        .zip(len)
        .zip(dst.shape().iter().zip(src.shape()))
    {
        if ds.saturating_add(*l) > *dn || ss.saturating_add(*l) > *sn {
            return Err(bad("block out of range"));
        }
    }
    go!(Bool, I8, I16, I32, I64, U8, U16, U32, U64, F32, F64, Str)
}

/// `dst[start .. + src.shape] = src`.
pub(crate) fn assign(dst: &mut ArrayData, start: &[usize], src: &ArrayData) -> Result<(), Error> {
    let len = src.shape().to_vec();
    copy(dst, start, src, &vec![0; len.len()], &len)
}

/// `data[start .. + shape]`, owned.
pub(crate) fn extract(
    data: &ArrayData,
    start: &[usize],
    shape: &[usize],
) -> Result<ArrayData, Error> {
    for ((s, n), d) in start.iter().zip(shape).zip(data.shape()) {
        if s.saturating_add(*n) > *d {
            return Err(bad("block out of range"));
        }
    }
    Ok(crate::data::map_each!(data, a => a
        .slice_each_axis(|ax| {
            let i = ax.axis.index();
            let s = start.get(i).copied().unwrap_or(0);
            Slice::from(s..s.saturating_add(shape.get(i).copied().unwrap_or(0)))
        })
        .to_owned()))
}

/// Whether every value equals `fill` (NaN matching NaN).
pub(crate) fn all_fill(data: &ArrayData, fill: &Fill) -> bool {
    match (data, fill) {
        (ArrayData::Bool(a), Fill::Bool(f)) => a.iter().all(|v| v == f),
        (ArrayData::F32(a), Fill::Float(f)) => a
            .iter()
            .all(|v| (v.is_nan() && f.is_nan()) || f64::from(*v) == *f),
        (ArrayData::F64(a), Fill::Float(f)) => {
            a.iter().all(|v| (v.is_nan() && f.is_nan()) || v == f)
        }
        (ArrayData::Str(a), Fill::Str(f)) => a.iter().all(|v| v == f),
        (ArrayData::I8(a), Fill::Int(f)) => a.iter().all(|v| i128::from(*v) == *f),
        (ArrayData::I16(a), Fill::Int(f)) => a.iter().all(|v| i128::from(*v) == *f),
        (ArrayData::I32(a), Fill::Int(f)) => a.iter().all(|v| i128::from(*v) == *f),
        (ArrayData::I64(a), Fill::Int(f)) => a.iter().all(|v| i128::from(*v) == *f),
        (ArrayData::U8(a), Fill::Int(f)) => a.iter().all(|v| i128::from(*v) == *f),
        (ArrayData::U16(a), Fill::Int(f)) => a.iter().all(|v| i128::from(*v) == *f),
        (ArrayData::U32(a), Fill::Int(f)) => a.iter().all(|v| i128::from(*v) == *f),
        (ArrayData::U64(a), Fill::Int(f)) => a.iter().all(|v| i128::from(*v) == *f),
        _ => false,
    }
}

/// `data` as stored (text for the text types; others must match exactly).
fn convert(data: &ArrayData, dtype: DType) -> Result<ArrayData, Error> {
    let text = matches!(
        dtype,
        DType::FixedBytes(_) | DType::FixedUnicode(_) | DType::String
    );
    if data.dtype() == dtype || (text && matches!(data, ArrayData::Str(_))) {
        return Ok(data.clone());
    }
    Err(bad(format!(
        "values of type {:?} for an array of {dtype:?}",
        data.dtype()
    )))
}

/// The compression of a new array.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Compressor {
    /// Stored as is.
    None,
    /// gzip at a level, 0–9.
    Gzip(u8),
    /// zlib at a level (v2 `zlib`; v3 `numcodecs.zlib`).
    Zlib(u8),
    /// Zstandard (the level is recorded; written at the fastest level).
    Zstd(i32),
    /// numcodecs' LZ4 (v2 `lz4`; v3 `numcodecs.lz4`).
    Lz4,
    /// Blosc.
    Blosc {
        /// Inner compressor.
        cname: annoray_codecs::BloscCodec,
        /// Level, 0–9.
        clevel: u8,
        /// Shuffle.
        shuffle: annoray_codecs::BloscShuffle,
    },
}

impl Default for Compressor {
    /// Zstandard (zarr-python 3's default) when compiled in, otherwise gzip.
    fn default() -> Self {
        if annoray_codecs::zstd::available() {
            Self::Zstd(0)
        } else {
            Self::Gzip(5)
        }
    }
}

/// Settings of a new array; [`create`](Self::create) writes its metadata.
#[derive(Debug, Clone)]
pub struct ArrayBuilder {
    dtype: DType,
    shape: Vec<usize>,
    chunks: Vec<usize>,
    fill: Option<Fill>,
    compressor: Compressor,
    shard_inner: Option<Vec<usize>>,
    dims: Option<Vec<String>>,
    attrs: Map<String, Value>,
}

impl ArrayBuilder {
    /// An array of `dtype` and `shape`, chunked as `chunks` (clipped to the shape; zero-length
    /// axes get chunks of 1).
    #[must_use]
    pub fn new(dtype: DType, shape: &[usize], chunks: &[usize]) -> Self {
        let chunks = shape
            .iter()
            .zip(chunks)
            .map(|(n, c)| (*c).min(*n).max(1))
            .collect();
        Self {
            dtype,
            shape: shape.to_vec(),
            chunks,
            fill: None,
            compressor: Compressor::default(),
            shard_inner: None,
            dims: None,
            attrs: Map::new(),
        }
    }

    /// The fill value.
    #[must_use]
    pub fn fill_value(mut self, fill: Fill) -> Self {
        self.fill = Some(fill);
        self
    }

    /// The compression.
    #[must_use]
    pub fn compressor(mut self, c: Compressor) -> Self {
        self.compressor = c;
        self
    }

    /// v3 only: store each chunk as a shard of inner chunks of this shape (each chunk side a
    /// multiple of the inner one).
    #[must_use]
    pub fn shards(mut self, inner: &[usize]) -> Self {
        self.shard_inner = Some(inner.to_vec());
        self
    }

    /// Dimension names.
    #[must_use]
    pub fn dimension_names(mut self, names: Vec<String>) -> Self {
        self.dims = Some(names);
        self
    }

    /// User attributes.
    #[must_use]
    pub fn attrs(mut self, attrs: Map<String, Value>) -> Self {
        self.attrs = attrs;
        self
    }

    /// Write the metadata of the array at `path` (replacing what is there) and return it.
    ///
    /// # Errors
    /// [`Error::Invalid`] for inconsistent settings; [`Error::Unsupported`] for sharding in v2
    /// or codecs this crate does not write; [`Error::Io`].
    pub fn create(self, store: &Store, path: &str, format: ZarrFormat) -> Result<Array, Error> {
        if self.chunks.len() != self.shape.len() {
            return Err(bad("chunk rank differs from the shape's"));
        }
        if self
            .dims
            .as_ref()
            .is_some_and(|d| d.len() != self.shape.len())
        {
            return Err(bad("dimension names do not match the shape"));
        }
        let typesize = self.dtype.size().unwrap_or(1);
        let mut codecs = vec![if self.dtype == DType::String {
            Codec::VlenUtf8
        } else {
            Codec::Bytes(Endian::Little)
        }];
        match self.compressor {
            Compressor::None => {}
            Compressor::Gzip(l) => codecs.push(Codec::Gzip(l)),
            Compressor::Zlib(l) => codecs.push(Codec::Zlib(l)),
            Compressor::Zstd(l) => codecs.push(Codec::Zstd(l)),
            Compressor::Lz4 => codecs.push(Codec::Lz4),
            Compressor::Blosc {
                cname,
                clevel,
                shuffle,
            } => {
                codecs.push(Codec::Blosc {
                    cname,
                    clevel,
                    shuffle,
                    typesize,
                });
            }
        }
        if let Some(inner) = &self.shard_inner {
            if format == ZarrFormat::V2 {
                return Err(Error::Unsupported("sharding in zarr v2".into()));
            }
            codecs = vec![Codec::Sharding(Box::new(Sharding {
                chunk_shape: inner.clone(),
                codecs,
                index_codecs: vec![Codec::Bytes(Endian::Little), Codec::Crc32c],
                index_at_end: true,
            }))];
        }
        store.delete_dir(path).ok();
        let a = Array {
            store: store.clone(),
            path: path.to_owned(),
            format,
            shape: self.shape,
            chunks: self.chunks,
            dtype: self.dtype,
            fill: self.fill,
            codecs,
            key: match format {
                ZarrFormat::V2 => KeyEncoding {
                    prefixed: false,
                    separator: '.',
                },
                ZarrFormat::V3 => KeyEncoding {
                    prefixed: true,
                    separator: '/',
                },
            },
            dims: self.dims,
            attrs: self.attrs,
        };
        a.check()?;
        a.write_meta()?;
        Ok(a)
    }
}
