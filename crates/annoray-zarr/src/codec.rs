//! The codec pipeline: array→array (`transpose`), one array→bytes (`bytes`, `vlen-utf8`,
//! `sharding_indexed`), then bytes→bytes (`blosc`, `gzip`, `zlib`, `zstd`, `lz4`, `bz2`,
//! `shuffle`, `crc32c`). Zarr v2's `order`, `filters` and `compressor` are expressed in the same
//! pipeline.

use annoray_codecs::{BloscCodec, BloscShuffle};
use ndarray::{ArrayD, IxDyn};
use serde_json::{Value, json};

use crate::data::{ArrayData, DType, Endian, Fill, map_each};
use crate::{Error, bad};

/// Upper bound on a decoded variable-length chunk (fixed-size chunks are bounded exactly).
const VLEN_LIMIT: usize = 1 << 32;

/// One codec of a pipeline.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Codec {
    /// Axis permutation: encoded axis `i` is decoded axis `order[i]`.
    Transpose(Vec<usize>),
    /// Fixed-size elements as bytes in this order.
    Bytes(Endian),
    /// Variable-length UTF-8 (numcodecs `VLenUTF8`): a `u32` count, then `u32` length + text
    /// per element, little-endian.
    VlenUtf8,
    /// Chunks of chunks in one object, with an index.
    Sharding(Box<Sharding>),
    /// Blosc.
    Blosc {
        /// Inner compressor.
        cname: BloscCodec,
        /// Level, 0–9.
        clevel: u8,
        /// Shuffle.
        shuffle: BloscShuffle,
        /// Element size for the shuffle.
        typesize: usize,
    },
    /// gzip at a level.
    Gzip(u8),
    /// zlib at a level (v2 / numcodecs).
    Zlib(u8),
    /// Zstandard at a level (written at the fastest level).
    Zstd(i32),
    /// numcodecs' LZ4.
    Lz4,
    /// bzip2 (read only).
    Bz2,
    /// Byte shuffle of elements of this size (numcodecs `Shuffle`).
    Shuffle(usize),
    /// A CRC-32C appended to the bytes.
    Crc32c,
}

/// The `sharding_indexed` codec.
#[derive(Debug, Clone, PartialEq)]
pub struct Sharding {
    /// Shape of the inner chunks.
    pub chunk_shape: Vec<usize>,
    /// Their codecs.
    pub codecs: Vec<Codec>,
    /// Codecs of the index (`bytes` little-endian and `crc32c`, as written).
    pub index_codecs: Vec<Codec>,
    /// Whether the index follows the chunks (`"end"`, the default) or precedes them.
    pub index_at_end: bool,
}

fn conf<'a>(c: &'a Value, key: &str) -> Option<&'a Value> {
    c.get("configuration").and_then(|x| x.get(key))
}

fn level(v: Option<&Value>, default: i64) -> u8 {
    u8::try_from(v.and_then(Value::as_i64).unwrap_or(default).clamp(0, 9)).unwrap_or(5)
}

fn blosc_shuffle(v: Option<&Value>, typesize: usize) -> BloscShuffle {
    match v {
        Some(Value::String(s)) if s == "shuffle" => BloscShuffle::Byte,
        Some(Value::String(s)) if s == "bitshuffle" => BloscShuffle::Bit,
        Some(Value::Number(n)) => match n.as_i64() {
            Some(1) => BloscShuffle::Byte,
            Some(2) => BloscShuffle::Bit,
            // numcodecs' AUTOSHUFFLE: bit-shuffle bytes, byte-shuffle wider elements.
            Some(-1) if typesize == 1 => BloscShuffle::Bit,
            Some(-1) => BloscShuffle::Byte,
            _ => BloscShuffle::None,
        },
        _ => BloscShuffle::None,
    }
}

fn blosc(
    cname: Option<&Value>,
    clevel: Option<&Value>,
    shuffle: Option<&Value>,
    typesize: usize,
) -> Result<Codec, Error> {
    let name = cname.and_then(Value::as_str).unwrap_or("lz4");
    let cname = BloscCodec::from_name(name)
        .ok_or_else(|| Error::Unsupported(format!("Blosc compressor {name:?}")))?;
    Ok(Codec::Blosc {
        cname,
        clevel: level(clevel, 5),
        shuffle: blosc_shuffle(shuffle, typesize),
        typesize,
    })
}

/// A numcodecs compressor or filter (v2 `compressor`/`filters`, or a v3 `numcodecs.*` codec)
/// as a bytes→bytes codec.
fn numcodecs(id: &str, c: &Value, typesize: usize) -> Result<Codec, Error> {
    let get = |k: &str| c.get(k).or_else(|| conf(c, k));
    Ok(match id {
        "blosc" => blosc(get("cname"), get("clevel"), get("shuffle"), typesize)?,
        "zlib" => Codec::Zlib(level(get("level"), 1)),
        "gzip" => Codec::Gzip(level(get("level"), 1)),
        "zstd" => Codec::Zstd(
            i32::try_from(get("level").and_then(Value::as_i64).unwrap_or(0)).unwrap_or(0),
        ),
        "lz4" => Codec::Lz4,
        "bz2" => Codec::Bz2,
        "shuffle" => Codec::Shuffle(
            get("elementsize")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(typesize),
        ),
        "crc32c" => Codec::Crc32c,
        other => return Err(Error::Unsupported(format!("zarr codec {other:?}"))),
    })
}

/// The pipeline of a v2 array.
pub(crate) fn from_v2(
    dtype: DType,
    endian: Endian,
    order_f: bool,
    ndim: usize,
    filters: &Value,
    compressor: &Value,
) -> Result<Vec<Codec>, Error> {
    let typesize = dtype.size().unwrap_or(1);
    let mut out = Vec::new();
    if order_f && ndim > 1 {
        out.push(Codec::Transpose((0..ndim).rev().collect()));
    }
    let mut bytes_stage = if dtype == DType::String {
        Codec::VlenUtf8
    } else {
        Codec::Bytes(endian)
    };
    let mut after = Vec::new();
    if let Value::Array(fs) = filters {
        for f in fs {
            match f.get("id").and_then(Value::as_str) {
                Some("vlen-utf8") => bytes_stage = Codec::VlenUtf8,
                Some(id) => after.push(numcodecs(id, f, typesize)?),
                None => return Err(bad("zarr v2 filter without an id")),
            }
        }
    }
    out.push(bytes_stage);
    out.extend(after);
    if let Some(id) = compressor.get("id").and_then(Value::as_str) {
        out.push(numcodecs(id, compressor, typesize)?);
    }
    Ok(out)
}

/// The pipeline of a v3 array.
pub(crate) fn from_v3(codecs: &Value, dtype: DType, ndim: usize) -> Result<Vec<Codec>, Error> {
    let list = codecs
        .as_array()
        .ok_or_else(|| bad("codecs is not a list"))?;
    let typesize = dtype.size().unwrap_or(1);
    let mut out = Vec::with_capacity(list.len());
    for c in list {
        let name = match c {
            Value::String(s) => s.as_str(),
            _ => c
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("codec without a name"))?,
        };
        out.push(match name {
            "transpose" => {
                let order = match conf(c, "order") {
                    Some(Value::String(s)) if s == "F" => (0..ndim).rev().collect(),
                    Some(Value::String(_)) | None => (0..ndim).collect(),
                    Some(Value::Array(a)) => a
                        .iter()
                        .map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| bad("transpose order"))?,
                    Some(_) => return Err(bad("transpose order")),
                };
                Codec::Transpose(order)
            }
            "bytes" | "endian" => Codec::Bytes(match conf(c, "endian").and_then(Value::as_str) {
                Some("big") => Endian::Big,
                _ => Endian::Little,
            }),
            "vlen-utf8" | "vlen-bytes" => Codec::VlenUtf8,
            "blosc" => blosc(
                conf(c, "cname"),
                conf(c, "clevel"),
                conf(c, "shuffle"),
                conf(c, "typesize")
                    .and_then(Value::as_u64)
                    .and_then(|n| usize::try_from(n).ok())
                    .unwrap_or(typesize),
            )?,
            "gzip" => Codec::Gzip(level(conf(c, "level"), 5)),
            "zstd" => Codec::Zstd(
                i32::try_from(conf(c, "level").and_then(Value::as_i64).unwrap_or(0)).unwrap_or(0),
            ),
            "crc32c" => Codec::Crc32c,
            "sharding_indexed" => {
                let shape = conf(c, "chunk_shape")
                    .and_then(Value::as_array)
                    .and_then(|a| {
                        a.iter()
                            .map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()))
                            .collect::<Option<Vec<_>>>()
                    })
                    .ok_or_else(|| bad("sharding chunk_shape"))?;
                let inner = from_v3(conf(c, "codecs").unwrap_or(&Value::Null), dtype, ndim)?;
                let index = match conf(c, "index_codecs") {
                    Some(v) => from_v3(v, DType::U64, ndim.saturating_add(1))?,
                    None => vec![Codec::Bytes(Endian::Little), Codec::Crc32c],
                };
                Codec::Sharding(Box::new(Sharding {
                    chunk_shape: shape,
                    codecs: inner,
                    index_codecs: index,
                    index_at_end: conf(c, "index_location").and_then(Value::as_str)
                        != Some("start"),
                }))
            }
            other => match other.strip_prefix("numcodecs.") {
                Some(id) => numcodecs(id, c, typesize)?,
                None => return Err(Error::Unsupported(format!("zarr codec {other:?}"))),
            },
        });
    }
    Ok(out)
}

/// The v3 JSON of a pipeline.
pub(crate) fn to_v3(codecs: &[Codec]) -> Result<Value, Error> {
    let shuffle_name = |s: BloscShuffle| match s {
        BloscShuffle::Byte => "shuffle",
        BloscShuffle::Bit => "bitshuffle",
        _ => "noshuffle",
    };
    codecs
        .iter()
        .map(|c| {
            Ok(match c {
                Codec::Transpose(o) => json!({"name": "transpose", "configuration": {"order": o}}),
                Codec::Bytes(e) => json!({"name": "bytes", "configuration": {"endian": if *e == Endian::Big { "big" } else { "little" }}}),
                Codec::VlenUtf8 => json!({"name": "vlen-utf8", "configuration": {}}),
                Codec::Blosc { cname, clevel, shuffle, typesize } => json!({"name": "blosc", "configuration": {
                    "cname": cname.name(), "clevel": clevel, "shuffle": shuffle_name(*shuffle),
                    "typesize": typesize, "blocksize": 0}}),
                Codec::Gzip(l) => json!({"name": "gzip", "configuration": {"level": l}}),
                Codec::Zstd(l) => json!({"name": "zstd", "configuration": {"level": l, "checksum": false}}),
                Codec::Crc32c => json!({"name": "crc32c"}),
                Codec::Zlib(l) => json!({"name": "numcodecs.zlib", "configuration": {"level": l}}),
                Codec::Lz4 => json!({"name": "numcodecs.lz4", "configuration": {"acceleration": 1}}),
                Codec::Shuffle(n) => json!({"name": "numcodecs.shuffle", "configuration": {"elementsize": n}}),
                Codec::Bz2 => return Err(Error::Unsupported("writing bz2".into())),
                Codec::Sharding(s) => json!({"name": "sharding_indexed", "configuration": {
                    "chunk_shape": s.chunk_shape,
                    "codecs": to_v3(&s.codecs)?,
                    "index_codecs": to_v3(&s.index_codecs)?,
                    "index_location": if s.index_at_end { "end" } else { "start" }}}),
            })
        })
        .collect::<Result<Vec<_>, Error>>()
        .map(Value::Array)
}

/// The v2 `(filters, compressor)` of a pipeline (`bytes`/`vlen-utf8` first, then at most a
/// shuffle and a compressor).
pub(crate) fn to_v2(codecs: &[Codec]) -> Result<(Value, Value), Error> {
    let mut filters = Vec::new();
    let mut compressor = Value::Null;
    for c in codecs {
        match c {
            Codec::Bytes(_) => {}
            Codec::VlenUtf8 => filters.push(json!({"id": "vlen-utf8"})),
            Codec::Shuffle(n) => filters.push(json!({"id": "shuffle", "elementsize": n})),
            Codec::Blosc {
                cname,
                clevel,
                shuffle,
                ..
            } => {
                let s = match shuffle {
                    BloscShuffle::Byte => 1,
                    BloscShuffle::Bit => 2,
                    _ => 0,
                };
                compressor = json!({"id": "blosc", "cname": cname.name(), "clevel": clevel, "shuffle": s, "blocksize": 0});
            }
            Codec::Zlib(l) => compressor = json!({"id": "zlib", "level": l}),
            Codec::Gzip(l) => compressor = json!({"id": "gzip", "level": l}),
            Codec::Zstd(l) => compressor = json!({"id": "zstd", "level": l}),
            Codec::Lz4 => compressor = json!({"id": "lz4", "acceleration": 1}),
            other => return Err(Error::Unsupported(format!("{other:?} in zarr v2"))),
        }
    }
    let filters = if filters.is_empty() {
        Value::Null
    } else {
        Value::Array(filters)
    };
    Ok((filters, compressor))
}

// ----- elements as bytes -----

macro_rules! numeric {
    ($($var:ident $t:ty),*) => {
        fn to_bytes(data: &ArrayData, dtype: DType, endian: Endian) -> Result<Vec<u8>, Error> {
            let big = endian == Endian::Big;
            Ok(match data {
                $(ArrayData::$var(a) => {
                    let mut out = Vec::with_capacity(a.len().saturating_mul(std::mem::size_of::<$t>()));
                    for v in a.iter() {
                        out.extend_from_slice(&if big { v.to_be_bytes() } else { v.to_le_bytes() });
                    }
                    out
                })*
                ArrayData::Bool(a) => a.iter().map(|b| u8::from(*b)).collect(),
                ArrayData::Str(a) => match dtype {
                    DType::FixedBytes(n) => {
                        let mut out = Vec::with_capacity(a.len().saturating_mul(n));
                        for s in a.iter() {
                            let b = s.as_bytes();
                            if b.len() > n {
                                return Err(bad(format!("text longer than {n} bytes")));
                            }
                            out.extend_from_slice(b);
                            out.resize(out.len().saturating_add(n.saturating_sub(b.len())), 0);
                        }
                        out
                    }
                    DType::FixedUnicode(n) => {
                        let mut out = Vec::with_capacity(a.len().saturating_mul(n).saturating_mul(4));
                        for s in a.iter() {
                            let cs: Vec<char> = s.chars().collect();
                            if cs.len() > n {
                                return Err(bad(format!("text longer than {n} characters")));
                            }
                            for i in 0..n {
                                let c = cs.get(i).map_or(0, |c| u32::from(*c));
                                out.extend_from_slice(&if big { c.to_be_bytes() } else { c.to_le_bytes() });
                            }
                        }
                        out
                    }
                    _ => return Err(bad("variable-length text needs the vlen-utf8 codec")),
                },
            })
        }

        fn from_bytes(bytes: &[u8], dtype: DType, endian: Endian, shape: &[usize]) -> Result<ArrayData, Error> {
            let n: usize = shape.iter().product();
            let size = dtype.size().ok_or_else(|| bad("variable-length text needs the vlen-utf8 codec"))?;
            if bytes.len() != n.checked_mul(size).ok_or_else(|| bad("chunk too large"))? {
                return Err(bad(format!("chunk has {} bytes, expected {}", bytes.len(), n.saturating_mul(size))));
            }
            let big = endian == Endian::Big;
            let sh = IxDyn(shape);
            let shaped = |e: ndarray::ShapeError| bad(e.to_string());
            Ok(match dtype {
                $(DType::$var => {
                    let v: Vec<$t> = bytes
                        .chunks_exact(std::mem::size_of::<$t>())
                        .map(|c| {
                            let a: [u8; std::mem::size_of::<$t>()] = c.try_into().unwrap_or_default();
                            if big { <$t>::from_be_bytes(a) } else { <$t>::from_le_bytes(a) }
                        })
                        .collect();
                    ArrayData::$var(ArrayD::from_shape_vec(sh, v).map_err(shaped)?)
                })*
                DType::Bool => ArrayData::Bool(ArrayD::from_shape_vec(sh, bytes.iter().map(|b| *b != 0).collect()).map_err(shaped)?),
                DType::FixedBytes(w) => {
                    let v = bytes
                        .chunks(w.max(1))
                        .take(n)
                        .map(|c| {
                            let end = c.iter().position(|b| *b == 0).unwrap_or(c.len());
                            String::from_utf8_lossy(c.get(..end).unwrap_or_default()).into_owned()
                        })
                        .collect();
                    ArrayData::Str(ArrayD::from_shape_vec(sh, v).map_err(shaped)?)
                }
                DType::FixedUnicode(w) => {
                    let v = bytes
                        .chunks(w.saturating_mul(4).max(1))
                        .take(n)
                        .map(|c| {
                            c.chunks_exact(4)
                                .map(|q| {
                                    let a: [u8; 4] = q.try_into().unwrap_or_default();
                                    if big { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) }
                                })
                                .take_while(|u| *u != 0)
                                .map(|u| char::from_u32(u).unwrap_or(char::REPLACEMENT_CHARACTER))
                                .collect::<String>()
                        })
                        .collect();
                    ArrayData::Str(ArrayD::from_shape_vec(sh, v).map_err(shaped)?)
                }
                DType::String => return Err(bad("variable-length text needs the vlen-utf8 codec")),
            })
        }
    };
}
numeric!(I8 i8, I16 i16, I32 i32, I64 i64, U8 u8, U16 u16, U32 u32, U64 u64, F32 f32, F64 f64);

fn vlen_encode(data: &ArrayData) -> Result<Vec<u8>, Error> {
    let ArrayData::Str(a) = data else {
        return Err(bad("vlen-utf8 needs text"));
    };
    let count = u32::try_from(a.len()).map_err(|_| bad("chunk too large"))?;
    let mut out = count.to_le_bytes().to_vec();
    for s in a.iter() {
        let n = u32::try_from(s.len()).map_err(|_| bad("text too long"))?;
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    Ok(out)
}

fn vlen_decode(bytes: &[u8], shape: &[usize]) -> Result<ArrayData, Error> {
    let n: usize = shape.iter().product();
    let corrupt = || bad("vlen-utf8: corrupt chunk");
    let count = annoray_codecs::bits::le32(bytes, 0, "vlen-utf8")?;
    if usize::try_from(count).ok() != Some(n) {
        return Err(corrupt());
    }
    let mut at = 4usize;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let len = usize::try_from(annoray_codecs::bits::le32(bytes, at, "vlen-utf8")?)
            .map_err(|_| corrupt())?;
        at = at.checked_add(4).ok_or_else(corrupt)?;
        let end = at.checked_add(len).ok_or_else(corrupt)?;
        let s = bytes.get(at..end).ok_or_else(corrupt)?;
        v.push(String::from_utf8_lossy(s).into_owned());
        at = end;
    }
    Ok(ArrayData::Str(
        ArrayD::from_shape_vec(IxDyn(shape), v).map_err(|e| bad(e.to_string()))?,
    ))
}

// ----- array transforms -----

/// `data` with axes permuted (`out` axis `i` = `data` axis `order[i]`), in standard layout.
pub(crate) fn permute(data: ArrayData, order: &[usize]) -> Result<ArrayData, Error> {
    if order.len() != data.shape().len() || !is_permutation(order) {
        return Err(bad("transpose order does not match the array"));
    }
    Ok(map_each!(data, a => a.permuted_axes(IxDyn(order)).as_standard_layout().into_owned()))
}

fn is_permutation(order: &[usize]) -> bool {
    let mut seen = vec![false; order.len()];
    order.iter().all(|i| {
        seen.get_mut(*i)
            .is_some_and(|s| !std::mem::replace(s, true))
    })
}

fn inverse(order: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; order.len()];
    for (i, o) in order.iter().enumerate() {
        if let Some(slot) = inv.get_mut(*o) {
            *slot = i;
        }
    }
    inv
}

/// Split a pipeline into array→array, array→bytes and bytes→bytes stages.
fn stages(codecs: &[Codec]) -> Result<(&[Codec], &Codec, &[Codec]), Error> {
    let at = codecs
        .iter()
        .position(|c| matches!(c, Codec::Bytes(_) | Codec::VlenUtf8 | Codec::Sharding(_)))
        .ok_or_else(|| bad("no array-to-bytes codec"))?;
    let (pre, rest) = codecs.split_at(at);
    let (mid, post) = rest
        .split_first()
        .ok_or_else(|| bad("no array-to-bytes codec"))?;
    Ok((pre, mid, post))
}

/// Decode one stored chunk of `shape` (logical, C order).
pub(crate) fn decode(
    codecs: &[Codec],
    mut bytes: Vec<u8>,
    dtype: DType,
    shape: &[usize],
    fill: &Fill,
) -> Result<ArrayData, Error> {
    let (pre, mid, post) = stages(codecs)?;
    // The shape the array→bytes stage sees.
    let mut enc_shape = shape.to_vec();
    for c in pre {
        if let Codec::Transpose(o) = c {
            enc_shape = o
                .iter()
                .map(|i| enc_shape.get(*i).copied().unwrap_or(0))
                .collect();
        }
    }
    let fixed = dtype
        .size()
        .and_then(|s| s.checked_mul(enc_shape.iter().product()));
    let limit = match mid {
        Codec::Bytes(_) => fixed.unwrap_or(VLEN_LIMIT),
        _ => VLEN_LIMIT,
    };
    for c in post.iter().rev() {
        bytes = match c {
            Codec::Blosc { .. } => annoray_codecs::blosc::decompress(&bytes, limit)?,
            Codec::Gzip(_) => annoray_codecs::zlib::gzip_decompress(&bytes, limit)?,
            Codec::Zlib(_) => annoray_codecs::zlib::zlib_decompress(&bytes, limit)?,
            Codec::Zstd(_) => annoray_codecs::zstd::decompress(&bytes, limit)?,
            Codec::Lz4 => annoray_codecs::lz4::numcodecs_decompress(&bytes, limit)?,
            Codec::Bz2 => annoray_codecs::bzip2::decompress(&bytes, limit)?,
            Codec::Shuffle(n) => annoray_codecs::shuffle::unshuffle(&bytes, *n),
            Codec::Crc32c => {
                let n = bytes
                    .len()
                    .checked_sub(4)
                    .ok_or_else(|| bad("crc32c: chunk too short"))?;
                let stored = annoray_codecs::bits::le32(&bytes, n, "crc32c")?;
                bytes.truncate(n);
                if annoray_codecs::crc::crc32c(&bytes) != stored {
                    return Err(bad("crc32c checksum mismatch (corrupt chunk)"));
                }
                bytes
            }
            other => return Err(bad(format!("{other:?} is not a bytes-to-bytes codec"))),
        };
    }
    let mut data = match mid {
        Codec::Bytes(e) => from_bytes(&bytes, dtype, *e, &enc_shape)?,
        Codec::VlenUtf8 => vlen_decode(&bytes, &enc_shape)?,
        Codec::Sharding(s) => decode_shard(s, &bytes, dtype, &enc_shape, fill)?,
        _ => return Err(bad("array-to-bytes codec")),
    };
    for c in pre.iter().rev() {
        if let Codec::Transpose(o) = c {
            data = permute(data, &inverse(o))?;
        }
    }
    Ok(data)
}

/// Encode one chunk (logical shape, C order; already padded to the full chunk).
pub(crate) fn encode(
    codecs: &[Codec],
    data: ArrayData,
    dtype: DType,
    fill: &Fill,
) -> Result<Vec<u8>, Error> {
    let (pre, mid, post) = stages(codecs)?;
    let mut data = data;
    for c in pre {
        if let Codec::Transpose(o) = c {
            data = permute(data, o)?;
        }
    }
    let mut bytes = match mid {
        Codec::Bytes(e) => to_bytes(&data, dtype, *e)?,
        Codec::VlenUtf8 => vlen_encode(&data)?,
        Codec::Sharding(s) => encode_shard(s, &data, dtype, fill)?,
        _ => return Err(bad("array-to-bytes codec")),
    };
    for c in post {
        bytes = match c {
            Codec::Blosc {
                cname,
                clevel,
                shuffle,
                typesize,
            } => annoray_codecs::blosc::compress(&bytes, *typesize, *cname, *clevel, *shuffle)?,
            Codec::Gzip(l) => annoray_codecs::zlib::gzip_compress(&bytes, *l),
            Codec::Zlib(l) => annoray_codecs::zlib::zlib_compress(&bytes, *l),
            Codec::Zstd(_) => {
                if !annoray_codecs::zstd::available() {
                    return Err(Error::Unsupported(
                        "Zstandard (enable the `zstd` feature)".into(),
                    ));
                }
                annoray_codecs::zstd::compress(&bytes)
            }
            Codec::Lz4 => annoray_codecs::lz4::numcodecs_compress(&bytes)?,
            Codec::Shuffle(n) => annoray_codecs::shuffle::shuffle(&bytes, *n),
            Codec::Crc32c => {
                let crc = annoray_codecs::crc::crc32c(&bytes);
                bytes.extend_from_slice(&crc.to_le_bytes());
                bytes
            }
            other => return Err(Error::Unsupported(format!("writing {other:?}"))),
        };
    }
    Ok(bytes)
}

// ----- sharding -----

/// Inner chunks per shard along each axis.
fn grid(shard: &[usize], inner: &[usize]) -> Result<Vec<usize>, Error> {
    if shard.len() != inner.len() {
        return Err(bad("sharding chunk_shape has the wrong rank"));
    }
    shard
        .iter()
        .zip(inner)
        .map(|(s, i)| {
            if *i == 0 || s.checked_rem(*i) != Some(0) {
                Err(bad(
                    "shard shape is not a multiple of the inner chunk shape",
                ))
            } else {
                Ok(s.checked_div(*i).unwrap_or(0))
            }
        })
        .collect()
}

/// The `n`-th position (C order) in a grid.
fn unravel(mut k: usize, grid: &[usize]) -> Vec<usize> {
    let mut out = vec![0; grid.len()];
    for (slot, g) in out.iter_mut().zip(grid).rev() {
        if *g > 0 {
            *slot = k.checked_rem(*g).unwrap_or(0);
            k = k.checked_div(*g).unwrap_or(0);
        }
    }
    out
}

fn index_len(s: &Sharding, n: usize) -> Result<usize, Error> {
    let crc = s.index_codecs.iter().any(|c| matches!(c, Codec::Crc32c));
    if s.index_codecs
        .iter()
        .any(|c| !matches!(c, Codec::Bytes(_) | Codec::Crc32c))
    {
        return Err(Error::Unsupported(
            "shard index codecs other than bytes and crc32c".into(),
        ));
    }
    n.checked_mul(16)
        .and_then(|b| b.checked_add(if crc { 4 } else { 0 }))
        .ok_or_else(|| bad("shard index too large"))
}

fn decode_shard(
    s: &Sharding,
    bytes: &[u8],
    dtype: DType,
    shape: &[usize],
    fill: &Fill,
) -> Result<ArrayData, Error> {
    let g = grid(shape, &s.chunk_shape)?;
    let n: usize = g.iter().product();
    let ilen = index_len(s, n)?;
    let total = bytes.len();
    let raw = if s.index_at_end {
        bytes.get(
            total
                .checked_sub(ilen)
                .ok_or_else(|| bad("shard shorter than its index"))?..,
        )
    } else {
        bytes.get(..ilen)
    }
    .ok_or_else(|| bad("shard shorter than its index"))?;
    let mut ishape = g.clone();
    ishape.push(2);
    let ArrayData::U64(index) = decode(
        &s.index_codecs,
        raw.to_vec(),
        DType::U64,
        &ishape,
        &Fill::Int(0),
    )?
    else {
        return Err(bad("shard index"));
    };
    let entries: Vec<u64> = index.iter().copied().collect();
    let mut out = ArrayData::filled(dtype, shape, fill)?;
    for k in 0..n {
        let (Some(&off), Some(&len)) = (
            entries.get(k.saturating_mul(2)),
            entries.get(k.saturating_mul(2).saturating_add(1)),
        ) else {
            return Err(bad("shard index"));
        };
        if off == u64::MAX && len == u64::MAX {
            continue;
        }
        let at = usize::try_from(off).map_err(|_| bad("shard offset"))?;
        let end = at
            .checked_add(usize::try_from(len).map_err(|_| bad("shard length"))?)
            .ok_or_else(|| bad("shard length"))?;
        let chunk = bytes
            .get(at..end)
            .ok_or_else(|| bad("shard chunk out of range"))?;
        let inner = decode(&s.codecs, chunk.to_vec(), dtype, &s.chunk_shape, fill)?;
        let pos = unravel(k, &g);
        let start: Vec<usize> = pos
            .iter()
            .zip(&s.chunk_shape)
            .map(|(p, c)| p.saturating_mul(*c))
            .collect();
        crate::array::assign(&mut out, &start, &inner)?;
    }
    Ok(out)
}

fn encode_shard(
    s: &Sharding,
    data: &ArrayData,
    dtype: DType,
    fill: &Fill,
) -> Result<Vec<u8>, Error> {
    let g = grid(data.shape(), &s.chunk_shape)?;
    let n: usize = g.iter().product();
    let ilen = index_len(s, n)?;
    let mut body = Vec::new();
    let mut index = Vec::with_capacity(n.saturating_mul(2));
    let base = if s.index_at_end { 0 } else { ilen };
    for k in 0..n {
        let pos = unravel(k, &g);
        let start: Vec<usize> = pos
            .iter()
            .zip(&s.chunk_shape)
            .map(|(p, c)| p.saturating_mul(*c))
            .collect();
        let inner = crate::array::extract(data, &start, &s.chunk_shape)?;
        if crate::array::all_fill(&inner, fill) {
            index.extend([u64::MAX, u64::MAX]);
            continue;
        }
        let enc = encode(&s.codecs, inner, dtype, fill)?;
        let off =
            u64::try_from(base.saturating_add(body.len())).map_err(|_| bad("shard too large"))?;
        let len = u64::try_from(enc.len()).map_err(|_| bad("shard too large"))?;
        index.extend([off, len]);
        body.extend(enc);
    }
    let mut ishape = g;
    ishape.push(2);
    let idx = ArrayData::U64(
        ArrayD::from_shape_vec(IxDyn(&ishape), index).map_err(|e| bad(e.to_string()))?,
    );
    let idx_bytes = encode(&s.index_codecs, idx, DType::U64, &Fill::Int(0))?;
    Ok(if s.index_at_end {
        body.extend(idx_bytes);
        body
    } else {
        let mut out = idx_bytes;
        out.extend(body);
        out
    })
}
