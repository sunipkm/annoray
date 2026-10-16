//! Writing HDF5 files.
//!
//! Emits a conservative subset every HDF5 reader accepts, close to what netCDF-C writes:
//! superblock v2, version-2 object headers with link and attribute creation order tracked,
//! compact links and attributes, contiguous or chunked datasets (version-1 B-tree chunk index,
//! deflate and shuffle), variable-length strings in global heap collections, and object
//! references (for dimension scales).
//!
//! Writing takes a whole tree and lays it out in two passes: first every block (heap
//! collections, data, chunks, B-tree nodes, object headers) is built with placeholders for
//! addresses, then addresses are assigned, placeholders patched, and checksums computed. This
//! lets objects reference each other in any direction.

use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};
use std::sync::Arc;

use crate::{Array, BloscCodec, BloscShuffle, Compression, ValueType};

/// Values read slice by slice as they are written, so a dataset need never be in memory at
/// once.
pub trait Source: std::fmt::Debug + Send + Sync {
    /// The element type.
    fn dtype(&self) -> ValueType;

    /// The shape.
    fn shape(&self) -> &[usize];

    /// The values in the box `[start, start + count)`, of [`dtype`](Self::dtype).
    ///
    /// # Errors
    /// Whatever stops the values being produced; writing then fails with it.
    fn read_slice(
        &self,
        start: &[usize],
        count: &[usize],
    ) -> Result<Array, Box<dyn std::error::Error + Send + Sync>>;
}
use crate::Error;

use crate::bad;
use crate::checksum::lookup3;
use crate::filters::shuffle;

const UNDEF: u64 = u64::MAX;
/// Entries per v1 B-tree node: 2K with the library default K = 32 for chunk indexes.
const BTREE_ENTRIES: usize = 64;
/// Global heap objects per collection (indices are 16-bit) and the minimum collection size.
const HEAP_OBJECTS: usize = 60_000;
const HEAP_MIN: usize = 4096;

/// A group to write.
#[derive(Debug, Clone, Default)]
pub struct WGroup {
    /// Attributes, in order.
    pub attrs: Vec<(String, AttrData)>,
    /// Members, in creation order.
    pub members: Vec<(String, WNode)>,
}

/// A group member.
#[derive(Debug, Clone)]
pub enum WNode {
    /// A subgroup.
    Group(WGroup),
    /// A dataset.
    Dataset(WDataset),
}

/// A dataset to write.
#[derive(Debug, Clone)]
pub struct WDataset {
    /// Values. Strings become variable-length UTF-8 unless `char_data` is set on `U8` data.
    /// With a [`source`](Self::source), only its element type matters (an empty array will
    /// do).
    pub data: Array,
    /// Read the values from here instead, slice by slice, as they are written (so a dataset
    /// need never be in memory at once). Not for strings.
    pub source: Option<Arc<dyn Source>>,
    /// Write `U8` data as 1-byte strings (netCDF's `NC_CHAR`).
    pub char_data: bool,
    /// Per dimension: unlimited (forces chunking).
    pub unlimited: Vec<bool>,
    /// Chunk shape; `None` for contiguous storage (unless a dimension is unlimited).
    pub chunks: Option<Vec<usize>>,
    /// Deflate level.
    pub deflate: Option<u8>,
    /// Shuffle filter.
    pub shuffle: bool,
    /// A compression codec other than deflate (Blosc brings its own shuffle).
    pub compression: Option<Compression>,
    /// Fill value as raw little-endian element bytes.
    pub fill: Option<Vec<u8>>,
    /// Allocate no storage and write no values (readers see fill values). netCDF-C writes
    /// dimension-only scales this way.
    pub skip_data: bool,
    /// Attributes, in order.
    pub attrs: Vec<(String, AttrData)>,
}

impl WDataset {
    /// A dataset whose values are read from `source` as they are written.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for string sources.
    pub fn from_source(source: Arc<dyn Source>) -> Result<Self, Error> {
        if source.dtype() == ValueType::Str {
            return Err(Error::Unsupported(
                "writing strings from a lazy source".into(),
            ));
        }
        let proto = Array::empty(source.dtype(), &[0]).ok_or_else(|| bad("empty array"))?;
        let mut d = Self::new(proto);
        d.unlimited = vec![false; source.shape().len()];
        d.source = Some(source);
        Ok(d)
    }

    /// The shape of the values.
    fn shape(&self) -> Vec<usize> {
        self.source
            .as_ref()
            .map_or_else(|| self.data.shape().to_vec(), |s| s.shape().to_vec())
    }

    /// A contiguous dataset with no attributes.
    #[must_use]
    pub fn new(data: Array) -> Self {
        let rank = data.ndim();
        Self {
            data,
            source: None,
            char_data: false,
            unlimited: vec![false; rank],
            chunks: None,
            deflate: None,
            shuffle: false,
            compression: None,
            fill: None,
            skip_data: false,
            attrs: Vec::new(),
        }
    }
}

/// An attribute value.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AttrData {
    /// Values with the array's shape (1-D for netCDF; with no elements, the null dataspace).
    /// Strings are variable-length.
    Array(Array),
    /// A scalar fixed-length string exactly as long as the text (netCDF text attributes).
    Text(String),
    /// A scalar variable-length UTF-8 string.
    VarString(String),
    /// A scalar fixed-length string including its NUL terminator (HDF5 dimension-scale `CLASS`
    /// and `NAME`).
    CString(String),
    /// A scalar 32-bit integer.
    ScalarI32(i32),
    /// Object references to the objects at these absolute paths.
    References(Vec<String>),
    /// Variable-length sequences of object references (`DIMENSION_LIST`).
    ReferenceLists(Vec<Vec<String>>),
    /// `(reference, index)` pairs as a compound `{dataset, dimension}` (`REFERENCE_LIST`).
    ReferenceIndexPairs(Vec<(String, i32)>),
}

// ----- blocks with address placeholders -----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Object(usize),
    Block(usize),
    Heap(usize),
    /// An address already in the file (data written as it was produced).
    Abs(u64),
}

#[derive(Debug, Default)]
struct Bytes {
    data: Vec<u8>,
    fixups: Vec<(usize, Target)>,
}

impl Bytes {
    fn u8(&mut self, v: u8) {
        self.data.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }
    fn raw(&mut self, b: &[u8]) {
        self.data.extend_from_slice(b);
    }
    fn addr(&mut self, t: Target) {
        self.fixups.push((self.data.len(), t));
        self.u64(0);
    }
    fn append(&mut self, other: Bytes) {
        let base = self.data.len();
        self.fixups.extend(
            other
                .fixups
                .into_iter()
                .map(|(at, t)| (at.saturating_add(base), t)),
        );
        self.data.extend(other.data);
    }
}

#[derive(Debug)]
struct Block {
    bytes: Bytes,
    /// Object headers get a trailing lookup3 checksum.
    checksum: bool,
}

/// A global heap object: plain bytes, or references (patched to object addresses).
#[derive(Debug)]
enum HeapObj {
    Bytes(Vec<u8>),
    Refs(Vec<usize>),
}

/// Where data goes as it is produced.
trait Sink: Write + Seek {}
impl<T: Write + Seek> Sink for T {}

struct Plan<'a> {
    paths: HashMap<String, usize>,
    blocks: Vec<Block>,
    /// Block index of each object's header.
    headers: Vec<usize>,
    heap: Vec<HeapObj>,
    /// Data is written here as it is produced, from the superblock's end on; every other
    /// block is laid out after it.
    out: &'a mut dyn Sink,
    /// The next free address.
    pos: u64,
}

fn io(e: &std::io::Error) -> Error {
    bad(format!("writing: {e}"))
}

/// Bytes the data of a source take per write, at most (rows are never split).
const SLAB_BYTES: usize = 32 << 20;

fn too_big(what: &str) -> Error {
    bad(format!("{what} is too large to write"))
}

fn u16_of(n: usize, what: &str) -> Result<u16, Error> {
    u16::try_from(n).map_err(|_| too_big(what))
}

fn u32_of(n: usize, what: &str) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| too_big(what))
}

fn path_join(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

// ----- datatypes and dataspaces -----

fn int_type(size: u8, signed: bool) -> Vec<u8> {
    let mut m = vec![0x10, if signed { 0x08 } else { 0 }, 0, 0];
    m.extend_from_slice(&u32::from(size).to_le_bytes());
    m.extend_from_slice(&0u16.to_le_bytes());
    m.extend_from_slice(&u16::from(size).saturating_mul(8).to_le_bytes());
    m
}

fn float_type(size: u8) -> Vec<u8> {
    let (sign, exp_loc, exp_size, mant_size, bias): (u8, u8, u8, u8, u32) = if size == 4 {
        (31, 23, 8, 23, 127)
    } else {
        (63, 52, 11, 52, 1023)
    };
    let mut m = vec![0x11, 0x20, sign, 0];
    m.extend_from_slice(&u32::from(size).to_le_bytes());
    m.extend_from_slice(&0u16.to_le_bytes());
    m.extend_from_slice(&u16::from(size).saturating_mul(8).to_le_bytes());
    m.extend_from_slice(&[exp_loc, exp_size, 0, mant_size]);
    m.extend_from_slice(&bias.to_le_bytes());
    m
}

/// Fixed-length, null-terminated string.
fn fixed_string_type(size: usize, utf8: bool) -> Result<Vec<u8>, Error> {
    let mut m = vec![0x13, if utf8 { 0x10 } else { 0 }, 0, 0];
    m.extend_from_slice(&u32_of(size, "string")?.to_le_bytes());
    Ok(m)
}

/// Variable-length UTF-8 string (base type: 1-byte unsigned integer, as HDF5 writes it).
fn var_string_type() -> Vec<u8> {
    let mut m = vec![0x19, 0x01, 0x01, 0];
    m.extend_from_slice(&16u32.to_le_bytes());
    m.extend(int_type(1, false));
    m
}

fn reference_type() -> Vec<u8> {
    let mut m = vec![0x17, 0, 0, 0];
    m.extend_from_slice(&8u32.to_le_bytes());
    m
}

fn reference_list_type() -> Vec<u8> {
    let mut m = vec![0x19, 0, 0, 0];
    m.extend_from_slice(&16u32.to_le_bytes());
    m.extend(reference_type());
    m
}

/// `{ dataset: object reference, dimension: int32 }`, compound version 3.
fn reference_pair_type() -> Vec<u8> {
    let mut m = vec![0x36, 2, 0, 0];
    m.extend_from_slice(&12u32.to_le_bytes());
    m.extend_from_slice(b"dataset\0");
    m.push(0);
    m.extend(reference_type());
    m.extend_from_slice(b"dimension\0");
    m.push(8);
    m.extend(int_type(4, true));
    m
}

fn element_type(d: &Array, char_data: bool) -> Result<Vec<u8>, Error> {
    Ok(match d {
        Array::U8(_) if char_data => fixed_string_type(1, false)?,
        Array::I8(_) => int_type(1, true),
        Array::U8(_) => int_type(1, false),
        Array::I16(_) => int_type(2, true),
        Array::U16(_) => int_type(2, false),
        Array::I32(_) => int_type(4, true),
        Array::U32(_) => int_type(4, false),
        Array::I64(_) => int_type(8, true),
        Array::U64(_) => int_type(8, false),
        Array::F32(_) => float_type(4),
        Array::F64(_) => float_type(8),
        Array::Str(_) => var_string_type(),
    })
}

/// Dataspace v2: `None` dims for the null dataspace, `Some([])` for scalars.
fn dataspace(dims: Option<&[usize]>, unlimited: &[bool]) -> Vec<u8> {
    let Some(dims) = dims else {
        return vec![2, 0, 0, 2];
    };
    let rank = u8::try_from(dims.len()).unwrap_or(u8::MAX);
    let has_max = unlimited.iter().any(|u| *u);
    let mut m = vec![
        2,
        rank,
        u8::from(has_max),
        if dims.is_empty() { 0 } else { 1 },
    ];
    for d in dims {
        m.extend_from_slice(&(*d as u64).to_le_bytes());
    }
    if has_max {
        for (d, u) in dims
            .iter()
            .zip(unlimited.iter().chain(std::iter::repeat(&false)))
        {
            m.extend_from_slice(&(if *u { UNDEF } else { *d as u64 }).to_le_bytes());
        }
    }
    m
}

fn le_bytes(d: &Array) -> Option<Vec<u8>> {
    macro_rules! le {
        ($x:expr) => {
            $x.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
    }
    Some(match d {
        Array::I8(x) => le!(x),
        Array::U8(x) => x.iter().copied().collect(),
        Array::I16(x) => le!(x),
        Array::U16(x) => le!(x),
        Array::I32(x) => le!(x),
        Array::U32(x) => le!(x),
        Array::I64(x) => le!(x),
        Array::U64(x) => le!(x),
        Array::F32(x) => le!(x),
        Array::F64(x) => le!(x),
        _ => return None,
    })
}

impl Plan<'_> {
    /// Write `bytes` at the next 8-byte-aligned address; that address.
    fn put_data(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let at = self
            .pos
            .checked_add(7)
            .map(|x| x & !7)
            .ok_or_else(|| too_big("file"))?;
        self.out
            .seek(SeekFrom::Start(self.pos))
            .map_err(|e| io(&e))?;
        let pad = usize::try_from(at.saturating_sub(self.pos)).unwrap_or(0);
        self.out
            .write_all([0u8; 8].get(..pad.min(8)).unwrap_or_default())
            .map_err(|e| io(&e))?;
        self.out.write_all(bytes).map_err(|e| io(&e))?;
        self.pos = at
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| too_big("file"))?;
        Ok(at)
    }

    /// Write `bytes` right after the previous data (continuing one contiguous region).
    fn put_more(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.out
            .seek(SeekFrom::Start(self.pos))
            .map_err(|e| io(&e))?;
        self.out.write_all(bytes).map_err(|e| io(&e))?;
        self.pos = self
            .pos
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| too_big("file"))?;
        Ok(())
    }

    fn block(&mut self, bytes: Bytes, checksum: bool) -> usize {
        self.blocks.push(Block { bytes, checksum });
        self.blocks.len().saturating_sub(1)
    }

    /// A variable-length element pointing at a new heap object.
    fn vlen(&mut self, out: &mut Bytes, len: usize, obj: HeapObj) -> Result<(), Error> {
        self.heap.push(obj);
        let k = self.heap.len().saturating_sub(1);
        out.u32(u32_of(len, "variable-length element")?);
        out.addr(Target::Heap(k));
        out.u32(0); // index: patched with the collection-relative index
        Ok(())
    }

    fn object(&self, path: &str) -> Result<usize, Error> {
        self.paths
            .get(path)
            .copied()
            .ok_or_else(|| bad(format!("reference to unknown object {path:?}")))
    }

    /// Element bytes of a dataset (strings go to the global heap).
    fn element_bytes(&mut self, data: &Array) -> Result<Bytes, Error> {
        let mut out = Bytes::default();
        if let Array::Str(a) = data {
            for s in a {
                self.vlen(&mut out, s.len(), HeapObj::Bytes(s.as_bytes().to_vec()))?;
            }
        } else {
            out.data = le_bytes(data)
                .ok_or_else(|| Error::Unsupported(format!("writing {} data", data.dtype())))?;
        }
        Ok(out)
    }

    fn attribute(&mut self, name: &str, value: &AttrData) -> Result<Bytes, Error> {
        let (dtype, space, data) = match value {
            AttrData::Array(v) => {
                let shape = v.shape();
                let space = dataspace((!v.is_empty()).then_some(shape), &[]);
                let bytes = if let Array::Str(a) = v {
                    let mut b = Bytes::default();
                    for s in a {
                        self.vlen(&mut b, s.len(), HeapObj::Bytes(s.as_bytes().to_vec()))?;
                    }
                    b
                } else {
                    Bytes {
                        data: le_bytes(v).ok_or_else(|| {
                            Error::Unsupported(format!("attribute of {}", v.dtype()))
                        })?,
                        fixups: Vec::new(),
                    }
                };
                (element_type(v, false)?, space, bytes)
            }
            AttrData::Text(s) => {
                let size = s.len().max(1);
                let mut bytes = s.as_bytes().to_vec();
                bytes.resize(size, 0);
                (
                    fixed_string_type(size, true)?,
                    dataspace(Some(&[]), &[]),
                    Bytes {
                        data: bytes,
                        fixups: Vec::new(),
                    },
                )
            }
            AttrData::CString(s) => {
                let mut bytes = s.as_bytes().to_vec();
                bytes.push(0);
                (
                    fixed_string_type(bytes.len(), false)?,
                    dataspace(Some(&[]), &[]),
                    Bytes {
                        data: bytes,
                        fixups: Vec::new(),
                    },
                )
            }
            AttrData::ScalarI32(v) => (
                int_type(4, true),
                dataspace(Some(&[]), &[]),
                Bytes {
                    data: v.to_le_bytes().to_vec(),
                    fixups: Vec::new(),
                },
            ),
            AttrData::VarString(s) => {
                let mut b = Bytes::default();
                self.vlen(&mut b, s.len(), HeapObj::Bytes(s.as_bytes().to_vec()))?;
                (var_string_type(), dataspace(Some(&[]), &[]), b)
            }
            AttrData::References(paths) => {
                let mut b = Bytes::default();
                for p in paths {
                    b.addr(Target::Object(self.object(p)?));
                }
                (reference_type(), dataspace(Some(&[paths.len()]), &[]), b)
            }
            AttrData::ReferenceLists(lists) => {
                let mut b = Bytes::default();
                for l in lists {
                    let ids = l
                        .iter()
                        .map(|p| self.object(p))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.vlen(&mut b, ids.len(), HeapObj::Refs(ids))?;
                }
                (
                    reference_list_type(),
                    dataspace(Some(&[lists.len()]), &[]),
                    b,
                )
            }
            AttrData::ReferenceIndexPairs(pairs) => {
                let mut b = Bytes::default();
                for (p, i) in pairs {
                    b.addr(Target::Object(self.object(p)?));
                    b.raw(&i.to_le_bytes());
                }
                (
                    reference_pair_type(),
                    dataspace(Some(&[pairs.len()]), &[]),
                    b,
                )
            }
        };
        let mut m = Bytes::default();
        m.u8(3); // version
        m.u8(0); // flags
        m.u16(u16_of(name.len().saturating_add(1), "attribute name")?);
        m.u16(u16_of(dtype.len(), "attribute datatype")?);
        m.u16(u16_of(space.len(), "attribute dataspace")?);
        m.u8(1); // UTF-8 name
        m.raw(name.as_bytes());
        m.u8(0);
        m.raw(&dtype);
        m.raw(&space);
        m.append(data);
        Ok(m)
    }

    /// Chunk data blocks and the B-tree over them; returns the root node's block.
    fn chunked(
        &mut self,
        ds: &WDataset,
        cdims: &[usize],
        esize: usize,
    ) -> Result<Option<usize>, Error> {
        let shape = ds.shape();
        let rank = shape.len();
        let elements = match &ds.source {
            Some(_) => Bytes::default(),
            None => self.element_bytes(&ds.data)?,
        };
        let filtered = ds.deflate.is_some() || ds.shuffle || ds.compression.is_some();
        if filtered && !elements.fixups.is_empty() {
            return Err(Error::Unsupported(
                "compressing variable-length strings".into(),
            ));
        }
        let nchunks: Vec<usize> = shape
            .iter()
            .zip(cdims)
            .map(|(d, c)| d.div_ceil((*c).max(1)))
            .collect();
        let total: usize = nchunks
            .iter()
            .try_fold(1usize, |a, n| a.checked_mul(*n))
            .ok_or_else(|| too_big("chunk grid"))?;
        if total == 0 {
            return Ok(None);
        }
        let chunk_elems: usize = cdims
            .iter()
            .try_fold(1usize, |a, c| a.checked_mul(*c))
            .ok_or_else(|| too_big("chunk"))?;
        // Checked here so the buffer below cannot overflow.
        chunk_elems
            .checked_mul(esize)
            .ok_or_else(|| too_big("chunk"))?;
        let fill: Vec<u8> = match &ds.fill {
            Some(f) if f.len() == esize => f.clone(),
            _ => vec![0; esize],
        };
        let make = |ci: usize| -> Result<(Vec<usize>, Bytes), Error> {
            // Chunk coordinates (row-major).
            let mut idx = vec![0usize; rank];
            let mut rem = ci;
            for (slot, n) in idx.iter_mut().zip(&nchunks).rev() {
                *slot = rem.checked_rem(*n).unwrap_or(0);
                rem = rem.checked_div(*n).unwrap_or(0);
            }
            let origin: Vec<usize> = idx
                .iter()
                .zip(cdims)
                .map(|(i, c)| i.saturating_mul(*c))
                .collect();
            // The part of the chunk inside the dataset.
            let count: Vec<usize> = origin
                .iter()
                .zip(cdims)
                .zip(&shape)
                .map(|((o, c), d)| (*c).min(d.saturating_sub(*o)))
                .collect();
            let mut chunk = Bytes {
                data: fill.repeat(chunk_elems),
                fixups: Vec::new(),
            };
            match &ds.source {
                Some(src) => {
                    let slab = src
                        .read_slice(&origin, &count)
                        .map_err(|e| bad(format!("reading data to write: {e}")))?;
                    let bytes = le_bytes(&slab).ok_or_else(|| {
                        Error::Unsupported(format!("writing {} data", slab.dtype()))
                    })?;
                    copy_rows(
                        &bytes,
                        &count,
                        &vec![0; rank],
                        &mut chunk.data,
                        cdims,
                        &count,
                        esize,
                    )?;
                }
                None if elements.fixups.is_empty() => {
                    copy_rows(
                        &elements.data,
                        &shape,
                        &origin,
                        &mut chunk.data,
                        cdims,
                        &count,
                        esize,
                    )?;
                }
                None => {
                    // Strings: element by element, keeping their heap references.
                    for e in 0..chunk_elems {
                        let mut rem = e;
                        let mut inside = true;
                        let mut src = 0usize;
                        let mut local = vec![0usize; rank];
                        for (slot, c) in local.iter_mut().zip(cdims).rev() {
                            *slot = rem.checked_rem(*c).unwrap_or(0);
                            rem = rem.checked_div(*c).unwrap_or(0);
                        }
                        for ((l, o), d) in local.iter().zip(&origin).zip(&shape) {
                            let g = o.saturating_add(*l);
                            if g >= *d {
                                inside = false;
                                break;
                            }
                            src = src.saturating_mul(*d).saturating_add(g);
                        }
                        if !inside {
                            continue;
                        }
                        let s = src.saturating_mul(esize);
                        let dst = e.saturating_mul(esize);
                        if let (Some(d), Some(v)) = (
                            chunk.data.get_mut(dst..dst.saturating_add(esize)),
                            elements.data.get(s..s.saturating_add(esize)),
                        ) {
                            d.copy_from_slice(v);
                        }
                        for (at, t) in elements
                            .fixups
                            .iter()
                            .filter(|(at, _)| *at >= s && *at < s.saturating_add(esize))
                        {
                            chunk
                                .fixups
                                .push((dst.saturating_add(at.saturating_sub(s)), *t));
                        }
                    }
                }
            }
            if ds.shuffle && !blosc(ds) {
                chunk.data = shuffle(&chunk.data, esize);
            }
            if let Some(level) = ds.deflate {
                chunk.data = annoray_codecs::zlib::zlib_compress(&chunk.data, level);
            }
            if let Some(c) = ds.compression {
                chunk.data = compress(c, &chunk.data, esize)?;
            }
            Ok((origin, chunk))
        };
        // Filtering dominates writing: chunks are built and compressed in parallel, a batch at
        // a time, then written in order. The batch holds at most the budget
        // (`set_batch_bytes`) of chunks, and at least one.
        #[cfg(feature = "rayon")]
        let batch = if filtered {
            let chunk_bytes = chunk_elems.saturating_mul(esize).max(1);
            rayon::current_num_threads()
                .saturating_mul(2)
                .min(crate::batch_bytes().checked_div(chunk_bytes).unwrap_or(1))
                .max(1)
        } else {
            1
        };
        #[cfg(not(feature = "rayon"))]
        let batch = 1usize;
        // Leaf entries: (chunk offset, stored size, where the chunk is).
        let mut leaves: Vec<(Vec<u64>, u32, Target)> = Vec::with_capacity(total);
        let mut next = 0usize;
        while next < total {
            let end = next.saturating_add(batch).min(total);
            #[cfg(feature = "rayon")]
            let built = {
                use rayon::prelude::*;
                (next..end)
                    .into_par_iter()
                    .map(make)
                    .collect::<Result<Vec<_>, Error>>()?
            };
            #[cfg(not(feature = "rayon"))]
            let built = (next..end).map(make).collect::<Result<Vec<_>, Error>>()?;
            for (origin, chunk) in built {
                let size = u32_of(chunk.data.len(), "chunk")?;
                let place = if chunk.fixups.is_empty() {
                    Target::Abs(self.put_data(&chunk.data)?)
                } else {
                    Target::Block(self.block(chunk, false))
                };
                leaves.push((origin.iter().map(|o| *o as u64).collect(), size, place));
            }
            next = end;
        }
        // Upper bound key: the dataset extent rounded up to whole chunks.
        let bound: Vec<u64> = nchunks
            .iter()
            .zip(cdims)
            .map(|(n, c)| n.saturating_mul(*c) as u64)
            .collect();
        let mut level: u8 = 0;
        let mut nodes = leaves;
        loop {
            let mut parents = Vec::new();
            for group in nodes.chunks(BTREE_ENTRIES) {
                let mut b = Bytes::default();
                b.raw(b"TREE");
                b.u8(1);
                b.u8(level);
                b.u16(u16_of(group.len(), "B-tree node")?);
                b.u64(UNDEF);
                b.u64(UNDEF);
                let key = |b: &mut Bytes, size: u32, off: &[u64]| {
                    b.u32(size);
                    b.u32(0);
                    for o in off {
                        b.u64(*o);
                    }
                    b.u64(0);
                };
                for (off, size, child) in group {
                    key(&mut b, *size, off);
                    b.addr(*child);
                }
                key(&mut b, 0, &bound);
                // Pad to the full node size: the HDF5 library reads whole nodes.
                let key_size = 8usize.saturating_add(8usize.saturating_mul(rank.saturating_add(1)));
                let full = 24usize
                    .saturating_add(BTREE_ENTRIES.saturating_mul(key_size.saturating_add(8)))
                    .saturating_add(key_size);
                b.data.resize(full.max(b.data.len()), 0);
                let first = group
                    .first()
                    .map(|(o, s, _)| (o.clone(), *s))
                    .unwrap_or_default();
                let node = self.block(b, false);
                parents.push((first.0, first.1, Target::Block(node)));
            }
            if parents.len() == 1 {
                return Ok(parents.first().and_then(|p| match p.2 {
                    Target::Block(k) => Some(k),
                    _ => None,
                }));
            }
            nodes = parents;
            level = level.saturating_add(1);
        }
    }

    fn dataset(&mut self, id: usize, ds: &WDataset) -> Result<(), Error> {
        let shape = ds.shape();
        let rank = shape.len();
        let dtype = element_type(&ds.data, ds.char_data)?;
        let esize = esize_of(&ds.data);
        let chunks = match (&ds.chunks, ds.unlimited.iter().any(|u| *u)) {
            (Some(c), _) if c.len() == rank && rank > 0 => {
                Some(c.iter().map(|x| (*x).max(1)).collect::<Vec<_>>())
            }
            (_, true) => Some(shape.iter().map(|d| (*d).clamp(1, 1024)).collect()),
            _ => None,
        };
        let mut msgs: Vec<(u8, Bytes)> = Vec::new();
        msgs.push((
            0x01,
            Bytes {
                data: dataspace(Some(&shape), &ds.unlimited),
                fixups: Vec::new(),
            },
        ));
        msgs.push((
            0x03,
            Bytes {
                data: dtype,
                fixups: Vec::new(),
            },
        ));
        // Fill value v3: allocation late (contiguous) or incremental (chunked), write "if set".
        let mut fill = Bytes::default();
        fill.u8(3);
        let alloc = if chunks.is_some() { 3 } else { 2 };
        match &ds.fill {
            Some(f) => {
                fill.u8(alloc | (2 << 2) | 0x20);
                fill.u32(u32_of(f.len(), "fill value")?);
                fill.raw(f);
            }
            None => fill.u8(alloc | (2 << 2)),
        }
        msgs.push((0x05, fill));
        let mut layout = Bytes::default();
        layout.u8(3);
        match &chunks {
            None => {
                layout.u8(1);
                let n: usize = shape.iter().product();
                let size = n.saturating_mul(esize);
                if ds.skip_data || size == 0 {
                    layout.u64(UNDEF);
                } else if let Some(src) = &ds.source {
                    // Slabs along the first axis, written back to back.
                    let addr = self.contiguous_from(src.as_ref(), &shape, esize)?;
                    layout.addr(Target::Abs(addr));
                } else {
                    let data = self.element_bytes(&ds.data)?;
                    if data.fixups.is_empty() {
                        layout.addr(Target::Abs(self.put_data(&data.data)?));
                    } else {
                        let block = self.block(data, false);
                        layout.addr(Target::Block(block));
                    }
                }
                layout.u64(size as u64);
            }
            Some(c) => {
                layout.u8(2);
                layout.u8(u8::try_from(rank.saturating_add(1)).map_err(|_| too_big("rank"))?);
                let root = if ds.skip_data {
                    None
                } else {
                    self.chunked(ds, c, esize)?
                };
                match root {
                    Some(root) => layout.addr(Target::Block(root)),
                    None => layout.u64(UNDEF),
                }
                for d in c {
                    layout.u32(u32_of(*d, "chunk dimension")?);
                }
                layout.u32(u32_of(esize, "element")?);
            }
        }
        msgs.push((0x08, layout));
        if chunks.is_some() && (ds.shuffle || ds.deflate.is_some() || ds.compression.is_some()) {
            let shuffled = ds.shuffle && !blosc(ds);
            let mut p = Bytes::default();
            p.u8(2);
            p.u8(u8::from(shuffled)
                .saturating_add(u8::from(ds.deflate.is_some()))
                .saturating_add(u8::from(ds.compression.is_some())));
            if shuffled {
                p.u16(2);
                p.u16(1); // optional
                p.u16(1);
                p.u32(u32_of(esize, "element")?);
            }
            if let Some(level) = ds.deflate {
                p.u16(1);
                p.u16(1);
                p.u16(1);
                p.u32(u32::from(level.min(9)));
            }
            if let Some(c) = ds.compression {
                let chunk_bytes = chunks
                    .as_ref()
                    .map_or(Some(esize), |c| {
                        c.iter().try_fold(esize, |a, d| a.checked_mul(*d))
                    })
                    .ok_or_else(|| bad("chunk too large"))?;
                let (id, client) = filter_params(c, esize, chunk_bytes)?;
                // Plugin filters (id ≥ 256) carry a name length (0: no name).
                p.u16(id);
                p.u16(0);
                p.u16(0); // mandatory
                p.u16(u16::try_from(client.len()).unwrap_or(0));
                for v in client {
                    p.u32(v);
                }
            }
            msgs.push((0x0B, p));
        }
        self.finish_object(id, msgs, &ds.attrs)
    }

    /// Write a source's values contiguously, a slab of whole rows at a time; the address.
    fn contiguous_from(
        &mut self,
        src: &dyn Source,
        shape: &[usize],
        esize: usize,
    ) -> Result<u64, Error> {
        let read = |start: &[usize], count: &[usize]| -> Result<Vec<u8>, Error> {
            let t = src
                .read_slice(start, count)
                .map_err(|e| bad(format!("reading data to write: {e}")))?;
            le_bytes(&t).ok_or_else(|| Error::Unsupported(format!("writing {} data", t.dtype())))
        };
        let Some((&rows, rest)) = shape.split_first() else {
            return self.put_data(&read(&[], &[])?);
        };
        let row_bytes = rest
            .iter()
            .try_fold(esize, |a, d| a.checked_mul(*d))
            .ok_or_else(|| too_big("row"))?
            .max(1);
        let step = SLAB_BYTES.checked_div(row_bytes).unwrap_or(1).max(1);
        let mut start = vec![0usize; shape.len()];
        let mut addr = None;
        let mut r = 0usize;
        while r < rows {
            let n = step.min(rows.saturating_sub(r));
            if let Some(s) = start.first_mut() {
                *s = r;
            }
            let mut count = shape.to_vec();
            if let Some(c) = count.first_mut() {
                *c = n;
            }
            let bytes = read(&start, &count)?;
            match addr {
                None => addr = Some(self.put_data(&bytes)?),
                Some(_) => self.put_more(&bytes)?,
            }
            r = r.saturating_add(n);
        }
        addr.ok_or_else(|| bad("empty contiguous data"))
    }

    fn group(&mut self, id: usize, g: &WGroup, path: &str) -> Result<(), Error> {
        let mut msgs: Vec<(u8, Bytes)> = Vec::new();
        // Link info: creation order tracked, compact storage.
        let mut linfo = Bytes::default();
        linfo.u8(0);
        linfo.u8(1);
        linfo.u64(g.members.len() as u64);
        linfo.u64(UNDEF);
        linfo.u64(UNDEF);
        msgs.push((0x02, linfo));
        msgs.push((
            0x0A,
            Bytes {
                data: vec![0, 0],
                fixups: Vec::new(),
            },
        )); // group info
        for (order, (name, _)) in g.members.iter().enumerate() {
            let child = self.object(&path_join(path, name))?;
            let mut l = Bytes::default();
            l.u8(1);
            let wide = name.len() > 255;
            l.u8(u8::from(wide) | 0x04 | 0x10);
            l.u64(order as u64);
            l.u8(1); // UTF-8
            if wide {
                l.u16(u16_of(name.len(), "link name")?);
            } else {
                l.u8(u8::try_from(name.len()).unwrap_or(0));
            }
            l.raw(name.as_bytes());
            l.addr(Target::Object(child));
            msgs.push((0x06, l));
        }
        self.finish_object(id, msgs, &g.attrs)
    }

    /// Add attributes, assemble the object header, and record it.
    fn finish_object(
        &mut self,
        id: usize,
        mut msgs: Vec<(u8, Bytes)>,
        attrs: &[(String, AttrData)],
    ) -> Result<(), Error> {
        let mut ainfo = Bytes::default();
        ainfo.u8(0);
        ainfo.u8(1);
        ainfo.u16(u16_of(attrs.len(), "attribute count")?);
        ainfo.u64(UNDEF);
        ainfo.u64(UNDEF);
        msgs.push((0x15, ainfo));
        let mut order: u16 = 0;
        let mut attr_msgs = Vec::new();
        for (name, value) in attrs {
            let m = self.attribute(name, value)?;
            if m.data.len() > usize::from(u16::MAX) {
                return Err(Error::Unsupported(format!(
                    "attribute {name:?} is larger than 64 KiB (dense attribute storage is not implemented)"
                )));
            }
            attr_msgs.push((order, m));
            order = order.saturating_add(1);
        }
        let mut body = Bytes::default();
        for (kind, m) in msgs {
            body.u8(kind);
            body.u16(u16_of(m.data.len(), "header message")?);
            body.u8(0);
            body.u16(0); // creation order (not meaningful for non-attribute messages)
            body.append(m);
        }
        for (o, m) in attr_msgs {
            body.u8(0x0C);
            body.u16(u16_of(m.data.len(), "attribute")?);
            body.u8(0);
            body.u16(o);
            body.append(m);
        }
        let mut h = Bytes::default();
        h.raw(b"OHDR");
        h.u8(2);
        h.u8(0x02 | 0x04); // 4-byte chunk size; attribute creation order tracked
        h.u32(u32_of(body.data.len(), "object header")?);
        h.append(body);
        h.u32(0); // checksum
        let block = self.block(h, true);
        if let Some(slot) = self.headers.get_mut(id) {
            *slot = block;
        }
        Ok(())
    }

    fn walk_ids(&mut self, g: &WGroup, path: &str) {
        for (name, node) in &g.members {
            let p = path_join(path, name);
            let id = self.headers.len();
            self.headers.push(usize::MAX);
            self.paths.insert(p.clone(), id);
            if let WNode::Group(child) = node {
                self.walk_ids(child, &p);
            }
        }
    }

    fn walk_objects(&mut self, g: &WGroup, path: &str) -> Result<(), Error> {
        for (name, node) in &g.members {
            let p = path_join(path, name);
            let id = self.object(&p)?;
            match node {
                WNode::Group(child) => {
                    self.walk_objects(child, &p)?;
                    self.group(id, child, &p)?;
                }
                WNode::Dataset(d) => self.dataset(id, d)?,
            }
        }
        Ok(())
    }
}

fn esize_of(d: &Array) -> usize {
    match d {
        Array::I8(_) | Array::U8(_) => 1,
        Array::I16(_) | Array::U16(_) => 2,
        Array::I32(_) | Array::U32(_) | Array::F32(_) => 4,
        Array::Str(_) => 16,
        _ => 8,
    }
}

/// Serialise a tree as an HDF5 file in memory.
///
/// # Errors
/// As for [`write_to`].
pub fn write_file(root: &WGroup) -> Result<Vec<u8>, Error> {
    let mut out = std::io::Cursor::new(Vec::new());
    write_to(root, &mut out)?;
    Ok(out.into_inner())
}

/// Write a tree as an HDF5 file to `out` (a file, typically). Values are written as they
/// are produced, read from [`WDataset::source`]s slice by slice, so datasets larger than
/// memory can be written; object headers, B-trees and heaps follow the data, and the
/// superblock is written last.
///
/// # Errors
/// [`Error::Unsupported`] for element types or features the writer does not handle;
/// [`Error::Invalid`] for sizes beyond the format's limits, dangling references, source
/// read errors, or write errors.
pub fn write_to(root: &WGroup, out: &mut (impl Write + Seek)) -> Result<(), Error> {
    let mut plan = Plan {
        paths: HashMap::new(),
        blocks: Vec::new(),
        headers: Vec::new(),
        heap: Vec::new(),
        out,
        pos: 48,
    };
    plan.headers.push(usize::MAX);
    plan.paths.insert("/".into(), 0);
    plan.walk_ids(root, "/");
    plan.walk_objects(root, "/")?;
    plan.group(0, root, "/")?;

    // Global heap collections.
    let mut heap_slots: Vec<(usize, u32)> = Vec::with_capacity(plan.heap.len()); // (collection block, index)
    let heap = std::mem::take(&mut plan.heap);
    for (c, objs) in heap.chunks(HEAP_OBJECTS).enumerate() {
        let mut b = Bytes::default();
        b.raw(b"GCOL");
        b.u8(1);
        b.raw(&[0, 0, 0]);
        b.u64(0); // collection size, set below
        for (i, obj) in objs.iter().enumerate() {
            let index = u16_of(i.saturating_add(1), "global heap index")?;
            b.u16(index);
            b.u16(0);
            b.u32(0);
            let len = match obj {
                HeapObj::Bytes(v) => v.len(),
                HeapObj::Refs(r) => r.len().saturating_mul(8),
            };
            b.u64(len as u64);
            match obj {
                HeapObj::Bytes(v) => b.raw(v),
                HeapObj::Refs(r) => {
                    for id in r {
                        b.addr(Target::Object(*id));
                    }
                }
            }
            while b.data.len() % 8 != 0 {
                b.u8(0);
            }
            heap_slots.push((c, u32::from(index)));
        }
        // Free space object filling the rest of the collection (its size includes its header).
        let used = b.data.len();
        let size = used.saturating_add(16).max(HEAP_MIN);
        b.u16(0);
        b.u16(0);
        b.u32(0);
        b.u64(size.saturating_sub(used) as u64);
        b.data.resize(size, 0);
        if let Some(sz) = b.data.get_mut(8..16) {
            sz.copy_from_slice(&(size as u64).to_le_bytes());
        }
        let block = plan.block(b, false);
        heap_slots
            .iter_mut()
            .filter(|(cb, _)| *cb == c)
            .for_each(|(cb, _)| *cb = block);
    }

    // Lay out every block after the data, 8-byte aligned.
    let mut addrs = Vec::with_capacity(plan.blocks.len());
    let mut at = usize::try_from(plan.pos).map_err(|_| too_big("file"))?;
    for b in &plan.blocks {
        at = at
            .checked_add(7)
            .map(|x| x & !7)
            .ok_or_else(|| too_big("file"))?;
        addrs.push(at as u64);
        at = at
            .checked_add(b.bytes.data.len())
            .ok_or_else(|| too_big("file"))?;
    }
    let eof = at as u64;
    let object_addr = |id: usize| -> Result<u64, Error> {
        plan.headers
            .get(id)
            .and_then(|b| addrs.get(*b))
            .copied()
            .ok_or_else(|| bad("object without a header"))
    };

    for (i, b) in plan.blocks.iter().enumerate() {
        let mut data = b.bytes.data.clone();
        for (pos, t) in &b.bytes.fixups {
            let (value, index) = match t {
                Target::Object(id) => (object_addr(*id)?, None),
                Target::Block(k) => (
                    addrs
                        .get(*k)
                        .copied()
                        .ok_or_else(|| bad("dangling block"))?,
                    None,
                ),
                Target::Abs(a) => (*a, None),
                Target::Heap(k) => {
                    let (cb, index) = heap_slots
                        .get(*k)
                        .copied()
                        .ok_or_else(|| bad("dangling heap object"))?;
                    (
                        addrs.get(cb).copied().ok_or_else(|| bad("dangling heap"))?,
                        Some(index),
                    )
                }
            };
            if let Some(slot) = data.get_mut(*pos..pos.saturating_add(8)) {
                slot.copy_from_slice(&value.to_le_bytes());
            }
            if let (Some(index), Some(slot)) = (
                index,
                data.get_mut(pos.saturating_add(8)..pos.saturating_add(12)),
            ) {
                slot.copy_from_slice(&index.to_le_bytes());
            }
        }
        if b.checksum {
            let n = data.len().saturating_sub(4);
            let sum = lookup3(data.get(..n).unwrap_or_default(), 0);
            if let Some(slot) = data.get_mut(n..) {
                slot.copy_from_slice(&sum.to_le_bytes());
            }
        }
        let start = addrs.get(i).copied().unwrap_or(0);
        plan.out.seek(SeekFrom::Start(start)).map_err(|e| io(&e))?;
        plan.out.write_all(&data).map_err(|e| io(&e))?;
    }

    // Superblock v2.
    let mut sb = Bytes::default();
    sb.raw(crate::format::SIGNATURE);
    sb.raw(&[2, 8, 8, 0]);
    sb.u64(0); // base address
    sb.u64(UNDEF); // superblock extension
    sb.u64(eof);
    sb.u64(object_addr(0)?);
    let sum = lookup3(&sb.data, 0);
    sb.u32(sum);
    plan.out.seek(SeekFrom::Start(0)).map_err(|e| io(&e))?;
    plan.out.write_all(&sb.data).map_err(|e| io(&e))?;
    // The file ends at the last block (data may have been the last thing written).
    let end = eof.max(plan.pos);
    plan.out.seek(SeekFrom::Start(end)).map_err(|e| io(&e))?;
    plan.out.flush().map_err(|e| io(&e))?;
    Ok(())
}

/// Copy a box of extent `count` starting at `src_start` in `src` (row-major, shape
/// `src_shape`) to the start of `dst` (row-major, shape `dst_shape`), a row at a time.
fn copy_rows(
    src: &[u8],
    src_shape: &[usize],
    src_start: &[usize],
    dst: &mut [u8],
    dst_shape: &[usize],
    count: &[usize],
    esize: usize,
) -> Result<(), Error> {
    let rank = count.len();
    if count.contains(&0) {
        return Ok(());
    }
    let Some((&last, outer)) = count.split_last() else {
        // A scalar.
        let (Some(d), Some(s)) = (dst.get_mut(..esize), src.get(..esize)) else {
            return Err(bad("scalar out of range"));
        };
        d.copy_from_slice(s);
        return Ok(());
    };
    let run = last.checked_mul(esize).ok_or_else(|| too_big("row"))?;
    let rows: usize = outer.iter().product();
    let mut idx = vec![0usize; rank.saturating_sub(1)];
    for _ in 0..rows {
        let (mut s, mut d) = (0usize, 0usize);
        for k in 0..rank {
            let i = if k.saturating_add(1) == rank {
                0
            } else {
                idx.get(k).copied().unwrap_or(0)
            };
            let sk = src_start.get(k).copied().unwrap_or(0).saturating_add(i);
            s = s
                .saturating_mul(src_shape.get(k).copied().unwrap_or(0))
                .saturating_add(sk);
            d = d
                .saturating_mul(dst_shape.get(k).copied().unwrap_or(0))
                .saturating_add(i);
        }
        let (s, d) = (s.saturating_mul(esize), d.saturating_mul(esize));
        let (Some(dd), Some(ss)) = (
            dst.get_mut(d..d.saturating_add(run)),
            src.get(s..s.saturating_add(run)),
        ) else {
            return Err(bad("copy outside the data"));
        };
        dd.copy_from_slice(ss);
        // Next row index (row-major over the outer dimensions).
        for (k, slot) in idx.iter_mut().enumerate().rev() {
            *slot = slot.saturating_add(1);
            if *slot < outer.get(k).copied().unwrap_or(0) {
                break;
            }
            *slot = 0;
        }
    }
    Ok(())
}

fn blosc(ds: &WDataset) -> bool {
    matches!(ds.compression, Some(Compression::Blosc { .. }))
}

/// Blosc's code for a codec inside its frames.
/// Encode one chunk with a plugin codec.
fn compress(c: Compression, data: &[u8], esize: usize) -> Result<Vec<u8>, Error> {
    match c {
        Compression::Zstd { .. } if annoray_codecs::zstd::available() => {
            Ok(annoray_codecs::zstd::compress(data))
        }
        Compression::Lz4 => crate::filters::lz4_compress(data),
        Compression::Blosc {
            codec,
            level,
            shuffle,
        } => Ok(annoray_codecs::blosc::compress(
            data, esize, codec, level, shuffle,
        )?),
        other => Err(Error::Unsupported(format!(
            "writing {other:?} (needs the `zstd` feature)"
        ))),
    }
}

/// The filter id and client values that describe a codec, as the HDF5 plugins write them.
fn filter_params(
    c: Compression,
    esize: usize,
    chunk_bytes: usize,
) -> Result<(u16, Vec<u32>), Error> {
    let big = || bad("chunk too large");
    Ok(match c {
        Compression::Zstd { level } => (
            crate::filters::ZSTD,
            vec![u32::from_ne_bytes(level.to_ne_bytes())],
        ),
        Compression::Lz4 => (
            crate::filters::LZ4,
            vec![u32::try_from(chunk_bytes).map_err(|_| big())?],
        ),
        Compression::Blosc {
            codec,
            level,
            shuffle,
        } => {
            // Blosc's filter codes: 0 BloscLZ, 1 LZ4, 2 LZ4HC, 3 Snappy, 4 zlib, 5 Zstandard.
            let code = match codec {
                BloscCodec::Lz4 | BloscCodec::Lz4Hc => 1,
                BloscCodec::Zlib => 4,
                BloscCodec::Zstd => 5,
                _ => return Err(Error::Unsupported("writing Blosc with BloscLZ".into())),
            };
            let shuffle = match shuffle {
                BloscShuffle::Byte => 1,
                BloscShuffle::Bit => 2,
                _ => 0,
            };
            (
                crate::filters::BLOSC,
                vec![
                    2,
                    2,
                    u32::try_from(esize).map_err(|_| big())?,
                    u32::try_from(chunk_bytes).map_err(|_| big())?,
                    u32::from(level.min(9)),
                    shuffle,
                    code,
                ],
            )
        }
    })
}
