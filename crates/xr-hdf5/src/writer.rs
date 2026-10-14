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

use xr_core::{AttrValue, Owned, TypedArray};
use xr_io::IoError;

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
    pub data: TypedArray<Owned>,
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
    /// Fill value as raw little-endian element bytes.
    pub fill: Option<Vec<u8>>,
    /// Attributes, in order.
    pub attrs: Vec<(String, AttrData)>,
}

impl WDataset {
    /// A contiguous dataset with no attributes.
    #[must_use]
    pub fn new(data: TypedArray<Owned>) -> Self {
        let rank = data.ndim();
        Self {
            data,
            char_data: false,
            unlimited: vec![false; rank],
            chunks: None,
            deflate: None,
            shuffle: false,
            fill: None,
            attrs: Vec::new(),
        }
    }
}

/// An attribute value.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AttrData {
    /// Numbers as 1-D arrays; `Str` as a scalar fixed-length string; `Strs` as variable-length
    /// strings.
    Value(AttrValue),
    /// A scalar variable-length UTF-8 string.
    VarString(String),
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

#[derive(Default)]
struct Plan {
    paths: HashMap<String, usize>,
    blocks: Vec<Block>,
    /// Block index of each object's header.
    headers: Vec<usize>,
    heap: Vec<HeapObj>,
}

fn too_big(what: &str) -> IoError {
    bad(format!("{what} is too large to write"))
}

fn u16_of(n: usize, what: &str) -> Result<u16, IoError> {
    u16::try_from(n).map_err(|_| too_big(what))
}

fn u32_of(n: usize, what: &str) -> Result<u32, IoError> {
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
fn fixed_string_type(size: usize, utf8: bool) -> Result<Vec<u8>, IoError> {
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

fn element_type(d: &TypedArray<Owned>, char_data: bool) -> Result<Vec<u8>, IoError> {
    Ok(match d {
        TypedArray::U8(_) if char_data => fixed_string_type(1, false)?,
        TypedArray::I8(_) => int_type(1, true),
        TypedArray::U8(_) => int_type(1, false),
        TypedArray::I16(_) => int_type(2, true),
        TypedArray::U16(_) => int_type(2, false),
        TypedArray::I32(_) => int_type(4, true),
        TypedArray::U32(_) => int_type(4, false),
        TypedArray::I64(_) => int_type(8, true),
        TypedArray::U64(_) => int_type(8, false),
        TypedArray::F32(_) => float_type(4),
        TypedArray::F64(_) => float_type(8),
        TypedArray::Str(_) => var_string_type(),
        other => {
            return Err(IoError::Unsupported(format!(
                "writing HDF5 element type {}",
                other.dtype()
            )));
        }
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

fn le_bytes(d: &TypedArray<Owned>) -> Option<Vec<u8>> {
    macro_rules! le {
        ($x:expr) => {
            $x.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
    }
    Some(match d {
        TypedArray::I8(x) => le!(x),
        TypedArray::U8(x) => x.iter().copied().collect(),
        TypedArray::I16(x) => le!(x),
        TypedArray::U16(x) => le!(x),
        TypedArray::I32(x) => le!(x),
        TypedArray::U32(x) => le!(x),
        TypedArray::I64(x) => le!(x),
        TypedArray::U64(x) => le!(x),
        TypedArray::F32(x) => le!(x),
        TypedArray::F64(x) => le!(x),
        _ => return None,
    })
}

impl Plan {
    fn block(&mut self, bytes: Bytes, checksum: bool) -> usize {
        self.blocks.push(Block { bytes, checksum });
        self.blocks.len().saturating_sub(1)
    }

    /// A variable-length element pointing at a new heap object.
    fn vlen(&mut self, out: &mut Bytes, len: usize, obj: HeapObj) -> Result<(), IoError> {
        self.heap.push(obj);
        let k = self.heap.len().saturating_sub(1);
        out.u32(u32_of(len, "variable-length element")?);
        out.addr(Target::Heap(k));
        out.u32(0); // index: patched with the collection-relative index
        Ok(())
    }

    fn object(&self, path: &str) -> Result<usize, IoError> {
        self.paths
            .get(path)
            .copied()
            .ok_or_else(|| bad(format!("reference to unknown object {path:?}")))
    }

    /// Element bytes of a dataset (strings go to the global heap).
    fn element_bytes(&mut self, data: &TypedArray<Owned>) -> Result<Bytes, IoError> {
        let mut out = Bytes::default();
        if let TypedArray::Str(a) = data {
            for s in a {
                self.vlen(&mut out, s.len(), HeapObj::Bytes(s.as_bytes().to_vec()))?;
            }
        } else {
            out.data = le_bytes(data)
                .ok_or_else(|| IoError::Unsupported(format!("writing {} data", data.dtype())))?;
        }
        Ok(out)
    }

    fn attribute(&mut self, name: &str, value: &AttrData) -> Result<Bytes, IoError> {
        let (dtype, space, data) = match value {
            AttrData::Value(v) => {
                let num = |dt: Vec<u8>, n: usize, bytes: Vec<u8>| {
                    let dims = [n];
                    (
                        dt,
                        dataspace(if n == 0 { None } else { Some(&dims) }, &[]),
                        Bytes {
                            data: bytes,
                            fixups: Vec::new(),
                        },
                    )
                };
                macro_rules! le {
                    ($x:expr) => {
                        $x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()
                    };
                }
                match v {
                    AttrValue::Str(s) => {
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
                    AttrValue::Strs(v) => {
                        let mut b = Bytes::default();
                        for s in v {
                            self.vlen(&mut b, s.len(), HeapObj::Bytes(s.as_bytes().to_vec()))?;
                        }
                        let dims = [v.len()];
                        (
                            var_string_type(),
                            dataspace(if v.is_empty() { None } else { Some(&dims) }, &[]),
                            b,
                        )
                    }
                    AttrValue::I8(x) => num(int_type(1, true), x.len(), le!(x)),
                    AttrValue::U8(x) => num(int_type(1, false), x.len(), x.clone()),
                    AttrValue::I16(x) => num(int_type(2, true), x.len(), le!(x)),
                    AttrValue::U16(x) => num(int_type(2, false), x.len(), le!(x)),
                    AttrValue::I32(x) => num(int_type(4, true), x.len(), le!(x)),
                    AttrValue::U32(x) => num(int_type(4, false), x.len(), le!(x)),
                    AttrValue::I64(x) => num(int_type(8, true), x.len(), le!(x)),
                    AttrValue::U64(x) => num(int_type(8, false), x.len(), le!(x)),
                    AttrValue::F32(x) => num(float_type(4), x.len(), le!(x)),
                    AttrValue::F64(x) => num(float_type(8), x.len(), le!(x)),
                    other => {
                        return Err(IoError::Unsupported(format!("attribute value {other:?}")));
                    }
                }
            }
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
    ) -> Result<Option<usize>, IoError> {
        let shape = ds.data.shape().to_vec();
        let rank = shape.len();
        let elements = self.element_bytes(&ds.data)?;
        let filtered = ds.deflate.is_some() || ds.shuffle;
        if filtered && !elements.fixups.is_empty() {
            return Err(IoError::Unsupported(
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
        let chunk_bytes = chunk_elems
            .checked_mul(esize)
            .ok_or_else(|| too_big("chunk"))?;
        let fill: Vec<u8> = match &ds.fill {
            Some(f) if f.len() == esize => f.clone(),
            _ => vec![0; esize],
        };
        // Leaf entries: (chunk offset, stored size, data block).
        let mut leaves: Vec<(Vec<u64>, u32, usize)> = Vec::with_capacity(total);
        for ci in 0..total {
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
            let mut chunk = Bytes {
                data: fill.iter().copied().cycle().take(chunk_bytes).collect(),
                fixups: Vec::new(),
            };
            // Copy every element of the chunk that lies inside the dataset.
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
            if ds.shuffle {
                chunk.data = shuffle(&chunk.data, esize);
            }
            if let Some(level) = ds.deflate {
                chunk.data = miniz_oxide::deflate::compress_to_vec_zlib(&chunk.data, level.min(9));
            }
            let size = u32_of(chunk.data.len(), "chunk")?;
            let block = self.block(chunk, false);
            leaves.push((origin.iter().map(|o| *o as u64).collect(), size, block));
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
                    b.addr(Target::Block(*child));
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
                parents.push((first.0, first.1, node));
            }
            if parents.len() == 1 {
                return Ok(parents.first().map(|p| p.2));
            }
            nodes = parents;
            level = level.saturating_add(1);
        }
    }

    fn dataset(&mut self, id: usize, ds: &WDataset) -> Result<(), IoError> {
        let shape = ds.data.shape().to_vec();
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
                let data = self.element_bytes(&ds.data)?;
                let size = data.data.len();
                if size == 0 {
                    layout.u64(UNDEF);
                } else {
                    let block = self.block(data, false);
                    layout.addr(Target::Block(block));
                }
                layout.u64(size as u64);
            }
            Some(c) => {
                layout.u8(2);
                layout.u8(u8::try_from(rank.saturating_add(1)).map_err(|_| too_big("rank"))?);
                match self.chunked(ds, c, esize)? {
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
        if chunks.is_some() && (ds.shuffle || ds.deflate.is_some()) {
            let mut p = Bytes::default();
            p.u8(2);
            p.u8(u8::from(ds.shuffle).saturating_add(u8::from(ds.deflate.is_some())));
            if ds.shuffle {
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
            msgs.push((0x0B, p));
        }
        self.finish_object(id, msgs, &ds.attrs)
    }

    fn group(&mut self, id: usize, g: &WGroup, path: &str) -> Result<(), IoError> {
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
    ) -> Result<(), IoError> {
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
                return Err(IoError::Unsupported(format!(
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

    fn walk_objects(&mut self, g: &WGroup, path: &str) -> Result<(), IoError> {
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

fn esize_of(d: &TypedArray<Owned>) -> usize {
    match d {
        TypedArray::I8(_) | TypedArray::U8(_) => 1,
        TypedArray::I16(_) | TypedArray::U16(_) => 2,
        TypedArray::I32(_) | TypedArray::U32(_) | TypedArray::F32(_) => 4,
        TypedArray::Str(_) => 16,
        _ => 8,
    }
}

/// Serialise a tree as an HDF5 file.
///
/// # Errors
/// [`IoError::Unsupported`] for element types or features the writer does not handle;
/// [`IoError::Backend`] for sizes beyond the format's limits or dangling references.
pub fn write_file(root: &WGroup) -> Result<Vec<u8>, IoError> {
    let mut plan = Plan::default();
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

    // Lay out: superblock, then every block, 8-byte aligned.
    let mut addrs = Vec::with_capacity(plan.blocks.len());
    let mut at: usize = 48;
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
    let object_addr = |id: usize| -> Result<u64, IoError> {
        plan.headers
            .get(id)
            .and_then(|b| addrs.get(*b))
            .copied()
            .ok_or_else(|| bad("object without a header"))
    };

    let mut out = vec![0u8; at];
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
        let start =
            usize::try_from(addrs.get(i).copied().unwrap_or(0)).map_err(|_| too_big("file"))?;
        if let Some(slot) = out.get_mut(start..start.saturating_add(data.len())) {
            slot.copy_from_slice(&data);
        }
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
    if let Some(slot) = out.get_mut(..sb.data.len()) {
        slot.copy_from_slice(&sb.data);
    }
    Ok(out)
}
