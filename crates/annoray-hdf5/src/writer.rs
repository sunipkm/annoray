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

use crate::{Array, BloscCodec, BloscShuffle, Compression, Datatype, ValueType};

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
pub struct WGroup<'a> {
    /// Attributes, in order.
    pub attrs: Vec<(String, AttrData)>,
    /// Members, in creation order.
    pub members: Vec<(String, WNode<'a>)>,
}

/// A group member.
#[derive(Debug, Clone)]
// Built once per dataset, never in bulk: boxing the dataset would only complicate building
// trees.
#[allow(clippy::large_enum_variant)]
pub enum WNode<'a> {
    /// A subgroup.
    Group(WGroup<'a>),
    /// A dataset.
    Dataset(WDataset<'a>),
    /// A committed (named) datatype, as netCDF-4 stores its user-defined types. Datasets of
    /// an equal type need not refer to it: readers match types by structure.
    Datatype(Datatype),
}

/// A dataset to write; `'a` is how long its [`source`](Self::source) may borrow from.
#[derive(Debug, Clone)]
pub struct WDataset<'a> {
    /// Values. Strings become variable-length UTF-8 unless `char_data` is set on `U8` data.
    /// With a [`source`](Self::source), only its element type matters (an empty array will
    /// do).
    pub data: Array,
    /// Read the values from here instead, slice by slice, as they are written (so a dataset
    /// need never be in memory at once). Not for strings.
    pub source: Option<Arc<dyn Source + 'a>>,
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
    /// Fletcher-32 checksum filter (forces chunking): a checksum after each chunk, verified
    /// when it is read.
    pub fletcher32: bool,
    /// Store numbers big-endian (the values are swapped as they are written).
    pub big_endian: bool,
    /// A compound (record) dataset: the record type and one array per field (as
    /// [`Dataset::read_compound`](crate::Dataset::read_compound) gives them); `data` is then
    /// unused.
    pub compound: Option<WCompound>,
    /// Store integers as this enumeration (its base type the integers' own, in the dataset's
    /// byte order); written inline: a group may also hold it as a committed
    /// [`WNode::Datatype`], which readers match by structure.
    pub enumeration: Option<Datatype>,
    /// A variable-length (ragged) dataset: each element's length, in the dataset's shape.
    /// [`data`](Self::data) then holds every element's values concatenated in C order, and
    /// the dataset is a variable-length sequence of their type (netCDF `vlen`).
    pub ragged: Option<ndarray::ArrayD<u64>>,
    /// Store floats as IEEE half precision (`float16`), rounded to nearest, ties to even.
    /// netCDF has no such type: netCDF-C cannot read these datasets (h5py and h5netcdf can).
    pub half: bool,
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

impl<'a> WDataset<'a> {
    /// A dataset whose values are read from `source` as they are written.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for string sources.
    pub fn from_source(source: Arc<dyn Source + 'a>) -> Result<Self, Error> {
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
        if let Some(c) = &self.compound {
            return c.shape.clone();
        }
        if let Some(r) = &self.ragged {
            return r.shape().to_vec();
        }
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
            fletcher32: false,
            big_endian: false,
            half: false,
            ragged: None,
            compound: None,
            enumeration: None,
            compression: None,
            fill: None,
            skip_data: false,
            attrs: Vec::new(),
        }
    }
}

/// A compound dataset to write: the record type and the values of each leaf field (dotted
/// paths for nested members; array members with their dimensions after the dataset's).
#[derive(Debug, Clone)]
pub struct WCompound {
    /// The record type ([`Datatype::Compound`]).
    pub dtype: Datatype,
    /// The dataset's shape.
    pub shape: Vec<usize>,
    /// `(path, values)` for every leaf field, the values in the member's type.
    pub fields: Vec<(String, Array)>,
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
    /// Floats stored as IEEE half precision (rounded to nearest, ties to even): the
    /// `_FillValue` of a half-precision dataset.
    Half(Vec<f32>),
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
    /// Or one at this offset, over the bytes before it (a B-tree node, padded after it).
    checksum_at: Option<usize>,
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

fn int_type(size: u8, signed: bool, big_endian: bool) -> Vec<u8> {
    let mut m = vec![
        0x10,
        (if signed { 0x08 } else { 0 }) | u8::from(big_endian),
        0,
        0,
    ];
    m.extend_from_slice(&u32::from(size).to_le_bytes());
    m.extend_from_slice(&0u16.to_le_bytes());
    m.extend_from_slice(&u16::from(size).saturating_mul(8).to_le_bytes());
    m
}

fn float_type(size: u8, big_endian: bool) -> Vec<u8> {
    let (sign, exp_loc, exp_size, mant_size, bias): (u8, u8, u8, u8, u32) = match size {
        2 => (15, 10, 5, 10, 15),
        4 => (31, 23, 8, 23, 127),
        _ => (63, 52, 11, 52, 1023),
    };
    let mut m = vec![0x11, 0x20 | u8::from(big_endian), sign, 0];
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
    m.extend(int_type(1, false, false));
    m
}

/// A datatype message for `dt` (numbers, strings, enumerations, compounds and arrays of
/// them; version 3 messages for the composite classes).
fn datatype_message(dt: &Datatype) -> Result<Vec<u8>, Error> {
    let header = |class_version: u8, bits: u16, size: usize| -> Result<Vec<u8>, Error> {
        let mut m = vec![class_version];
        m.extend_from_slice(&bits.to_le_bytes());
        m.push(0);
        m.extend_from_slice(&u32_of(size, "datatype")?.to_le_bytes());
        Ok(m)
    };
    Ok(match dt {
        Datatype::Int {
            size,
            signed,
            big_endian,
        } => int_type(*size, *signed, *big_endian),
        Datatype::Float { size, big_endian } => float_type(*size, *big_endian),
        Datatype::FixedString { size, pad } => {
            let pad = match pad {
                crate::StrPad::NullTerm => 0u8,
                crate::StrPad::NullPad => 1,
                crate::StrPad::SpacePad => 2,
            };
            let mut m = vec![0x13, pad | 0x10, 0, 0];
            m.extend_from_slice(&u32_of(*size, "string")?.to_le_bytes());
            m
        }
        Datatype::VarString { .. } => var_string_type(),
        Datatype::Enum { base, members } => {
            let n = u16::try_from(members.len()).map_err(|_| too_big("enumeration"))?;
            let mut m = header(0x38, n, base.size())?;
            m.extend(datatype_message(base)?);
            for (name, _) in members {
                m.extend_from_slice(name.as_bytes());
                m.push(0);
            }
            for (_, v) in members {
                m.extend_from_slice(v);
            }
            m
        }
        Datatype::Compound { size, members } => {
            let n = u16::try_from(members.len()).map_err(|_| too_big("compound"))?;
            let mut m = header(0x36, n, *size)?;
            let offset_bytes = match *size {
                0..=0xff => 1,
                0x100..=0xffff => 2,
                0x1_0000..=0xff_ffff => 3,
                _ => 4,
            };
            for mem in members {
                m.extend_from_slice(mem.name.as_bytes());
                m.push(0);
                let off = (mem.offset as u64).to_le_bytes();
                m.extend_from_slice(off.get(..offset_bytes).unwrap_or_default());
                m.extend(datatype_message(&mem.dtype)?);
            }
            m
        }
        Datatype::ArrayOf { base, dims } => {
            let mut m = header(0x3A, 0, dt.size())?;
            m.push(u8::try_from(dims.len()).map_err(|_| too_big("array rank"))?);
            for d in dims {
                m.extend_from_slice(&u32_of(*d, "array dimension")?.to_le_bytes());
            }
            m.extend(datatype_message(base)?);
            m
        }
        other => {
            return Err(Error::Unsupported(format!("writing datatype {other:?}")));
        }
    })
}

/// A variable-length sequence of `base` (class 9, type sequence).
fn sequence_type(base: Vec<u8>) -> Vec<u8> {
    let mut m = vec![0x19, 0, 0, 0];
    m.extend_from_slice(&16u32.to_le_bytes());
    m.extend(base);
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
    m.extend(int_type(4, true, false));
    m
}

fn element_type(d: &Array, char_data: bool, be: bool, half: bool) -> Result<Vec<u8>, Error> {
    Ok(match d {
        Array::F32(_) | Array::F64(_) if half => float_type(2, be),
        Array::U8(_) if char_data => fixed_string_type(1, false)?,
        Array::I8(_) => int_type(1, true, be),
        Array::U8(_) => int_type(1, false, be),
        Array::I16(_) => int_type(2, true, be),
        Array::U16(_) => int_type(2, false, be),
        Array::I32(_) => int_type(4, true, be),
        Array::U32(_) => int_type(4, false, be),
        Array::I64(_) => int_type(8, true, be),
        Array::U64(_) => int_type(8, false, be),
        Array::F32(_) => float_type(4, be),
        Array::F64(_) => float_type(8, be),
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

/// The file bytes of numbers: little-endian, or big-endian when `big_endian`; floats as
/// halves when `half`.
pub(crate) fn file_bytes(d: &Array, big_endian: bool, half: bool) -> Option<Vec<u8>> {
    let halves = |it: &mut dyn Iterator<Item = u16>| -> Vec<u8> {
        it.flat_map(|h| {
            if big_endian {
                h.to_be_bytes()
            } else {
                h.to_le_bytes()
            }
        })
        .collect()
    };
    match d {
        Array::F32(x) if half => {
            return Some(halves(&mut x.iter().map(|v| crate::half::from_f32(*v))));
        }
        Array::F64(x) if half => {
            return Some(halves(&mut x.iter().map(|v| crate::half::from_f64(*v))));
        }
        _ => {}
    }
    let mut b = le_bytes(d)?;
    if big_endian {
        let size = esize_of(d).max(1);
        for c in b.chunks_exact_mut(size) {
            c.reverse();
        }
    }
    Some(b)
}

fn le_bytes(d: &Array) -> Option<Vec<u8>> {
    // Contiguous values on a little-endian machine are already their file bytes.
    macro_rules! le {
        ($x:expr) => {
            match $x.as_slice() {
                Some(s) if cfg!(target_endian = "little") => bytemuck::cast_slice(s).to_vec(),
                _ => $x.iter().flat_map(|v| v.to_le_bytes()).collect(),
            }
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
        self.blocks.push(Block {
            bytes,
            checksum,
            checksum_at: None,
        });
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

    /// The records of a compound dataset, packed from its fields (variable-length strings go
    /// to the global heap).
    fn compound_bytes(&mut self, c: &WCompound) -> Result<Bytes, Error> {
        let Datatype::Compound { size, members } = &c.dtype else {
            return Err(bad("a compound dataset needs a compound type"));
        };
        let mut leaves = Vec::new();
        crate::dtype::fields(members, "", 0, &mut leaves)?;
        let n = c
            .shape
            .iter()
            .try_fold(1usize, |a, d| a.checked_mul(*d))
            .ok_or_else(|| too_big("compound dataset"))?;
        // Each field, ready to copy: fixed-size bytes per element, or strings.
        enum Column<'c> {
            Fixed {
                bytes: Vec<u8>,
                size: usize,
            },
            Text {
                strings: Vec<&'c String>,
                size: Option<usize>,
                pad: u8,
            },
        }
        let mut columns = Vec::with_capacity(leaves.len());
        for leaf in &leaves {
            let (_, values) = c
                .fields
                .iter()
                .find(|(p, _)| *p == leaf.path)
                .ok_or_else(|| bad(format!("compound field {:?} missing", leaf.path)))?;
            let per = leaf.dims.iter().product::<usize>().max(1);
            if values.len() != n.saturating_mul(per) {
                return Err(bad(format!(
                    "compound field {:?} has the wrong size",
                    leaf.path
                )));
            }
            let column = match leaf.dtype {
                Datatype::FixedString { size, pad } => Column::Text {
                    strings: match values {
                        Array::Str(a) => a.iter().collect(),
                        _ => return Err(bad(format!("compound field {:?}: text", leaf.path))),
                    },
                    size: Some(*size),
                    pad: if *pad == crate::StrPad::SpacePad {
                        b' '
                    } else {
                        0
                    },
                },
                Datatype::VarString { .. } => Column::Text {
                    strings: match values {
                        Array::Str(a) => a.iter().collect(),
                        _ => return Err(bad(format!("compound field {:?}: text", leaf.path))),
                    },
                    size: None,
                    pad: 0,
                },
                dt => {
                    let (be, half) = match dt {
                        Datatype::Int { big_endian, .. } => (*big_endian, false),
                        Datatype::Float { size, big_endian } => (*big_endian, *size == 2),
                        Datatype::Enum { base, .. } => match base.as_ref() {
                            Datatype::Int { big_endian, .. } => (*big_endian, false),
                            _ => (false, false),
                        },
                        other => {
                            return Err(Error::Unsupported(format!(
                                "compound field {:?} of type {other:?}",
                                leaf.path
                            )));
                        }
                    };
                    let bytes = file_bytes(values, be, half)
                        .ok_or_else(|| bad(format!("compound field {:?}: numbers", leaf.path)))?;
                    let size = dt.size();
                    if bytes.len() != values.len().saturating_mul(size) {
                        return Err(bad(format!(
                            "compound field {:?}: values of type {:?} for a {size}-byte member",
                            leaf.path,
                            values.dtype()
                        )));
                    }
                    Column::Fixed { bytes, size }
                }
            };
            columns.push((leaf.offset, per, column));
        }
        let mut out = Bytes::default();
        for i in 0..n {
            let mut rec = Bytes {
                data: vec![0; *size],
                fixups: Vec::new(),
            };
            for (offset, per, column) in &columns {
                for j in 0..*per {
                    let k = i.saturating_mul(*per).saturating_add(j);
                    match column {
                        Column::Fixed { bytes, size } => {
                            let at = offset.saturating_add(j.saturating_mul(*size));
                            let src = bytes
                                .get(
                                    k.saturating_mul(*size)
                                        ..(k.saturating_add(1)).saturating_mul(*size),
                                )
                                .unwrap_or_default();
                            if let Some(dst) = rec.data.get_mut(at..at.saturating_add(*size)) {
                                dst.copy_from_slice(src);
                            }
                        }
                        Column::Text {
                            strings,
                            size: Some(width),
                            pad,
                        } => {
                            let s = strings.get(k).map_or(&[][..], |s| s.as_bytes());
                            if s.len() > *width {
                                return Err(bad(format!(
                                    "{:?} is longer than its {width}-byte compound field",
                                    String::from_utf8_lossy(s)
                                )));
                            }
                            let at = offset.saturating_add(j.saturating_mul(*width));
                            if let Some(dst) = rec.data.get_mut(at..at.saturating_add(*width)) {
                                dst.fill(*pad);
                                dst.get_mut(..s.len())
                                    .unwrap_or_default()
                                    .copy_from_slice(s);
                            }
                        }
                        Column::Text {
                            strings,
                            size: None,
                            ..
                        } => {
                            let s = strings.get(k).map_or("", |s| s.as_str());
                            let mut tmp = Bytes::default();
                            self.vlen(&mut tmp, s.len(), HeapObj::Bytes(s.as_bytes().to_vec()))?;
                            let at = offset.saturating_add(j.saturating_mul(tmp.data.len()));
                            if let Some(dst) =
                                rec.data.get_mut(at..at.saturating_add(tmp.data.len()))
                            {
                                dst.copy_from_slice(&tmp.data);
                            }
                            rec.fixups.extend(
                                tmp.fixups
                                    .into_iter()
                                    .map(|(p, t)| (p.saturating_add(at), t)),
                            );
                        }
                    }
                }
            }
            out.append(rec);
        }
        Ok(out)
    }

    /// Element bytes of a dataset: its values, or for a ragged dataset one heap object per
    /// element.
    fn dataset_bytes(&mut self, ds: &WDataset) -> Result<Bytes, Error> {
        if let Some(c) = &ds.compound {
            return self.compound_bytes(c);
        }
        let Some(counts) = &ds.ragged else {
            return self.element_bytes(&ds.data, ds.big_endian, ds.half);
        };
        let values = file_bytes(&ds.data, ds.big_endian, ds.half)
            .ok_or_else(|| Error::Unsupported(format!("writing {} data", ds.data.dtype())))?;
        let size = if ds.half && is_float(&ds.data) {
            2
        } else {
            esize_of(&ds.data)
        };
        let total: u64 = counts.iter().copied().fold(0u64, u64::saturating_add);
        if usize::try_from(total)
            .ok()
            .and_then(|t| t.checked_mul(size))
            != Some(values.len())
        {
            return Err(bad("ragged lengths do not add up to the values"));
        }
        let mut out = Bytes::default();
        let mut at = 0usize;
        for c in counts {
            let n = usize::try_from(*c).map_err(|_| too_big("ragged element"))?;
            let len = n
                .checked_mul(size)
                .ok_or_else(|| too_big("ragged element"))?;
            let end = at
                .checked_add(len)
                .ok_or_else(|| too_big("ragged element"))?;
            let bytes = values.get(at..end).unwrap_or_default().to_vec();
            self.vlen(&mut out, n, HeapObj::Bytes(bytes))?;
            at = end;
        }
        Ok(out)
    }

    /// Element bytes of a dataset (strings go to the global heap), big-endian if asked.
    fn element_bytes(
        &mut self,
        data: &Array,
        big_endian: bool,
        half: bool,
    ) -> Result<Bytes, Error> {
        let mut out = Bytes::default();
        if let Array::Str(a) = data {
            for s in a {
                self.vlen(&mut out, s.len(), HeapObj::Bytes(s.as_bytes().to_vec()))?;
            }
        } else {
            out.data = file_bytes(data, big_endian, half)
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
                (element_type(v, false, false, false)?, space, bytes)
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
                int_type(4, true, false),
                dataspace(Some(&[]), &[]),
                Bytes {
                    data: v.to_le_bytes().to_vec(),
                    fixups: Vec::new(),
                },
            ),
            AttrData::Half(v) => (
                float_type(2, false),
                dataspace((!v.is_empty()).then_some(&[v.len()][..]), &[]),
                Bytes {
                    data: v
                        .iter()
                        .flat_map(|x| crate::half::from_f32(*x).to_le_bytes())
                        .collect(),
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
            None => self.dataset_bytes(ds)?,
        };
        let filtered = filtered(ds);
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
                    let bytes = file_bytes(&slab, ds.big_endian, ds.half).ok_or_else(|| {
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
            if ds.fletcher32 {
                // Last in the pipeline, as HDF5 stores it: the checksum of the filtered chunk.
                let sum = crate::filters::fletcher32(&chunk.data);
                chunk.data.extend_from_slice(&sum.to_le_bytes());
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
        let base = match &ds.compound {
            Some(c) => datatype_message(&c.dtype)?,
            None => match &ds.enumeration {
                Some(t @ Datatype::Enum { base, .. }) => {
                    let own = element_type(&ds.data, ds.char_data, ds.big_endian, ds.half)?;
                    if datatype_message(base)? != own {
                        return Err(bad("an enumeration's base type must be the values' type"));
                    }
                    datatype_message(t)?
                }
                Some(_) => return Err(bad("an enumeration type must be an Enum")),
                None => element_type(&ds.data, ds.char_data, ds.big_endian, ds.half)?,
            },
        };
        let dtype = if ds.compound.is_some() {
            base
        } else if ds.ragged.is_some() {
            if ds.source.is_some() || matches!(ds.data, Array::Str(_)) {
                return Err(Error::Unsupported(
                    "a ragged dataset of strings, or from a lazy source".into(),
                ));
            }
            sequence_type(base)
        } else {
            base
        };
        let esize = if let Some(c) = &ds.compound {
            c.dtype.size()
        } else if ds.ragged.is_some() {
            16
        } else if ds.half && is_float(&ds.data) {
            2
        } else {
            esize_of(&ds.data)
        };
        let chunks = match (&ds.chunks, ds.unlimited.iter().any(|u| *u)) {
            (Some(c), _) if c.len() == rank && rank > 0 => {
                Some(c.iter().map(|x| (*x).max(1)).collect::<Vec<_>>())
            }
            (_, true) => Some(shape.iter().map(|d| (*d).clamp(1, 1024)).collect()),
            // Filters need chunks: without a chunk shape, one chunk per (capped) extent.
            _ if filtered(ds) && rank > 0 => {
                Some(shape.iter().map(|d| (*d).clamp(1, 1024)).collect())
            }
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
                // Given little-endian (as f32/f64 for a half-precision dataset); stored in the
                // dataset's type and byte order.
                let mut f = match (ds.half && is_float(&ds.data), f.len()) {
                    (true, 4) => f
                        .get(..4)
                        .and_then(|b| b.try_into().ok())
                        .map(|b| {
                            crate::half::from_f32(f32::from_le_bytes(b))
                                .to_le_bytes()
                                .to_vec()
                        })
                        .unwrap_or_default(),
                    (true, 8) => f
                        .get(..8)
                        .and_then(|b| b.try_into().ok())
                        .map(|b| {
                            crate::half::from_f64(f64::from_le_bytes(b))
                                .to_le_bytes()
                                .to_vec()
                        })
                        .unwrap_or_default(),
                    _ => f.clone(),
                };
                if ds.big_endian && !ds.char_data && !matches!(ds.data, Array::Str(_)) {
                    f.reverse();
                }
                fill.u32(u32_of(f.len(), "fill value")?);
                fill.raw(&f);
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
                    let addr =
                        self.contiguous_from(src.as_ref(), &shape, esize, ds.big_endian, ds.half)?;
                    layout.addr(Target::Abs(addr));
                } else {
                    let data = self.dataset_bytes(ds)?;
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
        if chunks.is_some() && filtered(ds) {
            let shuffled = ds.shuffle && !blosc(ds);
            let mut p = Bytes::default();
            p.u8(2);
            p.u8(u8::from(shuffled)
                .saturating_add(u8::from(ds.deflate.is_some()))
                .saturating_add(u8::from(ds.compression.is_some()))
                .saturating_add(u8::from(ds.fletcher32)));
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
            if ds.fletcher32 {
                p.u16(3);
                p.u16(0); // mandatory
                p.u16(0); // no client data
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
        big_endian: bool,
        half: bool,
    ) -> Result<u64, Error> {
        let read = |start: &[usize], count: &[usize]| -> Result<Vec<u8>, Error> {
            let t = src
                .read_slice(start, count)
                .map_err(|e| bad(format!("reading data to write: {e}")))?;
            file_bytes(&t, big_endian, half)
                .ok_or_else(|| Error::Unsupported(format!("writing {} data", t.dtype())))
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
        let mut order: u16 = 0;
        let mut attr_msgs = Vec::new();
        for (name, value) in attrs {
            attr_msgs.push((order, name.as_str(), self.attribute(name, value)?));
            order = order.saturating_add(1);
        }
        let mut ainfo = Bytes::default();
        ainfo.u8(0);
        ainfo.u8(1);
        ainfo.u16(u16_of(attrs.len(), "attribute count")?);
        // A header message holds at most 64 KiB: past that, every attribute goes to dense
        // storage (a fractal heap, indexed by name in a v2 B-tree).
        if attr_msgs
            .iter()
            .any(|(_, _, m)| m.data.len() > usize::from(u16::MAX))
        {
            let (heap, index) = self.dense_attributes(std::mem::take(&mut attr_msgs))?;
            ainfo.addr(Target::Block(heap));
            ainfo.addr(Target::Block(index));
        } else {
            ainfo.u64(UNDEF);
            ainfo.u64(UNDEF);
        }
        msgs.push((0x15, ainfo));
        let mut body = Bytes::default();
        for (kind, m) in msgs {
            body.u8(kind);
            body.u16(u16_of(m.data.len(), "header message")?);
            body.u8(0);
            body.u16(0); // creation order (not meaningful for non-attribute messages)
            body.append(m);
        }
        for (o, _, m) in attr_msgs {
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

    /// Dense attribute storage: a fractal heap with HDF5's parameters for attributes, the
    /// messages in it as huge objects (each in a block of its own, found through a v2 B-tree
    /// of type 1 by heap ID), and a v2 B-tree (type 8) indexing them by name hash. The heap has
    /// no managed blocks yet, so a library adding attributes later starts them as it would.
    /// Returns the blocks of the heap header and the name index header.
    fn dense_attributes(
        &mut self,
        attrs: Vec<(u16, &str, Bytes)>,
    ) -> Result<(usize, usize), Error> {
        let mut huge = Vec::with_capacity(attrs.len());
        let mut names = Vec::with_capacity(attrs.len());
        let mut total: u64 = 0;
        for (k, (order, name, m)) in attrs.into_iter().enumerate() {
            // Huge object IDs count from 1; the heap ID is the type byte and 7 bytes of it.
            let key = (k as u64).saturating_add(1);
            let len = m.data.len() as u64;
            total = total.saturating_add(len);
            let mut id = [0u8; 8];
            id[0] = 0x10;
            id[1..].copy_from_slice(key.to_le_bytes().get(..7).unwrap_or_default());
            names.push((lookup3(name.as_bytes(), 0), name, id, order));
            huge.push((self.block(m, false), len, key));
        }
        let n = names.len() as u64;
        // Huge objects: (address, length, ID) by ID.
        let mut leaf = Bytes::default();
        leaf.raw(b"BTLF");
        leaf.u8(0);
        leaf.u8(1);
        for (block, len, key) in &huge {
            leaf.addr(Target::Block(*block));
            leaf.u64(*len);
            leaf.u64(*key);
        }
        let huge_tree = self.btree2(leaf, 1, 24, huge.len())?;
        let mut h = Bytes::default();
        h.raw(b"FRHP");
        h.u8(0);
        h.u16(8); // heap ID length
        h.u16(0); // no I/O filters
        h.u8(0x02); // flags: direct blocks checksummed
        h.u32(4096); // largest managed object
        h.u64(n); // the last huge object ID
        h.addr(Target::Block(huge_tree));
        h.u64(0); // free space
        h.u64(UNDEF); // free space manager
        for _ in 0..4 {
            h.u64(0); // managed space, allocated, allocation iterator, objects
        }
        h.u64(total); // huge objects: size, count
        h.u64(n);
        h.u64(0); // tiny objects: size, count
        h.u64(0);
        h.u16(4); // table width
        h.u64(1024); // starting block size
        h.u64(65536); // largest direct block
        h.u16(40); // heap size bits
        h.u16(1); // starting rows of the root indirect block
        h.u64(UNDEF); // no root block
        h.u16(0);
        h.u32(0); // checksum
        let heap = self.block(h, true);
        // The name index: records by hash (then name).
        names.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
        let mut leaf = Bytes::default();
        leaf.raw(b"BTLF");
        leaf.u8(0);
        leaf.u8(8);
        for (hash, _, id, order) in &names {
            leaf.raw(id);
            leaf.u8(0); // message flags
            leaf.u32(u32::from(*order));
            leaf.u32(*hash);
        }
        let index = self.btree2(leaf, 8, 17, names.len())?;
        Ok((heap, index))
    }

    /// A v2 B-tree of one leaf (`leaf`: its signature, version, type and records, without the
    /// checksum), its node size the larger of HDF5's 512 bytes and what the leaf needs (a
    /// reader takes the size from the header). Returns the block of the header.
    fn btree2(
        &mut self,
        mut leaf: Bytes,
        kind: u8,
        record: u16,
        nrec: usize,
    ) -> Result<usize, Error> {
        let node_size = leaf.data.len().saturating_add(4).max(512);
        // The checksum covers the leaf up to the records; the node is padded after it.
        let at = leaf.data.len();
        leaf.u32(0);
        leaf.data.resize(node_size, 0);
        let leaf = self.block(leaf, false);
        if let Some(b) = self.blocks.get_mut(leaf) {
            b.checksum_at = Some(at);
        }
        let mut t = Bytes::default();
        t.raw(b"BTHD");
        t.u8(0);
        t.u8(kind);
        t.u32(u32_of(node_size, "B-tree node")?);
        t.u16(record);
        t.u16(0); // depth
        t.u8(100); // split percent
        t.u8(40); // merge percent
        t.addr(Target::Block(leaf));
        t.u16(u16_of(nrec, "B-tree records")?);
        t.u64(nrec as u64);
        t.u32(0); // checksum
        Ok(self.block(t, true))
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
                WNode::Datatype(dt) => {
                    let mut m = Bytes::default();
                    m.raw(&datatype_message(dt)?);
                    self.finish_object(id, vec![(0x03, m)], &[])?;
                }
            }
        }
        Ok(())
    }
}

fn is_float(d: &Array) -> bool {
    matches!(d, Array::F32(_) | Array::F64(_))
}

/// Whether chunks go through a filter pipeline.
fn filtered(ds: &WDataset) -> bool {
    ds.deflate.is_some() || ds.shuffle || ds.compression.is_some() || ds.fletcher32
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
        if let Some(n) = b.checksum_at {
            let sum = lookup3(data.get(..n).unwrap_or_default(), 0);
            if let Some(slot) = data.get_mut(n..n.saturating_add(4)) {
                slot.copy_from_slice(&sum.to_le_bytes());
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
