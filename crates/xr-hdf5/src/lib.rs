//! Pure-Rust HDF5 for xarray-rs.
//!
//! Reads every structure current HDF5 libraries write: superblocks v0–v3, object headers v1
//! and v2, old-style (symbol table), compact and dense (fractal heap + v2 B-tree) groups,
//! compact and dense attributes, compact/contiguous/chunked layouts with every chunk index
//! (v1 B-tree, single chunk, implicit, fixed array, extensible array, v2 B-tree), and the
//! deflate, shuffle and fletcher32 filters.
//!
//! Files are untrusted input: every read is bounds-checked, sizes use checked arithmetic, tree
//! walks are depth-limited, and loading refuses datasets larger than [`MAX_LOAD_BYTES`].
//!
//! ```no_run
//! # fn main() -> Result<(), xr_io::IoError> {
//! let file = xr_hdf5::File::open("data.h5")?;
//! for (name, node) in file.root()?.members()? {
//!     if let xr_hdf5::Node::Dataset(ds) = node {
//!         println!("{name}: {:?} {:?}", ds.shape(), ds.read()?.dtype());
//!     }
//! }
//! # Ok(())
//! # }
//! ```

mod backend;
mod checksum;
mod ck;
mod cursor;
mod dtype;
mod filters;
mod format;
mod storage;
mod writer;

use std::path::Path;
use std::sync::Arc;

use xr_core::{Owned, TypedArray};
use xr_io::IoError;

pub use backend::Hdf5;
pub use checksum::lookup3;
use ck::Ck;
use cursor::{Cursor, Sizes};
pub use dtype::{Datatype, StrPad};
use format::{ChunkIndex, Filter, Layout, Message, Space, kind};
use storage::{ChunkRef, Element, FractalHeap};
pub use writer::{AttrData, WDataset, WGroup, WNode, write_file};

/// Default limit on the decoded size of one dataset read ([`File::with_load_limit`]).
/// Chunked data can legitimately decode to more than the file size (unwritten chunks become
/// fill values), so this cannot simply be the file size.
pub const MAX_LOAD_BYTES: usize = 1 << 34;

const NAME: &str = "hdf5";

pub(crate) fn bad(message: impl Into<String>) -> IoError {
    IoError::Backend {
        backend: NAME,
        message: message.into(),
    }
}

/// Whether `bytes` start with the HDF5 signature (at offset 0).
#[must_use]
pub fn is_hdf5(bytes: &[u8]) -> bool {
    bytes.starts_with(format::SIGNATURE)
}

/// An open HDF5 file (held in memory). Cloning is cheap.
#[derive(Debug, Clone)]
pub struct File {
    buf: Arc<Vec<u8>>,
    pub(crate) sizes: Sizes,
    base: u64,
    root: u64,
    superblock_version: u8,
    load_limit: usize,
}

impl File {
    /// Open a file.
    ///
    /// # Errors
    /// I/O errors, or [`IoError::Backend`] if it is not a readable HDF5 file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        let buf = std::fs::read(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        Self::from_bytes(buf)
    }

    /// Parse a file held in memory.
    ///
    /// # Errors
    /// [`IoError::Backend`] if it is not a readable HDF5 file.
    pub fn from_bytes(buf: Vec<u8>) -> Result<Self, IoError> {
        let sb = format::superblock(&buf)?;
        Ok(Self {
            buf: Arc::new(buf),
            sizes: sb.sizes,
            base: sb.base,
            root: sb.root,
            superblock_version: sb.version,
            load_limit: MAX_LOAD_BYTES,
        })
    }

    /// Refuse to decode datasets larger than `bytes` (default [`MAX_LOAD_BYTES`]). Use a small
    /// limit for untrusted files: a corrupt header can claim huge dimensions.
    #[must_use]
    pub fn with_load_limit(mut self, bytes: usize) -> Self {
        self.load_limit = bytes;
        self
    }

    /// The superblock version (0–3).
    #[must_use]
    pub fn superblock_version(&self) -> u8 {
        self.superblock_version
    }

    pub(crate) fn cursor(&self, addr: u64) -> Result<Cursor<'_>, IoError> {
        let at = addr
            .checked_add(self.base)
            .and_then(|a| usize::try_from(a).ok())
            .ok_or_else(|| bad("bad address"))?;
        if at > self.buf.len() {
            return Err(bad(format!("address {addr} is past the end of the file")));
        }
        Ok(Cursor::new(&self.buf, at, self.sizes))
    }

    /// The root group.
    ///
    /// # Errors
    /// If the root object header is unreadable.
    pub fn root(&self) -> Result<Group, IoError> {
        Ok(Group {
            file: self.clone(),
            addr: self.root,
            path: "/".into(),
        })
    }

    fn node(&self, addr: u64, path: String) -> Result<Node, IoError> {
        let msgs = format::object_header(self, addr)?;
        let has = |k: u16| msgs.iter().any(|m| m.kind == k);
        Ok(if has(kind::LAYOUT) {
            Node::Dataset(Box::new(Dataset::from_messages(self.clone(), path, &msgs)?))
        } else if has(kind::DATATYPE)
            && !has(kind::LINK)
            && !has(kind::LINK_INFO)
            && !has(kind::SYMBOL_TABLE)
        {
            Node::Other
        } else {
            Node::Group(Group {
                file: self.clone(),
                addr,
                path,
            })
        })
    }
}

/// A member of a group.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Node {
    /// A group.
    Group(Group),
    /// A dataset.
    Dataset(Box<Dataset>),
    /// Something else (a committed datatype).
    Other,
}

/// A group.
#[derive(Debug, Clone)]
pub struct Group {
    file: File,
    addr: u64,
    path: String,
}

fn child_path(parent: &str, name: &str) -> String {
    if parent.ends_with('/') {
        format!("{parent}{name}")
    } else {
        format!("{parent}/{name}")
    }
}

impl Group {
    /// The group's path (`/` for the root).
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The file.
    #[must_use]
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Members in creation order when the file tracks it, otherwise in name order.
    ///
    /// # Errors
    /// If the group's structures are corrupt.
    pub fn members(&self) -> Result<Vec<(String, Node)>, IoError> {
        let msgs = format::object_header(&self.file, self.addr)?;
        let s = self.file.sizes;
        let mut links: Vec<(String, Option<u64>, Option<u64>)> = Vec::new();
        for m in &msgs {
            match m.kind {
                kind::SYMBOL_TABLE => {
                    let (bt, heap) = format::symbol_table(&m.data, s)?;
                    let seg = storage::local_heap(&self.file, heap)?;
                    for (name, addr) in storage::group_btree(&self.file, bt, &seg)? {
                        links.push((name, Some(addr), None));
                    }
                }
                kind::LINK => {
                    let l = format::link(&m.data, s)?;
                    links.push((l.name, l.target, l.order));
                }
                kind::LINK_INFO => {
                    let info = format::link_info(&m.data, s)?;
                    if let (Some(heap), Some(index)) = (info.heap, info.name_index) {
                        let heap = FractalHeap::open(&self.file, heap)?;
                        let (_, records) = storage::btree2(&self.file, index)?;
                        for r in records {
                            let id = r.get(4..).ok_or_else(|| bad("bad link record"))?;
                            let l = format::link(&heap.get(&self.file, id)?, s)?;
                            links.push((l.name, l.target, l.order));
                        }
                    }
                }
                _ => {}
            }
        }
        if links.iter().all(|(_, _, o)| o.is_some()) {
            links.sort_by_key(|(_, _, o)| *o);
        } else {
            links.sort_by(|a, b| a.0.cmp(&b.0));
        }
        links
            .into_iter()
            .filter_map(|(name, addr, _)| addr.map(|a| (name, a)))
            .map(|(name, addr)| {
                let path = child_path(&self.path, &name);
                Ok((name, self.file.node(addr, path)?))
            })
            .collect()
    }

    /// The member at a relative `/`-separated path.
    ///
    /// # Errors
    /// [`IoError::NoSuchGroup`] if it does not exist.
    pub fn get(&self, path: &str) -> Result<Node, IoError> {
        let mut node = Node::Group(self.clone());
        for part in path.split('/').filter(|p| !p.is_empty()) {
            let Node::Group(g) = node else {
                return Err(IoError::NoSuchGroup(path.to_owned()));
            };
            node = g
                .members()?
                .into_iter()
                .find(|(n, _)| n == part)
                .map(|(_, n)| n)
                .ok_or_else(|| IoError::NoSuchGroup(path.to_owned()))?;
        }
        Ok(node)
    }

    /// Attributes, in creation order.
    ///
    /// # Errors
    /// If attribute storage is corrupt.
    pub fn attrs(&self) -> Result<Vec<Attribute>, IoError> {
        attributes(&self.file, &format::object_header(&self.file, self.addr)?)
    }
}

/// An attribute.
#[derive(Debug, Clone)]
pub struct Attribute {
    file: File,
    /// Name.
    pub name: String,
    /// Element type.
    pub dtype: Datatype,
    /// Shape (`[]` for scalars, `None` for the null dataspace).
    pub shape: Option<Vec<usize>>,
    data: Vec<u8>,
}

impl Attribute {
    /// The value.
    ///
    /// # Errors
    /// [`IoError::Unsupported`] for datatypes that cannot be converted.
    pub fn value(&self) -> Result<TypedArray<Owned>, IoError> {
        let shape = self.shape.clone().unwrap_or_else(|| vec![0]);
        let n = shape
            .iter()
            .try_fold(Ck::from(1usize), |a, d| Some(a * *d))
            .and_then(Ck::usize)
            .ok_or_else(|| bad("attribute too large"))?;
        let bytes = (Ck::from(n) * self.dtype.element_size(self.file.sizes.offset))
            .usize()
            .ok_or_else(|| bad("attribute too large"))?;
        let raw = self
            .data
            .get(..bytes)
            .ok_or_else(|| bad(format!("attribute {:?} is truncated", self.name)))?;
        dtype::decode(&self.file, &self.dtype, raw, &shape)
    }
}

fn attributes(file: &File, msgs: &[Message]) -> Result<Vec<Attribute>, IoError> {
    let mut out = Vec::new();
    let to_attr = |a: format::AttributeMsg| -> Result<Attribute, IoError> {
        let shape = a
            .dims
            .map(|d| {
                d.into_iter()
                    .map(|x| usize::try_from(x).map_err(|_| bad("attribute too large")))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        Ok(Attribute {
            file: file.clone(),
            name: a.name,
            dtype: a.dtype,
            shape,
            data: a.data,
        })
    };
    for m in msgs {
        match m.kind {
            kind::ATTRIBUTE => {
                if m.flags & 2 != 0 {
                    return Err(IoError::Unsupported("shared attribute messages".into()));
                }
                out.push(to_attr(format::attribute(file, &m.data)?)?);
            }
            kind::ATTRIBUTE_INFO => {
                let info = format::attribute_info(&m.data, file.sizes)?;
                if let (Some(heap), Some(index)) = (info.heap, info.name_index) {
                    let heap = FractalHeap::open(file, heap)?;
                    let (_, records) = storage::btree2(file, index)?;
                    let mut dense = Vec::new();
                    for r in records {
                        // Type 8 record: heap ID (8), flags (1), creation order (4), hash (4).
                        let id = r.get(..8).ok_or_else(|| bad("bad attribute record"))?;
                        let order = r
                            .get(9..13)
                            .and_then(|b| b.try_into().ok())
                            .map_or(0, u32::from_le_bytes);
                        dense.push((
                            order,
                            to_attr(format::attribute(file, &heap.get(file, id)?)?)?,
                        ));
                    }
                    dense.sort_by_key(|(o, _)| *o);
                    out.extend(dense.into_iter().map(|(_, a)| a));
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// A dataset.
#[derive(Debug, Clone)]
pub struct Dataset {
    file: File,
    path: String,
    space: Option<Space>,
    dtype: Datatype,
    layout: Layout,
    filters: Vec<Filter>,
    fill: Option<Vec<u8>>,
    msgs: Vec<Message>,
}

impl Dataset {
    fn from_messages(file: File, path: String, msgs: &[Message]) -> Result<Self, IoError> {
        let s = file.sizes;
        let find = |k: u16| msgs.iter().find(|m| m.kind == k);
        let dt_msg = find(kind::DATATYPE)
            .ok_or_else(|| bad(format!("{path}: dataset without a datatype")))?;
        let dtype = if dt_msg.flags & 2 != 0 {
            dtype::shared_datatype(&file, &dt_msg.data)?
        } else {
            Datatype::parse(&dt_msg.data)?
        };
        let space = format::dataspace(
            &find(kind::DATASPACE)
                .ok_or_else(|| bad(format!("{path}: dataset without a dataspace")))?
                .data,
            s,
        )?;
        let layout = format::layout(&find(kind::LAYOUT).ok_or_else(|| bad("no layout"))?.data, s)?;
        let filters = find(kind::FILTERS)
            .map(|m| format::filters(&m.data, s))
            .transpose()?
            .unwrap_or_default();
        let fill = match find(kind::FILL).or_else(|| find(kind::FILL_OLD)) {
            Some(m) => format::fill_value(m.kind, &m.data, s)?,
            None => None,
        };
        Ok(Self {
            file,
            path,
            space,
            dtype,
            layout,
            filters,
            fill,
            msgs: msgs.to_vec(),
        })
    }

    /// The dataset's path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Element type.
    #[must_use]
    pub fn dtype(&self) -> &Datatype {
        &self.dtype
    }

    /// Current shape (`[]` for scalars, `[0]` for the null dataspace).
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.space.as_ref().map_or_else(
            || vec![0],
            |s| {
                s.dims
                    .iter()
                    .map(|d| usize::try_from(*d).unwrap_or(usize::MAX))
                    .collect()
            },
        )
    }

    /// Maximum shape (`None` per unlimited dimension).
    #[must_use]
    pub fn max_shape(&self) -> Vec<Option<usize>> {
        self.space.as_ref().map_or_else(Vec::new, |s| {
            s.max
                .iter()
                .map(|m| {
                    if *m == u64::MAX {
                        None
                    } else {
                        usize::try_from(*m).ok()
                    }
                })
                .collect()
        })
    }

    /// Chunk shape, if chunked.
    #[must_use]
    pub fn chunks(&self) -> Option<Vec<usize>> {
        match &self.layout {
            Layout::Chunked { chunk, .. } => Some(
                chunk
                    .iter()
                    .take(self.shape().len())
                    .map(|c| usize::try_from(*c).unwrap_or(usize::MAX))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// `(deflate level, shuffle)` from the filter pipeline.
    #[must_use]
    pub fn compression(&self) -> (Option<u8>, bool) {
        let deflate = self
            .filters
            .iter()
            .find(|f| f.id == filters::DEFLATE)
            .map(|f| {
                f.client
                    .first()
                    .copied()
                    .map_or(6, |l| u8::try_from(l).unwrap_or(9))
            });
        (
            deflate,
            self.filters.iter().any(|f| f.id == filters::SHUFFLE),
        )
    }

    /// Attributes, in creation order.
    ///
    /// # Errors
    /// If attribute storage is corrupt.
    pub fn attrs(&self) -> Result<Vec<Attribute>, IoError> {
        attributes(&self.file, &self.msgs)
    }

    /// Read the whole dataset.
    ///
    /// # Errors
    /// [`IoError::Unsupported`] for unsupported types, filters or layouts; [`IoError::Backend`]
    /// for corrupt data or datasets over [`MAX_LOAD_BYTES`].
    pub fn read(&self) -> Result<TypedArray<Owned>, IoError> {
        let shape = self.shape();
        let esize = self.dtype.element_size(self.file.sizes.offset);
        if esize == 0 {
            return Err(IoError::Unsupported(format!(
                "{}: zero-sized elements",
                self.path
            )));
        }
        let n = shape
            .iter()
            .try_fold(Ck::from(1usize), |a, d| Some(a * *d))
            .and_then(Ck::usize)
            .ok_or_else(|| bad("dataset too large"))?;
        let total = (Ck::from(n) * esize)
            .usize()
            .filter(|t| *t <= self.file.load_limit)
            .ok_or_else(|| bad(format!("{}: too large to load", self.path)))?;
        let fill: Vec<u8> = match &self.fill {
            Some(f) if f.len() == esize => f.clone(),
            _ => vec![0; esize],
        };
        let bytes = match &self.layout {
            Layout::Compact(data) => data
                .get(..total)
                .ok_or_else(|| bad(format!("{}: compact data truncated", self.path)))?
                .to_vec(),
            Layout::Contiguous { addr: None, .. } => {
                fill.iter().copied().cycle().take(total).collect()
            }
            Layout::Contiguous {
                addr: Some(a),
                size,
            } => {
                if (total as u64) > *size {
                    return Err(bad(format!(
                        "{}: contiguous storage smaller than the data",
                        self.path
                    )));
                }
                if total == 0 {
                    Vec::new()
                } else {
                    self.file.cursor(*a)?.bytes(total)?.to_vec()
                }
            }
            Layout::Chunked { addr, chunk, index } => {
                let mut out: Vec<u8> = fill.iter().copied().cycle().take(total).collect();
                if let Some(addr) = addr {
                    let cdims: Vec<u64> = chunk.iter().take(shape.len()).copied().collect();
                    for c in self.chunk_list(*addr, &cdims, index, &shape, esize)? {
                        self.place_chunk(&c, &cdims, &shape, esize, &mut out)?;
                    }
                }
                out
            }
        };
        dtype::decode(&self.file, &self.dtype, &bytes, &shape)
    }

    fn chunk_list(
        &self,
        addr: u64,
        cdims: &[u64],
        index: &ChunkIndex,
        shape: &[usize],
        esize: usize,
    ) -> Result<Vec<ChunkRef>, IoError> {
        let rank = shape.len();
        let chunk_bytes = cdims
            .iter()
            .try_fold(Ck::from(esize), |a, d| Some(a * *d))
            .and_then(|x| x.to::<u64>())
            .ok_or_else(|| bad("chunk too large"))?;
        // Chunks per dimension, over the current and the maximum extent.
        let count = |extent: &[u64]| -> Vec<u64> {
            extent
                .iter()
                .zip(cdims)
                .map(|(d, c)| if *c == 0 { 0 } else { d.div_ceil(*c) })
                .collect()
        };
        let cur: Vec<u64> = shape.iter().map(|d| *d as u64).collect();
        let max: Vec<u64> = self.space.as_ref().map_or_else(
            || cur.clone(),
            |s| {
                s.max
                    .iter()
                    .zip(&cur)
                    .map(|(m, d)| if *m == u64::MAX { *d } else { *m })
                    .collect()
            },
        );
        let from_elements =
            |elems: Vec<Element>, coords: &dyn Fn(u64) -> Option<Vec<u64>>| -> Vec<ChunkRef> {
                elems
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, e)| {
                        let a = e.addr?;
                        let scaled = coords(i as u64)?;
                        Some(ChunkRef {
                            offset: scaled
                                .iter()
                                .zip(cdims)
                                .map(|(s, c)| s.saturating_mul(*c))
                                .collect(),
                            addr: a,
                            size: e.size.unwrap_or(chunk_bytes),
                            mask: e.mask,
                        })
                    })
                    .collect()
            };
        Ok(match index {
            ChunkIndex::BTreeV1 => storage::chunk_btree(&self.file, addr, rank)?,
            ChunkIndex::Single {
                filtered_size,
                mask,
            } => {
                vec![ChunkRef {
                    offset: vec![0; rank],
                    addr,
                    size: filtered_size.unwrap_or(chunk_bytes),
                    mask: *mask,
                }]
            }
            ChunkIndex::Implicit => {
                let nc = count(&max);
                let total: u64 = nc.iter().product();
                (0..total)
                    .filter_map(|i| {
                        Some(ChunkRef {
                            offset: unravel(i, &nc)?
                                .iter()
                                .zip(cdims)
                                .map(|(s, c)| s.saturating_mul(*c))
                                .collect(),
                            addr: addr.checked_add(i.checked_mul(chunk_bytes)?)?,
                            size: chunk_bytes,
                            mask: 0,
                        })
                    })
                    .collect()
            }
            ChunkIndex::FixedArray => {
                let nc = count(&max);
                from_elements(storage::fixed_array(&self.file, addr)?, &|i| {
                    unravel(i, &nc)
                })
            }
            ChunkIndex::ExtensibleArray => {
                // The single unlimited dimension is "swizzled" to be the slowest-varying.
                let unlim = self
                    .space
                    .as_ref()
                    .and_then(|s| s.max.iter().position(|m| *m == u64::MAX))
                    .unwrap_or(0);
                let nc_max = count(&max);
                let mut sw: Vec<u64> = nc_max.clone();
                let u = sw.remove(unlim.min(sw.len().saturating_sub(1)));
                sw.insert(0, u);
                let cur_chunks = count(&cur);
                let needed: u64 = {
                    // Highest linear index among current chunks, plus one.
                    let mut last = cur_chunks
                        .iter()
                        .map(|c| c.saturating_sub(1))
                        .collect::<Vec<_>>();
                    let lu = last.remove(unlim.min(last.len().saturating_sub(1)));
                    last.insert(0, lu);
                    ravel(&last, &sw).map_or(0, |x| x.saturating_add(1))
                };
                let elems = storage::extensible_array(&self.file, addr, needed)?;
                from_elements(elems, &|i| {
                    let mut c = unravel(i, &sw)?;
                    let first = c.remove(0);
                    c.insert(unlim.min(c.len()), first);
                    Some(c)
                })
            }
            ChunkIndex::BTreeV2 => {
                let (typ, records) = storage::btree2(&self.file, addr)?;
                let so = usize::from(self.file.sizes.offset);
                let mut out = Vec::new();
                for r in records {
                    let mut c = Cursor::new(&r, 0, self.file.sizes);
                    let a = c.addr()?;
                    let (size, mask) = if typ == 11 {
                        let fixed = (Ck::from(so) + 4usize + Ck::from(8usize) * rank)
                            .usize()
                            .ok_or_else(|| bad("bad chunk record"))?;
                        let size_len = r
                            .len()
                            .checked_sub(fixed)
                            .ok_or_else(|| bad("bad chunk record"))?;
                        (c.uvar(size_len)?, c.u32()?)
                    } else {
                        (chunk_bytes, 0)
                    };
                    let scaled = (0..rank).map(|_| c.u64()).collect::<Result<Vec<_>, _>>()?;
                    if let Some(a) = a {
                        out.push(ChunkRef {
                            offset: scaled
                                .iter()
                                .zip(cdims)
                                .map(|(s, c)| s.saturating_mul(*c))
                                .collect(),
                            addr: a,
                            size,
                            mask,
                        });
                    }
                }
                out
            }
        })
    }

    fn place_chunk(
        &self,
        c: &ChunkRef,
        cdims: &[u64],
        shape: &[usize],
        esize: usize,
        out: &mut [u8],
    ) -> Result<(), IoError> {
        let rank = shape.len();
        let cd: Vec<usize> = cdims
            .iter()
            .map(|d| usize::try_from(*d).unwrap_or(usize::MAX))
            .collect();
        let chunk_elems = cd
            .iter()
            .try_fold(Ck::from(1usize), |a, d| Some(a * *d))
            .and_then(Ck::usize)
            .ok_or_else(|| bad("chunk too large"))?;
        let expected = (Ck::from(chunk_elems) * esize)
            .usize()
            .ok_or_else(|| bad("chunk too large"))?;
        let stored = usize::try_from(c.size).map_err(|_| bad("chunk too large"))?;
        let raw = self.file.cursor(c.addr)?.bytes(stored)?.to_vec();
        let data = if self.filters.is_empty() {
            raw
        } else {
            filters::decode(&self.filters, c.mask, raw, expected)?
        };
        if data.len() < expected {
            return Err(bad(format!(
                "{}: chunk at {:?} is short",
                self.path, c.offset
            )));
        }
        let off: Vec<usize> = c
            .offset
            .iter()
            .map(|o| usize::try_from(*o).unwrap_or(usize::MAX))
            .collect();
        if rank == 0 {
            if let (Some(dst), Some(src)) = (out.get_mut(..esize), data.get(..esize)) {
                dst.copy_from_slice(src);
            }
            return Ok(());
        }
        let last = rank.saturating_sub(1);
        let (Some(&clast), Some(&dlast), Some(&olast)) =
            (cd.get(last), shape.get(last), off.get(last))
        else {
            return Ok(());
        };
        if olast >= dlast {
            return Ok(());
        }
        let row = clast.min(dlast.saturating_sub(olast));
        // Iterate over every row (all indices but the last) of the chunk.
        let outer: Vec<usize> = cd.iter().take(last).copied().collect();
        let nrows: usize = outer
            .iter()
            .try_fold(1usize, |a, d| a.checked_mul(*d))
            .ok_or_else(|| bad("chunk too large"))?;
        for r in 0..nrows {
            let idx = unravel(
                r as u64,
                &outer.iter().map(|x| *x as u64).collect::<Vec<_>>(),
            )
            .unwrap_or_default();
            let mut inside = true;
            let mut dst = Ck::from(0usize);
            let mut src = Ck::from(0usize);
            for (k, i) in idx.iter().enumerate() {
                let i = usize::try_from(*i).unwrap_or(usize::MAX);
                let g = off.get(k).copied().unwrap_or(0).saturating_add(i);
                let dim = shape.get(k).copied().unwrap_or(0);
                if g >= dim {
                    inside = false;
                    break;
                }
                dst = dst * dim + g;
                src = src * cd.get(k).copied().unwrap_or(0) + i;
            }
            if !inside {
                continue;
            }
            let dst = ((dst * dlast + olast) * esize)
                .usize()
                .ok_or_else(|| bad("offset overflow"))?;
            let src = ((src * clast) * esize)
                .usize()
                .ok_or_else(|| bad("offset overflow"))?;
            let len = (Ck::from(row) * esize)
                .usize()
                .ok_or_else(|| bad("offset overflow"))?;
            let (Some(d), Some(s)) = (
                out.get_mut(dst..dst.saturating_add(len)),
                data.get(src..src.saturating_add(len)),
            ) else {
                return Err(bad("chunk outside the dataset"));
            };
            d.copy_from_slice(s);
        }
        Ok(())
    }
}

/// Row-major multi-index of linear index `i` over `dims` (`None` if out of range).
fn unravel(mut i: u64, dims: &[u64]) -> Option<Vec<u64>> {
    let mut out = vec![0; dims.len()];
    for (slot, d) in out.iter_mut().zip(dims).rev() {
        *slot = i.checked_rem(*d)?;
        i = i.checked_div(*d)?;
    }
    (i == 0).then_some(out)
}

/// Inverse of [`unravel`].
fn ravel(idx: &[u64], dims: &[u64]) -> Option<u64> {
    idx.iter()
        .zip(dims)
        .try_fold(0u64, |acc, (i, d)| acc.checked_mul(*d)?.checked_add(*i))
}
