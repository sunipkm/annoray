//! A pure-Rust HDF5 reader and writer.
//!
//! Reads every structure current HDF5 libraries write: superblocks v0–v3, object headers v1
//! and v2, old-style (symbol table), compact and dense (fractal heap + v2 B-tree) groups,
//! compact and dense attributes, compact/contiguous/chunked layouts with every chunk index
//! (v1 B-tree, single chunk, implicit, fixed array, extensible array, v2 B-tree), and the
//! deflate, shuffle and fletcher32 filters. Compressed chunks are also read for LZF, N-bit,
//! scale-offset, bzip2, Blosc and Bitshuffle, and (with the `zstd` and `lz4` features)
//! Zstandard and LZ4. [`write_file`] writes a tree of groups and datasets.
//!
//! Values are read as an [`Array`]: an `ndarray` array of the element type the file stores.
//!
//! Files are untrusted input: every read is bounds-checked, sizes use checked arithmetic, tree
//! walks are depth-limited, and loading refuses datasets larger than [`MAX_LOAD_BYTES`].
//!
//! ```no_run
//! # fn main() -> Result<(), annoray_hdf5::Error> {
//! let file = annoray_hdf5::File::open("data.h5")?;
//! for (name, node) in file.root()?.members()? {
//!     if let annoray_hdf5::Node::Dataset(ds) = node {
//!         println!("{name}: {:?} {:?}", ds.shape(), ds.read()?.dtype());
//!     }
//! }
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]

// The README's examples run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod array;
mod bytes;
mod checksum;
mod ck;
mod codecs;
mod compression;
mod cursor;
mod dtype;
mod edit;
mod error;
mod filters;
mod format;
pub mod half;
mod storage;
mod virtual_ds;
mod writer;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub use array::{Array, ValueType};
pub use bytes::{FileBytes, map_file, read_file};
pub use checksum::lookup3;
use ck::Ck;
pub use compression::{BloscCodec, BloscShuffle, Compression};
use cursor::{Cursor, Sizes};
pub use dtype::{Datatype, Member, StrPad, compound_fields};
pub use edit::Editor;
pub(crate) use error::bad;
pub use error::{Error, Result};
use format::{ChunkIndex, Filter, Layout, Message, Space, kind};
use storage::{ChunkRef, Element, FractalHeap};
pub use writer::{AttrData, Source, WCompound, WDataset, WGroup, WNode, write_file, write_to};

/// Default limit on the decoded size of one dataset read ([`File::with_load_limit`]).
/// Chunked data can legitimately decode to more than the file size (unwritten chunks become
/// fill values), so this cannot simply be the file size.
pub const MAX_LOAD_BYTES: usize = 1 << 34;

static BATCH_BYTES: AtomicUsize = AtomicUsize::new(64 << 20);

/// Limit the memory that decoding or encoding chunks in parallel holds at once to about
/// `bytes` (at least one chunk is always in flight). The default is 64 MiB.
pub fn set_batch_bytes(bytes: usize) {
    BATCH_BYTES.store(bytes.max(1), Ordering::Relaxed);
}

/// The limit set by [`set_batch_bytes`].
#[must_use]
pub fn batch_bytes() -> usize {
    BATCH_BYTES.load(Ordering::Relaxed)
}

/// Whether `bytes` start with the HDF5 signature (at offset 0).
#[must_use]
pub fn is_hdf5(bytes: &[u8]) -> bool {
    bytes.starts_with(format::SIGNATURE)
}

/// An open HDF5 file (held in memory). Cloning is cheap.
#[derive(Debug, Clone)]
pub struct File {
    buf: Arc<FileBytes>,
    /// The directory of the file (for external links), when it was opened from a path.
    dir: Option<Arc<std::path::PathBuf>>,
    pub(crate) sizes: Sizes,
    base: u64,
    root: u64,
    superblock_version: u8,
    load_limit: usize,
    pub(crate) superblock: format::Superblock,
}

impl File {
    /// Open a file by memory-mapping it: only the metadata and the data actually read are
    /// paged in, so opening large files is cheap. See [`map_file`] for the one caveat
    /// (the file must not be modified while open) and [`File::open_in_memory`] to avoid it.
    ///
    /// # Errors
    /// [`Error::Io`], or [`Error::Invalid`] if it is not a readable HDF5 file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self::from_file_bytes(map_file(path.as_ref())?)?.at(path.as_ref()))
    }

    /// Open a file by reading it entirely into memory.
    ///
    /// # Errors
    /// As for [`File::open`].
    pub fn open_in_memory(path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self::from_file_bytes(read_file(path.as_ref())?)?.at(path.as_ref()))
    }

    /// Remember where the file is: external links are resolved from its directory.
    fn at(mut self, path: &Path) -> Self {
        self.dir = path.parent().map(|d| Arc::new(d.to_path_buf()));
        self
    }

    /// Parse a file held in memory.
    ///
    /// # Errors
    /// [`Error::Invalid`] if it is not a readable HDF5 file.
    pub fn from_bytes(buf: Vec<u8>) -> Result<Self, Error> {
        Self::from_file_bytes(FileBytes::Owned(buf))
    }

    /// Parse mapped or in-memory file bytes.
    ///
    /// # Errors
    /// [`Error::Invalid`] if it is not a readable HDF5 file.
    pub fn from_file_bytes(buf: FileBytes) -> Result<Self, Error> {
        let sb = format::superblock(&buf)?;
        Ok(Self {
            buf: Arc::new(buf),
            dir: None,
            sizes: sb.sizes,
            base: sb.base,
            root: sb.root,
            superblock_version: sb.version,
            load_limit: MAX_LOAD_BYTES,
            superblock: sb,
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

    /// Bytes from `addr` to the end of the file (zero past it).
    pub(crate) fn bytes_after(&self, addr: u64) -> u64 {
        let len = u64::try_from(self.buf.len()).unwrap_or(u64::MAX);
        len.saturating_sub(addr.saturating_add(self.base))
    }

    /// Page in `len` bytes at `addr` ahead of a large read (best effort).
    pub(crate) fn prefetch(&self, addr: u64, len: usize) {
        if let Some(at) = addr
            .checked_add(self.base)
            .and_then(|a| usize::try_from(a).ok())
            .filter(|a| a.checked_add(len).is_some_and(|e| e <= self.buf.len()))
        {
            self.buf.prefetch(at, len);
        }
    }

    pub(crate) fn cursor(&self, addr: u64) -> Result<Cursor<'_>, Error> {
        let at = addr
            .checked_add(self.base)
            .and_then(|a| usize::try_from(a).ok())
            .ok_or_else(|| bad("bad address"))?;
        if at > self.buf.len() {
            return Err(bad(format!("address {addr} is past the end of the file")));
        }
        let bytes: &[u8] = &self.buf;
        Ok(Cursor::new(bytes, at, self.sizes))
    }

    /// The root group.
    ///
    /// # Errors
    /// If the root object header is unreadable.
    pub fn root(&self) -> Result<Group, Error> {
        Ok(Group {
            file: self.clone(),
            addr: self.root,
            path: "/".into(),
        })
    }

    fn node(&self, addr: u64, path: String) -> Result<Node, Error> {
        let msgs = format::object_header(self, addr)?;
        let has = |k: u16| msgs.iter().any(|m| m.kind == k);
        Ok(if has(kind::LAYOUT) {
            Node::Dataset(Box::new(Dataset::from_messages(
                self.clone(),
                addr,
                path,
                &msgs,
            )?))
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

/// How many soft or external links a resolution may pass through (a cycle stops here).
const MAX_LINK_DEPTH: usize = 16;

fn child_path(parent: &str, name: &str) -> String {
    if parent.ends_with('/') {
        format!("{parent}{name}")
    } else {
        format!("{parent}/{name}")
    }
}

impl Group {
    /// The address of the group's object header (what object references point to).
    #[must_use]
    pub fn address(&self) -> u64 {
        self.addr
    }

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

    /// The committed (named) datatypes linked from the group (by hard link): their names,
    /// object header addresses (which [`Dataset::committed_type`] refers to) and types.
    ///
    /// # Errors
    /// If the group's structures are corrupt.
    pub fn datatypes(&self) -> Result<Vec<(String, u64, Datatype)>, Error> {
        let mut out = Vec::new();
        for (name, target) in self.links()? {
            let format::Target::Hard(addr) = target else {
                continue;
            };
            let msgs = format::object_header(&self.file, addr)?;
            let has = |k: u16| msgs.iter().any(|m| m.kind == k);
            if has(kind::LAYOUT)
                || has(kind::LINK)
                || has(kind::LINK_INFO)
                || has(kind::SYMBOL_TABLE)
            {
                continue;
            }
            if let Some(m) = msgs.iter().find(|m| m.kind == kind::DATATYPE) {
                out.push((name, addr, Datatype::parse(&m.data)?));
            }
        }
        Ok(out)
    }

    /// Members in creation order when the file tracks it, otherwise in name order. Soft and
    /// external links to datasets are followed; links that cannot be resolved, and links to
    /// groups (which would repeat or loop the tree), are left out: see
    /// [`members_and_skipped`](Self::members_and_skipped).
    ///
    /// # Errors
    /// If the group's structures are corrupt.
    pub fn members(&self) -> Result<Vec<(String, Node)>, Error> {
        Ok(self.members_at(0)?.0)
    }

    /// [`members`](Self::members), and the links left out, with the reason.
    ///
    /// # Errors
    /// As for [`members`](Self::members).
    #[allow(clippy::type_complexity)]
    pub fn members_and_skipped(
        &self,
    ) -> Result<(Vec<(String, Node)>, Vec<(String, String)>), Error> {
        self.members_at(0)
    }

    /// The links of the group, in member order.
    fn links(&self) -> Result<Vec<(String, format::Target)>, Error> {
        let msgs = format::object_header(&self.file, self.addr)?;
        let s = self.file.sizes;
        let mut links: Vec<(String, format::Target, Option<u64>)> = Vec::new();
        for m in &msgs {
            match m.kind {
                kind::SYMBOL_TABLE => {
                    let (bt, heap) = format::symbol_table(&m.data, s)?;
                    let seg = storage::local_heap(&self.file, heap)?;
                    for (name, target) in storage::group_btree(&self.file, bt, &seg)? {
                        links.push((name, target, None));
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
        Ok(links.into_iter().map(|(n, t, _)| (n, t)).collect())
    }

    /// What one link leads to: `Ok(Some(node))`, `Ok(None)` for an object that is not a member
    /// (a committed datatype), or why it is not followed.
    fn follow(
        &self,
        name: &str,
        target: format::Target,
        depth: usize,
    ) -> Result<Result<Option<Node>, String>, Error> {
        let linked = match target {
            format::Target::Hard(addr) => {
                let node = self.file.node(addr, child_path(&self.path, name))?;
                return Ok(Ok(Some(node).filter(|n| !matches!(n, Node::Other))));
            }
            format::Target::Soft(to) => self.resolve_soft(&to, depth),
            format::Target::External { file, path: to } => self.resolve_external(&file, &to, depth),
            format::Target::Unknown(t) => Err(format!("link of unsupported type {t}")),
        };
        Ok(match linked {
            Ok(Node::Dataset(d)) => Ok(Some(Node::Dataset(d))),
            Ok(Node::Group(_)) => Err(
                "a link to a group is not followed (it would repeat or loop the tree)".to_owned(),
            ),
            Ok(Node::Other) => Ok(None),
            Err(why) => Err(why),
        })
    }

    #[allow(clippy::type_complexity)]
    fn members_at(
        &self,
        depth: usize,
    ) -> Result<(Vec<(String, Node)>, Vec<(String, String)>), Error> {
        let mut members = Vec::new();
        let mut skipped = Vec::new();
        for (name, target) in self.links()? {
            match self.follow(&name, target, depth)? {
                Ok(Some(node)) => members.push((name, node)),
                Ok(None) => {}
                Err(why) => skipped.push((name, why)),
            }
        }
        Ok((members, skipped))
    }

    /// The object a soft link names: an absolute path, or one relative to this group.
    fn resolve_soft(&self, to: &str, depth: usize) -> Result<Node, String> {
        if depth >= MAX_LINK_DEPTH {
            return Err("soft links nest too deeply (a cycle?)".to_owned());
        }
        let start = if to.starts_with('/') {
            self.file.root().map_err(|e| e.to_string())?
        } else {
            self.clone()
        };
        start
            .get_at(to, depth.saturating_add(1))
            .map_err(|e| format!("dangling soft link to {to:?}: {e}"))
    }

    /// The object an external link names, in a file relative to this file's directory.
    fn resolve_external(&self, file: &str, to: &str, depth: usize) -> Result<Node, String> {
        if depth >= MAX_LINK_DEPTH {
            return Err("external links nest too deeply (a cycle?)".to_owned());
        }
        let p = Path::new(file);
        let full = match (&self.file.dir, p.is_absolute()) {
            (Some(dir), false) => dir.join(p),
            _ => p.to_path_buf(),
        };
        let other = File::open(&full).map_err(|e| format!("external file {file:?}: {e}"))?;
        other
            .root()
            .and_then(|r| r.get_at(to, depth.saturating_add(1)))
            .map_err(|e| format!("external link to {to:?} in {file:?}: {e}"))
    }

    /// The member at a relative `/`-separated path.
    ///
    /// # Errors
    /// [`Error::NotFound`] if it does not exist.
    pub fn get(&self, path: &str) -> Result<Node, Error> {
        self.get_at(path, 0)
    }

    fn get_at(&self, path: &str, depth: usize) -> Result<Node, Error> {
        let mut node = Node::Group(self.clone());
        for part in path.split('/').filter(|p| !p.is_empty()) {
            let Node::Group(g) = node else {
                return Err(Error::NotFound(path.to_owned()));
            };
            // Only the link needed is followed (following all of them at every step of a
            // chain of links would grow exponentially).
            let target = g
                .links()?
                .into_iter()
                .find(|(n, _)| n == part)
                .map(|(_, t)| t)
                .ok_or_else(|| Error::NotFound(path.to_owned()))?;
            node = g
                .follow(part, target, depth)?
                .ok()
                .flatten()
                .ok_or_else(|| Error::NotFound(path.to_owned()))?;
        }
        Ok(node)
    }

    /// Attributes, in creation order.
    ///
    /// # Errors
    /// If attribute storage is corrupt.
    pub fn attrs(&self) -> Result<Vec<Attribute>, Error> {
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
    /// Object reference targets (object header addresses), for reference attributes.
    ///
    /// # Errors
    /// [`Error::Unsupported`] if the attribute does not hold object references.
    pub fn references(&self) -> Result<Vec<Option<u64>>, Error> {
        if self.dtype != Datatype::ObjectReference {
            return Err(Error::Unsupported(format!(
                "attribute {:?} does not hold references",
                self.name
            )));
        }
        dtype::decode_references(&self.file, &self.data)
    }

    /// Lists of object reference targets, for variable-length reference attributes such as
    /// `DIMENSION_LIST`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] if the attribute does not hold reference lists.
    pub fn reference_lists(&self) -> Result<Vec<Vec<u64>>, Error> {
        if self.dtype != Datatype::ReferenceList {
            return Err(Error::Unsupported(format!(
                "attribute {:?} does not hold reference lists",
                self.name
            )));
        }
        dtype::decode_reference_lists(&self.file, &self.data)
    }

    /// The value.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for datatypes that cannot be converted.
    pub fn value(&self) -> Result<Array, Error> {
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

fn attributes(file: &File, msgs: &[Message]) -> Result<Vec<Attribute>, Error> {
    let mut out = Vec::new();
    let to_attr = |a: format::AttributeMsg| -> Result<Attribute, Error> {
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
    // Compact attributes in creation order when the header tracks it (as netCDF-C reads them).
    let mut compact: Vec<(Option<u16>, Attribute)> = Vec::new();
    for m in msgs {
        match m.kind {
            kind::ATTRIBUTE => {
                if m.flags & 2 != 0 {
                    return Err(Error::Unsupported("shared attribute messages".into()));
                }
                compact.push((m.order, to_attr(format::attribute(file, &m.data)?)?));
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
    if compact.iter().all(|(o, _)| o.is_some()) {
        compact.sort_by_key(|(o, _)| *o);
    }
    let mut all: Vec<Attribute> = compact.into_iter().map(|(_, a)| a).collect();
    all.extend(out);
    Ok(all)
}

/// A dataset.
#[derive(Debug, Clone)]
pub struct Dataset {
    file: File,
    addr: u64,
    path: String,
    space: Option<Space>,
    dtype: Datatype,
    committed: Option<u64>,
    layout: Layout,
    filters: Vec<Filter>,
    fill: Option<Vec<u8>>,
    msgs: Vec<Message>,
}

impl Dataset {
    fn from_messages(file: File, addr: u64, path: String, msgs: &[Message]) -> Result<Self, Error> {
        let s = file.sizes;
        let find = |k: u16| msgs.iter().find(|m| m.kind == k);
        let dt_msg = find(kind::DATATYPE)
            .ok_or_else(|| bad(format!("{path}: dataset without a datatype")))?;
        let (dtype, committed) = if dt_msg.flags & 2 != 0 {
            let (t, at) = dtype::shared_datatype_at(&file, &dt_msg.data)?;
            (t, Some(at))
        } else {
            (Datatype::parse(&dt_msg.data)?, None)
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
            addr,
            path,
            space,
            dtype,
            committed,
            layout,
            filters,
            fill,
            msgs: msgs.to_vec(),
        })
    }

    /// The address of the committed (named) datatype the dataset's type is, if it is one:
    /// [`Group::datatypes`] gives the names.
    #[must_use]
    pub fn committed_type(&self) -> Option<u64> {
        self.committed
    }

    /// The address of the dataset's object header (what object references point to).
    #[must_use]
    pub fn address(&self) -> u64 {
        self.addr
    }

    /// The fill value as raw element bytes, if one is defined.
    #[must_use]
    pub fn fill_value(&self) -> Option<&[u8]> {
        self.fill.as_deref()
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

    /// A plugin compression codec in the filter pipeline (Zstandard, LZ4 or Blosc), as the
    /// writer would reproduce it.
    #[must_use]
    pub fn codec(&self) -> Option<Compression> {
        self.filters.iter().find_map(|f| {
            let c = |i: usize| f.client.get(i).copied();
            match f.id {
                filters::ZSTD => Some(Compression::Zstd {
                    level: c(0).map_or(3, |v| i32::from_ne_bytes(v.to_ne_bytes())),
                }),
                filters::LZ4 => Some(Compression::Lz4),
                filters::BLOSC => Some(Compression::Blosc {
                    codec: match c(6) {
                        Some(0) => BloscCodec::BloscLz,
                        Some(2) => BloscCodec::Lz4Hc,
                        Some(4) => BloscCodec::Zlib,
                        Some(5) => BloscCodec::Zstd,
                        _ => BloscCodec::Lz4,
                    },
                    level: c(4).and_then(|v| u8::try_from(v).ok()).unwrap_or(5),
                    shuffle: match c(5) {
                        Some(1) => BloscShuffle::Byte,
                        Some(2) => BloscShuffle::Bit,
                        _ => BloscShuffle::None,
                    },
                }),
                _ => None,
            }
        })
    }

    /// Whether chunks carry a Fletcher-32 checksum.
    #[must_use]
    pub fn fletcher32(&self) -> bool {
        self.filters.iter().any(|f| f.id == filters::FLETCHER32)
    }

    /// Whether the values are stored big-endian.
    #[must_use]
    pub fn big_endian(&self) -> bool {
        match &self.dtype {
            Datatype::Int { big_endian, .. }
            | Datatype::PackedInt { big_endian, .. }
            | Datatype::Float { big_endian, .. } => *big_endian,
            _ => false,
        }
    }

    /// Whether the values are stored as IEEE half precision (read as `f32`).
    #[must_use]
    pub fn half(&self) -> bool {
        matches!(self.dtype, Datatype::Float { size: 2, .. })
    }

    /// Whether the values are stored contiguously (not chunked, not compact).
    #[must_use]
    pub fn contiguous(&self) -> bool {
        matches!(self.layout, Layout::Contiguous { .. })
    }

    /// Attributes, in creation order.
    ///
    /// # Errors
    /// If attribute storage is corrupt.
    pub fn attrs(&self) -> Result<Vec<Attribute>, Error> {
        attributes(&self.file, &self.msgs)
    }

    /// Read the whole dataset.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for unsupported types, filters or layouts; [`Error::Invalid`]
    /// for corrupt data or datasets over [`MAX_LOAD_BYTES`].
    pub fn read(&self) -> Result<Array, Error> {
        let shape = self.shape();
        let esize = self.dtype.element_size(self.file.sizes.offset);
        if esize == 0 {
            return Err(Error::Unsupported(format!(
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
        let bytes: std::borrow::Cow<'_, [u8]> = match &self.layout {
            Layout::Compact(data) => data
                .get(..total)
                .ok_or_else(|| bad(format!("{}: compact data truncated", self.path)))?
                .into(),
            Layout::Contiguous { addr: None, .. } => filled(&fill, n).into(),
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
                    Vec::new().into()
                } else {
                    // Straight from the (mapped) file: no intermediate copy.
                    self.file.cursor(*a)?.bytes(total)?.into()
                }
            }
            // The whole box as a slice: chunks placed in parallel, numbers straight into
            // their typed buffer.
            Layout::Chunked { .. } | Layout::Virtual { .. } if !shape.is_empty() => {
                return self.read_slice(&vec![0; shape.len()], &shape);
            }
            Layout::Virtual { .. } => {
                let mut out = filled(&fill, n);
                self.place_box(&shape, &[], &[], esize, &mut out)?;
                out.into()
            }
            Layout::Chunked { addr, chunk, index } => {
                let mut out = filled(&fill, n);
                if let Some(addr) = addr {
                    let cdims: Vec<u64> = chunk.iter().take(shape.len()).copied().collect();
                    let whole = vec![0usize; shape.len()];
                    let chunks = self.chunk_list(*addr, &cdims, index, &shape, esize)?;
                    self.place_chunks(&chunks, &cdims, &shape, esize, &whole, &shape, &mut out)?;
                }
                out.into()
            }
        };
        dtype::decode(&self.file, &self.dtype, &bytes, &shape)
    }

    /// A variable-length sequence dataset (a ragged array, [`Datatype::VarSeq`]): each
    /// element's length, shaped like the dataset, and all elements' values concatenated in C
    /// order.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for any other datatype; otherwise as for [`read`](Self::read).
    pub fn read_ragged(&self) -> Result<(ndarray::ArrayD<u64>, Array), Error> {
        let Datatype::VarSeq { base } = &self.dtype else {
            return Err(Error::Unsupported(format!(
                "{}: not a variable-length sequence",
                self.path
            )));
        };
        let shape = self.shape();
        let bytes = self.raw_bytes(&shape)?;
        dtype::decode_ragged(&self.file, base, &bytes, &shape)
    }

    /// A compound (record) dataset as one array per leaf field, in declaration order: nested
    /// members get dotted paths (`wind.u`), and array members their dimensions after the
    /// dataset's.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for other datatypes and for members this reader cannot decode
    /// (arrays of records, variable-length sequences); otherwise as for [`read`](Self::read).
    pub fn read_compound(&self) -> Result<Vec<(String, Array)>, Error> {
        let Datatype::Compound { size, members } = &self.dtype else {
            return Err(Error::Unsupported(format!(
                "{}: not a compound type",
                self.path
            )));
        };
        let shape = self.shape();
        let bytes = self.raw_bytes(&shape)?;
        let mut leaves = Vec::new();
        dtype::fields(members, "", 0, &mut leaves)?;
        let n = bytes.len().checked_div(*size).unwrap_or(0);
        let mut out = Vec::with_capacity(leaves.len());
        for f in leaves {
            let fsize = f
                .dims
                .iter()
                .try_fold(f.dtype.element_size(self.file.sizes.offset), |a, d| {
                    a.checked_mul(*d)
                })
                .ok_or_else(|| bad("compound member too large"))?;
            if f.offset.saturating_add(fsize) > *size {
                return Err(bad(format!(
                    "compound member {:?} outside its record",
                    f.path
                )));
            }
            let mut column = Vec::with_capacity(n.saturating_mul(fsize));
            for i in 0..n {
                let start = i.saturating_mul(*size).saturating_add(f.offset);
                column.extend_from_slice(
                    bytes
                        .get(start..start.saturating_add(fsize))
                        .ok_or_else(|| bad("compound data truncated"))?,
                );
            }
            let mut fshape = shape.clone();
            fshape.extend(&f.dims);
            let values =
                dtype::decode(&self.file, f.dtype, &column, &fshape).map_err(|e| match e {
                    Error::Unsupported(m) => {
                        Error::Unsupported(format!("compound member {:?}: {m}", f.path))
                    }
                    other => other,
                })?;
            out.push((f.path, values));
        }
        Ok(out)
    }

    /// Every element's stored bytes (before decoding), in C order.
    fn raw_bytes(&self, shape: &[usize]) -> Result<Vec<u8>, Error> {
        let esize = self.dtype.element_size(self.file.sizes.offset);
        let n = shape
            .iter()
            .try_fold(Ck::from(1usize), |a, d| Some(a * *d))
            .and_then(Ck::usize)
            .ok_or_else(|| bad("dataset too large"))?;
        let total = (Ck::from(n) * esize)
            .usize()
            .filter(|t| *t <= self.file.load_limit)
            .ok_or_else(|| bad(format!("{}: too large to load", self.path)))?;
        Ok(match &self.layout {
            Layout::Compact(data) => data
                .get(..total)
                .ok_or_else(|| bad(format!("{}: compact data truncated", self.path)))?
                .to_vec(),
            Layout::Contiguous { addr: None, .. } | Layout::Chunked { addr: None, .. } => {
                vec![0; total]
            }
            Layout::Virtual { .. } => {
                let mut out = filled(&self.fill_bytes(esize), n);
                self.place_box(shape, &vec![0; shape.len()], shape, esize, &mut out)?;
                out
            }
            Layout::Contiguous { addr: Some(a), .. } => {
                if total == 0 {
                    Vec::new()
                } else {
                    self.file.cursor(*a)?.bytes(total)?.to_vec()
                }
            }
            Layout::Chunked {
                addr: Some(addr),
                chunk,
                index,
            } => {
                let mut out = vec![0; total];
                let cdims: Vec<u64> = chunk.iter().take(shape.len()).copied().collect();
                let whole = vec![0usize; shape.len()];
                let chunks = self.chunk_list(*addr, &cdims, index, shape, esize)?;
                self.place_chunks(&chunks, &cdims, shape, esize, &whole, shape, &mut out)?;
                out
            }
        })
    }

    fn chunk_list(
        &self,
        addr: u64,
        cdims: &[u64],
        index: &ChunkIndex,
        shape: &[usize],
        esize: usize,
    ) -> Result<Vec<ChunkRef>, Error> {
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
                // The chunks are stored back to back, so a valid count fits in the file; a
                // corrupted extent must not make this loop run (practically) forever.
                let total = nc
                    .iter()
                    .try_fold(1u64, |a, n| a.checked_mul(*n))
                    .filter(|t| {
                        t.checked_mul(chunk_bytes)
                            .is_some_and(|b| b <= self.file.bytes_after(addr))
                    })
                    .ok_or_else(|| bad("implicit chunk index runs past the end of the file"))?;
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

    /// Decode `chunks` and copy their parts of the box `[lo, lo + count)` into `out` (which
    /// holds exactly that box, row-major). With the `rayon` feature, filtered (compressed)
    /// chunks are decoded in parallel, a few batches per thread at a time to bound memory.
    #[allow(clippy::too_many_arguments)]
    fn place_chunks(
        &self,
        chunks: &[ChunkRef],
        cdims: &[u64],
        shape: &[usize],
        esize: usize,
        lo: &[usize],
        count: &[usize],
        out: &mut [u8],
    ) -> Result<(), Error> {
        #[cfg(feature = "rayon")]
        if chunks.len() > 1
            && (!self.filters.is_empty() || out.len() >= PARALLEL_COPY_BYTES)
            && self.place_by_slab(chunks, cdims, shape, esize, lo, count, out)?
        {
            return Ok(());
        }
        #[cfg(feature = "rayon")]
        if !self.filters.is_empty() && chunks.len() > 1 {
            use rayon::prelude::*;
            // Decoded chunks in flight: up to four per thread, so large reads stay within
            // memory.
            let threads = rayon::current_num_threads().max(1);
            let chunk_bytes = cdims
                .iter()
                .try_fold(esize, |a, c| a.checked_mul(usize::try_from(*c).ok()?))
                .unwrap_or(usize::MAX)
                .max(1);
            // Within the budget of `set_batch_bytes`, at least one.
            let batch = batch_bytes()
                .checked_div(chunk_bytes)
                .unwrap_or(0)
                .clamp(1, threads.saturating_mul(4));
            for group in chunks.chunks(batch) {
                let decoded = group
                    .par_iter()
                    .map(|c| self.decode_chunk(c, cdims, shape, esize, lo, count))
                    .collect::<Result<Vec<_>, Error>>()?;
                for p in decoded.into_iter().flatten() {
                    p.copy_into(out, count, lo, esize)?;
                }
            }
            return Ok(());
        }
        for c in chunks {
            if let Some(p) = self.decode_chunk(c, cdims, shape, esize, lo, count)? {
                p.copy_into(out, count, lo, esize)?;
            }
        }
        Ok(())
    }

    /// [`place_chunks`](Self::place_chunks) in parallel by slab: the chunks starting at one
    /// position on the first axis cover a contiguous run of rows of the box, so each such
    /// slab of `out` is filled by one thread (decoding its chunks one at a time, so memory
    /// stays at a chunk per thread). False, with nothing done, when there are fewer than two
    /// slabs.
    #[cfg(feature = "rayon")]
    #[allow(clippy::too_many_arguments)]
    fn place_by_slab(
        &self,
        chunks: &[ChunkRef],
        cdims: &[u64],
        shape: &[usize],
        esize: usize,
        lo: &[usize],
        count: &[usize],
        out: &mut [u8],
    ) -> Result<bool, Error> {
        use rayon::prelude::*;
        let (Some(&lo0), Some(&n0), Some(&c0)) = (lo.first(), count.first(), cdims.first()) else {
            return Ok(false);
        };
        let c0 = usize::try_from(c0).map_err(|_| bad("chunk too large"))?;
        let row = count
            .iter()
            .skip(1)
            .try_fold(Ck::from(esize), |a, c| Some(a * *c))
            .and_then(Ck::usize)
            .ok_or_else(|| bad("box too large"))?;
        // Chunks grouped by their first-axis offset, in order.
        let mut groups: std::collections::BTreeMap<u64, Vec<&ChunkRef>> =
            std::collections::BTreeMap::new();
        for c in chunks {
            groups
                .entry(c.offset.first().copied().unwrap_or(0))
                .or_default()
                .push(c);
        }
        if groups.len() < 2 {
            return Ok(false);
        }
        // Each group's rows of the box, [a, b), cut out of `out` in order.
        let mut slabs = Vec::with_capacity(groups.len());
        let mut rest = out;
        let mut at = 0usize;
        for (o, group) in groups {
            let o = usize::try_from(o).map_err(|_| bad("chunk offset"))?;
            let a = o.max(lo0).saturating_sub(lo0);
            let b = o
                .saturating_add(c0)
                .min(lo0.saturating_add(n0))
                .saturating_sub(lo0);
            if a >= b {
                continue;
            }
            let skip = (Ck::from(a.saturating_sub(at)) * row)
                .usize()
                .ok_or_else(|| bad("box too large"))?;
            let len = (Ck::from(b.saturating_sub(a)) * row)
                .usize()
                .ok_or_else(|| bad("box too large"))?;
            let tail = rest
                .get_mut(skip..)
                .ok_or_else(|| bad("box out of range"))?;
            if len > tail.len() {
                return Err(bad("box out of range"));
            }
            let (slab, tail) = tail.split_at_mut(len);
            rest = tail;
            at = b;
            slabs.push((slab, a, b, group));
        }
        slabs
            .into_par_iter()
            .try_for_each(|(slab, a, b, group)| -> Result<(), Error> {
                let mut sub_lo = lo.to_vec();
                let mut sub_count = count.to_vec();
                if let (Some(l), Some(n)) = (sub_lo.first_mut(), sub_count.first_mut()) {
                    *l = lo0.saturating_add(a);
                    *n = b.saturating_sub(a);
                }
                for c in group {
                    if let Some(p) =
                        self.decode_chunk(c, cdims, shape, esize, &sub_lo, &sub_count)?
                    {
                        p.copy_into(slab, &sub_count, &sub_lo, esize)?;
                    }
                }
                Ok(())
            })?;
        Ok(true)
    }

    /// Decode one chunk if it overlaps the box `[lo, lo + count)`.
    fn decode_chunk(
        &self,
        c: &ChunkRef,
        cdims: &[u64],
        shape: &[usize],
        esize: usize,
        lo: &[usize],
        count: &[usize],
    ) -> Result<Option<Decoded<'_>>, Error> {
        let cd: Vec<usize> = cdims
            .iter()
            .map(|d| usize::try_from(*d).unwrap_or(usize::MAX))
            .collect();
        let off: Vec<usize> = c
            .offset
            .iter()
            .map(|o| usize::try_from(*o).unwrap_or(usize::MAX))
            .collect();
        // The part of this chunk inside both the dataset and the requested box.
        let mut from = Vec::with_capacity(shape.len());
        let mut to = Vec::with_capacity(shape.len());
        for k in 0..shape.len() {
            let o = off.get(k).copied().unwrap_or(0);
            let l = lo.get(k).copied().unwrap_or(0);
            let a = o.max(l);
            let b = o
                .saturating_add(cd.get(k).copied().unwrap_or(0))
                .min(shape.get(k).copied().unwrap_or(0))
                .min(l.saturating_add(count.get(k).copied().unwrap_or(0)));
            if a >= b {
                return Ok(None);
            }
            from.push(a);
            to.push(b);
        }
        let chunk_elems = cd
            .iter()
            .try_fold(Ck::from(1usize), |a, d| Some(a * *d))
            .and_then(Ck::usize)
            .ok_or_else(|| bad("chunk too large"))?;
        let expected = (Ck::from(chunk_elems) * esize)
            .usize()
            .ok_or_else(|| bad("chunk too large"))?;
        let stored = usize::try_from(c.size).map_err(|_| bad("chunk too large"))?;
        let raw = self.file.cursor(c.addr)?.bytes(stored)?;
        let data: std::borrow::Cow<'_, [u8]> = if self.filters.is_empty() {
            raw.into()
        } else {
            filters::decode(&self.filters, c.mask, raw.to_vec(), expected)?.into()
        };
        if data.len() < expected {
            return Err(bad(format!(
                "{}: chunk at {:?} is short",
                self.path, c.offset
            )));
        }
        Ok(Some(Decoded {
            data,
            dims: cd,
            origin: off,
            from,
            to,
        }))
    }

    /// Read the hyperslab `[start, start + count)`. Only the chunks (or, for contiguous data,
    /// the rows) that intersect it are touched, so slices of large mapped files are cheap.
    ///
    /// # Errors
    /// [`Error::Invalid`] if the slice is outside the dataset; otherwise as for
    /// [`read`](Self::read).
    pub fn read_slice(&self, start: &[usize], count: &[usize]) -> Result<Array, Error> {
        let shape = self.shape();
        if start.len() != shape.len() || count.len() != shape.len() {
            return Err(bad(format!(
                "{}: slice rank does not match the dataset",
                self.path
            )));
        }
        if start
            .iter()
            .zip(count)
            .zip(&shape)
            .any(|((s, c), d)| s.checked_add(*c).is_none_or(|e| e > *d))
        {
            return Err(bad(format!(
                "{}: slice {start:?}+{count:?} is outside shape {shape:?}",
                self.path
            )));
        }
        let esize = self.dtype.element_size(self.file.sizes.offset);
        let n = count
            .iter()
            .try_fold(Ck::from(1usize), |a, d| Some(a * *d))
            .and_then(Ck::usize)
            .ok_or_else(|| bad("slice too large"))?;
        // Enforce the load limit before allocating.
        (Ck::from(n) * esize)
            .usize()
            .filter(|t| *t <= self.file.load_limit)
            .ok_or_else(|| bad(format!("{}: slice too large to load", self.path)))?;
        // A box that is one run of contiguous storage (single positions on the leading
        // axes, then any range, then whole trailing axes) is decoded straight from the file.
        if let Layout::Contiguous { addr: Some(a), .. } = &self.layout
            && n > 0
            && let Some((offset, len)) = contiguous_run(&shape, start, count, esize)
        {
            let at = u64::try_from(offset)
                .ok()
                .and_then(|o| a.checked_add(o))
                .ok_or_else(|| bad("dataset too large"))?;
            if len >= PREFETCH_BYTES {
                self.file.prefetch(at, len);
            }
            let src = self.file.cursor(at)?.bytes(len)?;
            return dtype::decode(&self.file, &self.dtype, src, count);
        }
        let fill: Vec<u8> = match &self.fill {
            Some(f) if f.len() == esize => f.clone(),
            _ => vec![0; esize],
        };
        let place = |out: &mut [u8]| self.place_box(&shape, start, count, esize, out);
        // Native-endian numbers are placed straight into their typed buffer.
        let shaped = |a: Result<Array, ndarray::ShapeError>| {
            a.map_err(|e| bad(format!("data does not match its shape: {e}")))
        };
        let sh = ndarray::IxDyn(count);
        let native = |be: bool| be == cfg!(target_endian = "big");
        match self.dtype {
            Datatype::Int {
                size,
                signed,
                big_endian,
            } if native(big_endian) => {
                return shaped(match (size, signed) {
                    (1, true) => ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::I8),
                    (1, false) => ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::U8),
                    (2, true) => ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::I16),
                    (2, false) => ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::U16),
                    (4, true) => ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::I32),
                    (4, false) => ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::U32),
                    (8, true) => ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::I64),
                    _ => ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::U64),
                });
            }
            Datatype::Float {
                size: 4,
                big_endian,
            } if native(big_endian) => {
                return shaped(
                    ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::F32),
                );
            }
            Datatype::Float {
                size: 8,
                big_endian,
            } if native(big_endian) => {
                return shaped(
                    ndarray::ArrayD::from_shape_vec(sh, typed_fill(n, &fill, place)?)
                        .map(Array::F64),
                );
            }
            _ => {}
        }
        let mut out = filled(&fill, n);
        place(&mut out)?;
        dtype::decode(&self.file, &self.dtype, &out, count)
    }
}

impl Dataset {
    /// Copy the stored bytes of the box `[start, start + count)` into `out` (C order, `esize`
    /// bytes per element, already holding the fill value): what is not stored is left as it
    /// is.
    fn place_box(
        &self,
        shape: &[usize],
        start: &[usize],
        count: &[usize],
        esize: usize,
        out: &mut [u8],
    ) -> Result<(), Error> {
        if count.contains(&0) {
            return Ok(());
        }
        let to: Vec<usize> = start
            .iter()
            .zip(count)
            .map(|(s, c)| s.saturating_add(*c))
            .collect();
        let origin = vec![0usize; shape.len()];
        match &self.layout {
            Layout::Compact(data) => {
                copy_box(data, shape, &origin, out, count, start, start, &to, esize)
            }
            Layout::Contiguous { addr: Some(a), .. } => {
                let full = shape
                    .iter()
                    .try_fold(Ck::from(esize), |acc, d| Some(acc * *d))
                    .and_then(Ck::usize)
                    .ok_or_else(|| bad("dataset too large"))?;
                // A view of the mapped file; only the rows copied below are paged in.
                let src = self.file.cursor(*a)?.bytes(full)?;
                copy_box(src, shape, &origin, out, count, start, start, &to, esize)
            }
            Layout::Chunked {
                addr: Some(addr),
                chunk,
                index,
            } => {
                let cdims: Vec<u64> = chunk.iter().take(shape.len()).copied().collect();
                let chunks = self.chunk_list(*addr, &cdims, index, shape, esize)?;
                self.place_chunks(&chunks, &cdims, shape, esize, start, count, out)
            }
            Layout::Virtual { heap, index } => {
                virtual_ds::place(self, *heap, *index, shape, start, count, esize, out)
            }
            Layout::Contiguous { addr: None, .. } | Layout::Chunked { addr: None, .. } => Ok(()),
        }
    }

    /// The fill value's bytes (zeros when there is none).
    fn fill_bytes(&self, esize: usize) -> Vec<u8> {
        match &self.fill {
            Some(f) if f.len() == esize => f.clone(),
            _ => vec![0; esize],
        }
    }
}

/// Uncompressed chunked reads at least this large are copied in parallel.
#[cfg(feature = "rayon")]
const PARALLEL_COPY_BYTES: usize = 1 << 22;

/// Contiguous reads at least this large are paged in up front.
const PREFETCH_BYTES: usize = 1 << 22;

/// The byte range `(offset, len)` of the box `[start, start + count)` in C-order storage of
/// `shape`, if the box is one contiguous run: single positions on the leading axes, any range
/// on one axis, whole trailing axes.
fn contiguous_run(
    shape: &[usize],
    start: &[usize],
    count: &[usize],
    esize: usize,
) -> Option<(usize, usize)> {
    // The first axis (from the end) that is not taken whole.
    let k = (0..shape.len())
        .rev()
        .find(|&d| start.get(d) != Some(&0) || count.get(d) != shape.get(d))
        .unwrap_or(0);
    if count.iter().take(k).any(|c| *c != 1) {
        return None;
    }
    let mut stride = esize;
    let mut offset = 0usize;
    for d in (0..shape.len()).rev() {
        if d <= k {
            offset = offset.checked_add(start.get(d)?.checked_mul(stride)?)?;
        }
        stride = stride.checked_mul(*shape.get(d)?)?;
    }
    let len = count.iter().try_fold(esize, |a, c| a.checked_mul(*c))?;
    Some((offset, len))
}

/// `n` values, each the fill value (`fill`, one element's bytes), then filled in by `place`
/// through a byte view: native-endian numbers need no separate decoding pass.
fn typed_fill<T: bytemuck::Pod>(
    n: usize,
    fill: &[u8],
    place: impl FnOnce(&mut [u8]) -> Result<(), Error>,
) -> Result<Vec<T>, Error> {
    let f: T = bytemuck::try_pod_read_unaligned(fill).unwrap_or_else(|_| T::zeroed());
    let mut v = vec![f; n];
    place(bytemuck::cast_slice_mut(&mut v))?;
    Ok(v)
}

/// `n` copies of the fill value (zeros straight from the allocator).
fn filled(fill: &[u8], n: usize) -> Vec<u8> {
    if fill.iter().all(|b| *b == 0) {
        vec![0; fill.len().saturating_mul(n)]
    } else {
        fill.repeat(n)
    }
}

/// A decoded chunk and the part of it to copy.
struct Decoded<'a> {
    data: std::borrow::Cow<'a, [u8]>,
    dims: Vec<usize>,
    origin: Vec<usize>,
    from: Vec<usize>,
    to: Vec<usize>,
}

impl Decoded<'_> {
    fn copy_into(
        &self,
        out: &mut [u8],
        count: &[usize],
        lo: &[usize],
        esize: usize,
    ) -> Result<(), Error> {
        copy_box(
            &self.data,
            &self.dims,
            &self.origin,
            out,
            count,
            lo,
            &self.from,
            &self.to,
            esize,
        )
    }
}

/// Copy the box `[from, to)` (global coordinates) from `src` (dims `src_dims`, origin
/// `src_origin`) into `dst` (dims `dst_dims`, origin `dst_origin`). Row-major, `esize` bytes per
/// element; every index is bounds-checked.
#[allow(clippy::too_many_arguments)]
fn copy_box(
    src: &[u8],
    src_dims: &[usize],
    src_origin: &[usize],
    dst: &mut [u8],
    dst_dims: &[usize],
    dst_origin: &[usize],
    from: &[usize],
    to: &[usize],
    esize: usize,
) -> Result<(), Error> {
    let rank = from.len();
    if rank == 0 {
        let (Some(d), Some(s)) = (dst.get_mut(..esize), src.get(..esize)) else {
            return Err(bad("scalar out of range"));
        };
        d.copy_from_slice(s);
        return Ok(());
    }
    let last = rank.saturating_sub(1);
    let extent: Vec<usize> = from
        .iter()
        .zip(to)
        .map(|(a, b)| b.saturating_sub(*a))
        .collect();
    let run = (Ck::from(extent.get(last).copied().unwrap_or(0)) * esize)
        .usize()
        .ok_or_else(|| bad("offset overflow"))?;
    let outer: Vec<u64> = extent.iter().take(last).map(|e| *e as u64).collect();
    let rows: u64 = outer
        .iter()
        .try_fold(1u64, |a, e| a.checked_mul(*e))
        .ok_or_else(|| bad("box too large"))?;
    for r in 0..rows {
        let idx = unravel(r, &outer).unwrap_or_default();
        let mut s = Ck::from(0usize);
        let mut d = Ck::from(0usize);
        for k in 0..rank {
            let g = from.get(k).copied().unwrap_or(0).saturating_add(
                idx.get(k)
                    .map_or(0, |x| usize::try_from(*x).unwrap_or(usize::MAX)),
            );
            s = s * src_dims.get(k).copied().unwrap_or(0)
                + (Ck::from(g) - src_origin.get(k).copied().unwrap_or(0));
            d = d * dst_dims.get(k).copied().unwrap_or(0)
                + (Ck::from(g) - dst_origin.get(k).copied().unwrap_or(0));
        }
        let s = (s * esize).usize().ok_or_else(|| bad("offset overflow"))?;
        let d = (d * esize).usize().ok_or_else(|| bad("offset overflow"))?;
        let (Some(dd), Some(ss)) = (
            dst.get_mut(d..d.saturating_add(run)),
            src.get(s..s.saturating_add(run)),
        ) else {
            return Err(bad("data outside its bounds"));
        };
        dd.copy_from_slice(ss);
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::contiguous_run;

    #[test]
    fn contiguous_runs() {
        let shape = [4, 3, 5];
        // The whole array; a range of leading rows; one row's tail.
        assert_eq!(
            contiguous_run(&shape, &[0, 0, 0], &[4, 3, 5], 4),
            Some((0, 240))
        );
        assert_eq!(
            contiguous_run(&shape, &[1, 0, 0], &[2, 3, 5], 4),
            Some((60, 120))
        );
        assert_eq!(
            contiguous_run(&shape, &[2, 1, 0], &[1, 2, 5], 4),
            Some((140, 40))
        );
        assert_eq!(
            contiguous_run(&shape, &[2, 1, 3], &[1, 1, 2], 4),
            Some((152, 8))
        );
        // Not one run: a partial trailing axis under a range.
        assert_eq!(contiguous_run(&shape, &[0, 0, 1], &[2, 3, 4], 4), None);
        assert_eq!(contiguous_run(&shape, &[0, 1, 0], &[2, 2, 5], 4), None);
    }
}
