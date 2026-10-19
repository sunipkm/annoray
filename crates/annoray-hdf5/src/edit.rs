//! Changing an HDF5 file in place: writing values into existing datasets and growing
//! datasets along their unlimited dimensions.
//!
//! Nothing already in the file moves. New and rewritten chunks, and each chunk index rebuilt
//! to take them, are appended at the end of the file; the messages that point at them (the
//! layout's index address, the dataspace's current dimensions) keep their size and are
//! rewritten where they are, with the object header's checksum; so is the superblock's
//! end-of-file address. Space a rewritten chunk or index used before is not reclaimed (as
//! with the HDF5 library without a free-space manager); `h5repack` reclaims it.
//!
//! Supported storage: contiguous (allocated on the first write if it is not yet), compact,
//! and chunked with a version-1 B-tree index (what netCDF-C writes, h5py by default, and this
//! crate's writer) or a single chunk. The newer indexes of files written with
//! `libver="latest"` (fixed and extensible arrays, version-2 B-trees) are refused, as are
//! files with other than 8-byte offsets and lengths.
//!
//! Values are buffered per chunk and written when [`Editor::flush`] (or
//! [`Editor::close`]) runs, or when the buffered chunks pass [`Editor::set_cache_bytes`];
//! [`Editor::read`] sees them before that. Readers that opened the file earlier keep what
//! they had (chunks are never overwritten), and see the new metadata only once reopened.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::checksum::lookup3;
use crate::ck::Ck;
use crate::format::{ChunkIndex, Layout, kind};
use crate::{Array, Dataset, Datatype, Error, File, Node, bad, dtype};

/// Buffered chunks past this many bytes are written out (see [`Editor::set_cache_bytes`]).
const DEFAULT_CACHE_BYTES: usize = 1 << 28;

const UNDEF: u64 = u64::MAX;

/// An HDF5 file open for changes (see the module documentation).
#[derive(Debug)]
pub struct Editor {
    path: PathBuf,
    out: fs::File,
    file: File,
    /// The physical end of the file: where the next block goes.
    end: u64,
    sets: HashMap<String, Edited>,
    cache_limit: usize,
    cached: usize,
}

#[derive(Debug)]
struct Edited {
    ds: Dataset,
    shape: Vec<usize>,
    resized: bool,
    esize: usize,
    fill: Vec<u8>,
    store: Store,
}

#[derive(Debug)]
enum Store {
    Contiguous {
        /// The data's address (`None` until allocated).
        addr: Option<u64>,
        size: u64,
        /// Where the layout message holds the address (physical).
        addr_at: u64,
    },
    Compact {
        /// Where the values are (physical), inside the layout message.
        at: u64,
        len: usize,
    },
    Chunked {
        cdims: Vec<usize>,
        index: Index,
        /// Chunks in the file, by their first element: `(address, stored size, filter mask)`.
        chunks: BTreeMap<Vec<u64>, (u64, u32, u32)>,
        /// Chunks changed since the last write, decoded.
        dirty: BTreeMap<Vec<u64>, Vec<u8>>,
        /// Whether the index must be rebuilt.
        changed: bool,
    },
}

#[derive(Debug, Clone, Copy)]
enum Index {
    /// A version-1 B-tree; the layout message holds its address at `addr_at`.
    BTree1 { addr_at: u64 },
    /// One chunk; the layout message holds its address, and for filtered chunks its size and
    /// filter mask (`filtered_at`).
    Single {
        addr_at: u64,
        filtered_at: Option<u64>,
    },
}

impl Editor {
    /// Open `path` for changes.
    ///
    /// # Errors
    /// [`Error::Io`], [`Error::Invalid`] for files that are not HDF5, and
    /// [`Error::Unsupported`] for offset or length sizes other than 8 bytes.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref().to_path_buf();
        let out = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        let file = File::open(&path)?;
        if file.sizes.offset != 8 || file.sizes.length != 8 {
            return Err(Error::Unsupported(
                "changing files with offsets or lengths other than 8 bytes".into(),
            ));
        }
        let end = out.metadata().map_err(|e| Error::Io(e.to_string()))?.len();
        Ok(Self {
            path,
            out,
            file,
            end,
            sets: HashMap::new(),
            cache_limit: DEFAULT_CACHE_BYTES,
            cached: 0,
        })
    }

    /// The file as of the last [`flush`](Self::flush) (or opening).
    #[must_use]
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Write buffered chunks once they pass `bytes` (default 256 MiB).
    pub fn set_cache_bytes(&mut self, bytes: usize) {
        self.cache_limit = bytes;
    }

    /// A dataset's current shape (with changes not yet flushed).
    ///
    /// # Errors
    /// [`Error::NotFound`], or as for [`write`](Self::write) for storage that cannot change.
    pub fn shape(&mut self, path: &str) -> Result<Vec<usize>, Error> {
        Ok(self.edited(path)?.shape.clone())
    }

    /// A dataset's maximum shape (`None` along unlimited dimensions).
    ///
    /// # Errors
    /// As for [`shape`](Self::shape).
    pub fn max_shape(&mut self, path: &str) -> Result<Vec<Option<usize>>, Error> {
        let e = self.edited(path)?;
        Ok(e.ds.space.as_ref().map_or_else(Vec::new, |s| {
            s.max
                .iter()
                .map(|m| (*m != u64::MAX).then(|| usize::try_from(*m).unwrap_or(usize::MAX)))
                .collect()
        }))
    }

    /// Grow a chunked dataset to `shape` (within its maximum shape). New elements hold the
    /// fill value until written.
    ///
    /// # Errors
    /// [`Error::Invalid`] for a different rank, a smaller extent or one past the maximum;
    /// [`Error::Unsupported`] for datasets that are not chunked.
    pub fn resize(&mut self, path: &str, shape: &[usize]) -> Result<(), Error> {
        let e = self.edited(path)?;
        if shape == e.shape.as_slice() {
            return Ok(());
        }
        if shape.len() != e.shape.len() {
            return Err(bad(format!(
                "{path}: rank {} for {:?}",
                shape.len(),
                e.shape
            )));
        }
        if !matches!(e.store, Store::Chunked { .. }) {
            return Err(Error::Unsupported(format!(
                "{path}: only chunked datasets change shape"
            )));
        }
        let max =
            e.ds.space
                .as_ref()
                .map(|s| s.max.clone())
                .unwrap_or_default();
        for ((new, old), m) in shape.iter().zip(&e.shape).zip(&max) {
            if new < old {
                return Err(bad(format!(
                    "{path}: shrinking {:?} to {shape:?} is not supported",
                    e.shape
                )));
            }
            if *m != u64::MAX && (*new as u64) > *m {
                return Err(bad(format!(
                    "{path}: {shape:?} is past the maximum shape {max:?}"
                )));
            }
        }
        e.shape = shape.to_vec();
        e.resized = true;
        Ok(())
    }

    /// Write `values` into a dataset at `start` (the box `start .. start + values.shape()`).
    /// The values must be of the dataset's own element type (as [`Dataset::read`] gives
    /// it); bytes are stored in the dataset's byte order and precision.
    ///
    /// # Errors
    /// [`Error::Invalid`] outside the dataset or for another element type;
    /// [`Error::Unsupported`] for storage or element types this editor does not change
    /// (variable-length data, compound types, see the module documentation).
    pub fn write(&mut self, path: &str, start: &[usize], values: &Array) -> Result<(), Error> {
        let count = values.shape().to_vec();
        let e = self.edited(path)?;
        check_box(path, &e.shape, start, &count)?;
        let bytes = element_bytes(path, &e.ds.dtype, values)?;
        if count.contains(&0) {
            return Ok(());
        }
        let esize = e.esize;
        let shape = e.shape.clone();
        let fill = e.fill.clone();
        match &e.store {
            Store::Compact { at, len } => {
                let (at, len) = (*at, *len);
                let mut data = self.pread(at, len)?;
                copy_into(&bytes, &count, start, &mut data, &shape, esize)?;
                self.pwrite(at, &data)?;
                let checksummed = self.layout_checksum(path)?;
                self.rechecksum(checksummed)?;
            }
            Store::Contiguous {
                addr,
                size,
                addr_at,
            } => {
                let (addr, size, addr_at) = (*addr, *size, *addr_at);
                let addr = match addr {
                    Some(a) => a,
                    None => {
                        let a = self.allocate_filled(size, &fill)?;
                        self.pwrite(addr_at, &a.to_le_bytes())?;
                        let checksummed = self.layout_checksum(path)?;
                        self.rechecksum(checksummed)?;
                        if let Some(Edited {
                            store: Store::Contiguous { addr, .. },
                            ..
                        }) = self.sets.get_mut(path)
                        {
                            *addr = Some(a);
                        }
                        a
                    }
                };
                self.write_rows(addr, &shape, start, &count, esize, &bytes)?;
            }
            Store::Chunked { cdims, .. } => {
                let cdims = cdims.clone();
                for origin in chunks_in(&cdims, start, &count) {
                    let mut chunk = self.chunk_bytes(path, &origin)?;
                    let corigin: Vec<usize> = origin.iter().map(|o| *o as usize).collect();
                    copy_between(&bytes, &count, start, &mut chunk, &cdims, &corigin, esize)?;
                    self.put_dirty(path, origin, chunk)?;
                }
                if self.cached > self.cache_limit {
                    self.write_dirty()?;
                }
            }
        }
        Ok(())
    }

    /// Read the box `start .. start + count` of a dataset, changes not yet flushed included.
    ///
    /// # Errors
    /// [`Error::Invalid`] outside the dataset; decoding errors.
    pub fn read(&mut self, path: &str, start: &[usize], count: &[usize]) -> Result<Array, Error> {
        let e = self.edited(path)?;
        check_box(path, &e.shape, start, count)?;
        let esize = e.esize;
        let shape = e.shape.clone();
        let fill = e.fill.clone();
        let dtype = e.ds.dtype.clone();
        let n = count
            .iter()
            .try_fold(Ck::from(1usize), |a, c| Some(a * *c))
            .and_then(Ck::usize)
            .ok_or_else(|| bad("box too large"))?;
        let mut out = crate::filled(&fill, n);
        enum Kind {
            Compact(u64, usize),
            Contiguous(Option<u64>),
            Chunked(Vec<usize>),
        }
        let kind = match &e.store {
            Store::Compact { at, len } => Kind::Compact(*at, *len),
            Store::Contiguous { addr, .. } => Kind::Contiguous(*addr),
            Store::Chunked { cdims, .. } => Kind::Chunked(cdims.clone()),
        };
        if n > 0 {
            match kind {
                Kind::Compact(at, len) => {
                    let data = self.pread(at, len)?;
                    copy_out(&data, &shape, &mut out, start, count, esize)?;
                }
                Kind::Contiguous(None) => {}
                Kind::Contiguous(Some(a)) => {
                    self.read_rows(a, &shape, start, count, esize, &mut out)?;
                }
                Kind::Chunked(cdims) => {
                    for origin in chunks_in(&cdims, start, count) {
                        let chunk = self.chunk_bytes(path, &origin)?;
                        let corigin: Vec<usize> = origin.iter().map(|o| *o as usize).collect();
                        copy_between(&chunk, &cdims, &corigin, &mut out, count, start, esize)?;
                    }
                }
            }
        }
        dtype::decode(&self.file, &dtype, &out, count)
    }

    /// Write buffered chunks and rebuilt indexes, rewrite changed headers and the
    /// superblock, and reopen the file (so [`file`](Self::file) shows the changes).
    ///
    /// # Errors
    /// I/O and encoding errors.
    pub fn flush(&mut self) -> Result<(), Error> {
        self.write_dirty()?;
        let paths: Vec<String> = self.sets.keys().cloned().collect();
        let mut changed_any = false;
        for path in paths {
            let Some(e) = self.sets.get(&path) else {
                continue;
            };
            let resized = e.resized;
            let rebuild = match &e.store {
                Store::Chunked {
                    changed: true,
                    index,
                    chunks,
                    cdims,
                    ..
                } => Some((*index, chunks.clone(), cdims.clone(), e.shape.clone())),
                _ => None,
            };
            if let Some((index, chunks, cdims, shape)) = rebuild {
                self.write_index(&path, index, &chunks, &cdims, &shape)?;
                changed_any = true;
            }
            if resized {
                self.write_shape(&path)?;
                changed_any = true;
            }
            if let Some(e) = self.sets.get_mut(&path) {
                e.resized = false;
                if let Store::Chunked { changed, .. } = &mut e.store {
                    *changed = false;
                }
            }
        }
        if changed_any || self.end > self.file_eof()? {
            self.write_eof()?;
        }
        self.out.flush().map_err(|e| Error::Io(e.to_string()))?;
        self.file = File::open(&self.path)?;
        // Positions and indexes are re-read from the new file when next needed.
        self.sets.clear();
        Ok(())
    }

    /// [`flush`](Self::flush) and close.
    ///
    /// # Errors
    /// As for [`flush`](Self::flush).
    pub fn close(mut self) -> Result<(), Error> {
        self.flush()
    }

    // ----- datasets -----

    fn edited(&mut self, path: &str) -> Result<&mut Edited, Error> {
        if !self.sets.contains_key(path) {
            let e = self.load(path)?;
            self.sets.insert(path.to_owned(), e);
        }
        self.sets
            .get_mut(path)
            .ok_or_else(|| Error::NotFound(path.to_owned()))
    }

    fn load(&self, path: &str) -> Result<Edited, Error> {
        let ds = match self.file.root()?.get(path)? {
            Node::Dataset(d) => *d,
            _ => return Err(Error::NotFound(path.to_owned())),
        };
        let esize = ds.dtype.element_size(self.file.sizes.offset);
        let fill = ds.fill_bytes(esize);
        let shape = ds.shape();
        let layout = ds
            .msgs
            .iter()
            .find(|m| m.kind == kind::LAYOUT)
            .ok_or_else(|| bad(format!("{path}: no layout")))?;
        let version = layout.data.first().copied().unwrap_or(0);
        let unsupported = |what: &str| {
            Error::Unsupported(format!(
                "{path}: changing a dataset {what} (written with libver=\"latest\"?)"
            ))
        };
        let store = match &ds.layout {
            Layout::Compact(data) => {
                if version != 3 && version != 4 && version != 5 {
                    return Err(unsupported("with a version 1 or 2 compact layout"));
                }
                Store::Compact {
                    at: layout.at.saturating_add(4),
                    len: data.len(),
                }
            }
            Layout::Contiguous { addr, size } => Store::Contiguous {
                addr: *addr,
                size: *size,
                addr_at: layout.at.saturating_add(if version < 3 { 8 } else { 2 }),
            },
            Layout::Chunked { addr, chunk, index } => {
                let rank = shape.len();
                let cdims: Vec<usize> = chunk
                    .iter()
                    .take(rank)
                    .map(|c| usize::try_from(*c).unwrap_or(usize::MAX))
                    .collect();
                let index = match index {
                    ChunkIndex::BTreeV1 => Index::BTree1 {
                        addr_at: layout.at.saturating_add(if version < 3 { 8 } else { 3 }),
                    },
                    ChunkIndex::Single { filtered_size, .. } => {
                        let end = layout.at.saturating_add(layout.data.len() as u64);
                        let addr_at = end.saturating_sub(8);
                        Index::Single {
                            addr_at,
                            filtered_at: filtered_size.map(|_| addr_at.saturating_sub(12)),
                        }
                    }
                    ChunkIndex::Implicit => {
                        return Err(unsupported("with an implicit chunk index"));
                    }
                    ChunkIndex::FixedArray => {
                        return Err(unsupported("indexed by a fixed array"));
                    }
                    ChunkIndex::ExtensibleArray => {
                        return Err(unsupported("indexed by an extensible array"));
                    }
                    ChunkIndex::BTreeV2 => {
                        return Err(unsupported("indexed by a version-2 B-tree"));
                    }
                };
                let mut chunks = BTreeMap::new();
                if let Some(a) = addr {
                    let cd: Vec<u64> = cdims.iter().map(|c| *c as u64).collect();
                    for c in ds.chunk_list(*a, &cd, index_of(&ds.layout), &shape, esize)? {
                        let size = u32::try_from(c.size).map_err(|_| bad("chunk too large"))?;
                        chunks.insert(c.offset, (c.addr, size, c.mask));
                    }
                }
                Store::Chunked {
                    cdims,
                    index,
                    chunks,
                    dirty: BTreeMap::new(),
                    changed: false,
                }
            }
            Layout::Virtual { .. } => {
                return Err(Error::Unsupported(format!(
                    "{path}: writing to a virtual dataset"
                )));
            }
        };
        Ok(Edited {
            ds,
            shape,
            resized: false,
            esize,
            fill,
            store,
        })
    }

    /// A chunk's decoded bytes: buffered, else from the file, else the fill value.
    fn chunk_bytes(&mut self, path: &str, origin: &[u64]) -> Result<Vec<u8>, Error> {
        let e = self
            .sets
            .get(path)
            .ok_or_else(|| Error::NotFound(path.to_owned()))?;
        let Store::Chunked {
            cdims,
            chunks,
            dirty,
            ..
        } = &e.store
        else {
            return Err(bad("not chunked"));
        };
        if let Some(d) = dirty.get(origin) {
            return Ok(d.clone());
        }
        let n = cdims
            .iter()
            .try_fold(Ck::from(1usize), |a, c| Some(a * *c))
            .and_then(Ck::usize)
            .ok_or_else(|| bad("chunk too large"))?;
        let expected = n.saturating_mul(e.esize);
        match chunks.get(origin) {
            None => Ok(crate::filled(&e.fill, n)),
            Some(&(addr, size, mask)) => {
                let filters = e.ds.filters.clone();
                let base = self.file.base;
                let raw = self.pread(base.saturating_add(addr), size as usize)?;
                let mut data = crate::filters::decode(&filters, mask, raw, expected)?;
                data.resize(expected, 0);
                Ok(data)
            }
        }
    }

    fn put_dirty(&mut self, path: &str, origin: Vec<u64>, data: Vec<u8>) -> Result<(), Error> {
        let e = self
            .sets
            .get_mut(path)
            .ok_or_else(|| Error::NotFound(path.to_owned()))?;
        if let Store::Chunked { dirty, .. } = &mut e.store {
            let len = data.len();
            if dirty.insert(origin, data).is_none() {
                self.cached = self.cached.saturating_add(len);
            }
        }
        Ok(())
    }

    /// Encode and append every buffered chunk.
    fn write_dirty(&mut self) -> Result<(), Error> {
        let paths: Vec<String> = self.sets.keys().cloned().collect();
        for path in paths {
            let (filters, esize, dirty) = match self.sets.get_mut(&path) {
                Some(Edited {
                    store: Store::Chunked { dirty, .. },
                    ds,
                    esize,
                    ..
                }) if !dirty.is_empty() => (ds.filters.clone(), *esize, std::mem::take(dirty)),
                _ => continue,
            };
            let mut placed = Vec::with_capacity(dirty.len());
            for (origin, data) in dirty {
                let stored = crate::filters::encode(&filters, data, esize)?;
                let size = u32::try_from(stored.len()).map_err(|_| bad("chunk too large"))?;
                let at = self.append(&stored)?;
                placed.push((origin, (at, size, 0u32)));
            }
            if let Some(Edited {
                store: Store::Chunked {
                    chunks, changed, ..
                },
                ..
            }) = self.sets.get_mut(&path)
            {
                for (origin, entry) in placed {
                    chunks.insert(origin, entry);
                }
                *changed = true;
            }
        }
        self.cached = 0;
        Ok(())
    }

    // ----- indexes and headers -----

    fn write_index(
        &mut self,
        path: &str,
        index: Index,
        chunks: &BTreeMap<Vec<u64>, (u64, u32, u32)>,
        cdims: &[usize],
        shape: &[usize],
    ) -> Result<(), Error> {
        match index {
            Index::BTree1 { addr_at } => {
                let root = self.write_btree(chunks, cdims, shape)?;
                self.pwrite(addr_at, &root.unwrap_or(UNDEF).to_le_bytes())?;
            }
            Index::Single {
                addr_at,
                filtered_at,
            } => {
                let (addr, size, mask) = match chunks.values().next() {
                    Some(c) if chunks.len() == 1 => *c,
                    None => (UNDEF, 0, 0),
                    Some(_) => {
                        return Err(bad(format!(
                            "{path}: several chunks in a single-chunk index"
                        )));
                    }
                };
                self.pwrite(addr_at, &addr.to_le_bytes())?;
                if let Some(at) = filtered_at {
                    let mut b = u64::from(size).to_le_bytes().to_vec();
                    b.extend_from_slice(&mask.to_le_bytes());
                    self.pwrite(at, &b)?;
                }
            }
        }
        let checksummed = self.layout_checksum(path)?;
        self.rechecksum(checksummed)
    }

    /// A version-1 B-tree (type 1, raw-data chunks) over `chunks`, its nodes as full as HDF5
    /// allows (`2K` entries); `None` without chunks.
    fn write_btree(
        &mut self,
        chunks: &BTreeMap<Vec<u64>, (u64, u32, u32)>,
        cdims: &[usize],
        shape: &[usize],
    ) -> Result<Option<u64>, Error> {
        if chunks.is_empty() {
            return Ok(None);
        }
        let rank = cdims.len();
        let width = usize::from(self.file.superblock.istore_k.max(1)).saturating_mul(2);
        let key_size = 8usize.saturating_add(8usize.saturating_mul(rank.saturating_add(1)));
        let node_size = 24usize
            .saturating_add(width.saturating_mul(key_size.saturating_add(8)))
            .saturating_add(key_size);
        // The extent rounded up to whole chunks: the last node's upper bound.
        let bound: Vec<u64> = shape
            .iter()
            .zip(cdims)
            .map(|(d, c)| d.div_ceil((*c).max(1)).saturating_mul(*c) as u64)
            .collect();
        // Entries: (first element, chunk size, mask, child address).
        let mut level_entries: Vec<Entry> = chunks
            .iter()
            .map(|(o, (a, s, m))| (o.clone(), *s, *m, *a))
            .collect();
        let mut level: u8 = 0;
        loop {
            let groups: Vec<&[Entry]> = level_entries.chunks(width).collect();
            let first = self.end;
            let at = |i: usize| -> u64 {
                first
                    .saturating_sub(self.file.base)
                    .saturating_add((i as u64).saturating_mul(node_size as u64))
            };
            let mut parents = Vec::with_capacity(groups.len());
            let mut bytes = Vec::with_capacity(groups.len().saturating_mul(node_size));
            for (i, group) in groups.iter().enumerate() {
                let mut b = Vec::with_capacity(node_size);
                b.extend_from_slice(b"TREE");
                b.push(1);
                b.push(level);
                b.extend_from_slice(&u16::try_from(group.len()).unwrap_or(0).to_le_bytes());
                let left = if i == 0 {
                    UNDEF
                } else {
                    at(i.saturating_sub(1))
                };
                let right = if i.saturating_add(1) == groups.len() {
                    UNDEF
                } else {
                    at(i.saturating_add(1))
                };
                b.extend_from_slice(&left.to_le_bytes());
                b.extend_from_slice(&right.to_le_bytes());
                let key = |b: &mut Vec<u8>, size: u32, mask: u32, off: &[u64]| {
                    b.extend_from_slice(&size.to_le_bytes());
                    b.extend_from_slice(&mask.to_le_bytes());
                    for o in off {
                        b.extend_from_slice(&o.to_le_bytes());
                    }
                    b.extend_from_slice(&0u64.to_le_bytes());
                };
                for (off, size, mask, child) in group.iter() {
                    key(&mut b, *size, *mask, off);
                    b.extend_from_slice(&child.to_le_bytes());
                }
                // The upper bound: the next node's first key, or the extent's.
                let upper = groups
                    .get(i.saturating_add(1))
                    .and_then(|g| g.first())
                    .map_or_else(|| bound.clone(), |e| e.0.clone());
                key(&mut b, 0, 0, &upper);
                b.resize(node_size, 0);
                bytes.extend_from_slice(&b);
                if let Some(f) = group.first() {
                    parents.push((f.0.clone(), f.1, f.2, at(i)));
                }
            }
            self.append(&bytes)?;
            if parents.len() <= 1 {
                return Ok(parents.first().map(|p| p.3));
            }
            level_entries = parents;
            level = level.saturating_add(1);
        }
    }

    /// Rewrite a dataset's dataspace with its new current dimensions.
    fn write_shape(&mut self, path: &str) -> Result<(), Error> {
        let e = self
            .sets
            .get(path)
            .ok_or_else(|| Error::NotFound(path.to_owned()))?;
        let msg =
            e.ds.msgs
                .iter()
                .find(|m| m.kind == kind::DATASPACE)
                .ok_or_else(|| bad(format!("{path}: no dataspace")))?;
        let dims_at = match msg.data.first() {
            Some(1) => 8,
            Some(2) => 4,
            _ => return Err(bad(format!("{path}: dataspace version"))),
        };
        let mut b = Vec::with_capacity(e.shape.len().saturating_mul(8));
        for d in &e.shape {
            b.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        let (at, checksummed) = (msg.at.saturating_add(dims_at), msg.checksummed);
        self.pwrite(at, &b)?;
        self.rechecksum(checksummed)
    }

    fn layout_checksum(&self, path: &str) -> Result<Option<(u64, u64)>, Error> {
        let e = self
            .sets
            .get(path)
            .ok_or_else(|| Error::NotFound(path.to_owned()))?;
        Ok(e.ds
            .msgs
            .iter()
            .find(|m| m.kind == kind::LAYOUT)
            .and_then(|m| m.checksummed))
    }

    /// Recompute a version-2 object header chunk's checksum after a change inside it.
    fn rechecksum(&mut self, checksummed: Option<(u64, u64)>) -> Result<(), Error> {
        if let Some((start, len)) = checksummed {
            let len = usize::try_from(len).map_err(|_| bad("header too large"))?;
            let data = self.pread(start, len)?;
            let sum = lookup3(&data, 0);
            self.pwrite(start.saturating_add(len as u64), &sum.to_le_bytes())?;
        }
        Ok(())
    }

    fn file_eof(&self) -> Result<u64, Error> {
        let sb = &self.file.superblock;
        let at = sb.eof_at as u64;
        let mut b = [0u8; 8];
        let mut f = &self.out;
        f.seek(SeekFrom::Start(at))
            .map_err(|e| Error::Io(e.to_string()))?;
        f.read_exact(&mut b).map_err(|e| Error::Io(e.to_string()))?;
        Ok(u64::from_le_bytes(b).saturating_add(sb.base))
    }

    /// Record the end of the file in the superblock.
    fn write_eof(&mut self) -> Result<(), Error> {
        let sb = self.file.superblock;
        let eof = self.end.saturating_sub(sb.base);
        self.pwrite(sb.eof_at as u64, &eof.to_le_bytes())?;
        if let Some((start, len)) = sb.checksummed {
            self.rechecksum(Some((start as u64, len as u64)))?;
        }
        Ok(())
    }

    // ----- raw I/O -----

    fn pread(&self, at: u64, len: usize) -> Result<Vec<u8>, Error> {
        let mut b = vec![0u8; len];
        let mut f = &self.out;
        f.seek(SeekFrom::Start(at))
            .map_err(|e| Error::Io(e.to_string()))?;
        f.read_exact(&mut b)
            .map_err(|e| Error::Io(format!("{}: {e}", self.path.display())))?;
        Ok(b)
    }

    fn pwrite(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        self.out
            .seek(SeekFrom::Start(at))
            .map_err(|e| Error::Io(e.to_string()))?;
        self.out
            .write_all(bytes)
            .map_err(|e| Error::Io(format!("{}: {e}", self.path.display())))?;
        let end = at.saturating_add(bytes.len() as u64);
        self.end = self.end.max(end);
        Ok(())
    }

    /// Write at the end of the file; the HDF5 address.
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let at = self.end;
        self.pwrite(at, bytes)?;
        Ok(at.saturating_sub(self.file.base))
    }

    /// `size` bytes of fill values at the end of the file; the HDF5 address.
    fn allocate_filled(&mut self, size: u64, fill: &[u8]) -> Result<u64, Error> {
        let at = self.end;
        let size = usize::try_from(size).map_err(|_| bad("dataset too large"))?;
        let block_elems = (1usize << 20).checked_div(fill.len()).unwrap_or(1);
        let block = crate::filled(fill, block_elems.max(1));
        let mut written = 0usize;
        while written < size {
            let n = block.len().min(size.saturating_sub(written));
            self.pwrite(
                at.saturating_add(written as u64),
                block.get(..n).unwrap_or_default(),
            )?;
            written = written.saturating_add(n);
        }
        Ok(at.saturating_sub(self.file.base))
    }

    /// Write the box `start .. start + count` (bytes in C order) of contiguous data at HDF5
    /// address `addr`, row by row.
    fn write_rows(
        &mut self,
        addr: u64,
        shape: &[usize],
        start: &[usize],
        count: &[usize],
        esize: usize,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let base = self.file.base.saturating_add(addr);
        for (src, dst, len) in rows(shape, start, count, esize)? {
            let piece = bytes
                .get(src..src.saturating_add(len))
                .ok_or_else(|| bad("values shorter than their shape"))?;
            self.pwrite(base.saturating_add(dst as u64), piece)?;
        }
        Ok(())
    }

    fn read_rows(
        &self,
        addr: u64,
        shape: &[usize],
        start: &[usize],
        count: &[usize],
        esize: usize,
        out: &mut [u8],
    ) -> Result<(), Error> {
        let base = self.file.base.saturating_add(addr);
        for (dst, src, len) in rows(shape, start, count, esize)? {
            let piece = self.pread(base.saturating_add(src as u64), len)?;
            out.get_mut(dst..dst.saturating_add(len))
                .ok_or_else(|| bad("box outside its buffer"))?
                .copy_from_slice(&piece);
        }
        Ok(())
    }
}

/// A B-tree entry: first element, chunk size, filter mask, child address.
type Entry = (Vec<u64>, u32, u32, u64);

fn index_of(layout: &Layout) -> &ChunkIndex {
    match layout {
        Layout::Chunked { index, .. } => index,
        _ => &ChunkIndex::BTreeV1,
    }
}

fn check_box(path: &str, shape: &[usize], start: &[usize], count: &[usize]) -> Result<(), Error> {
    if start.len() != shape.len() || count.len() != shape.len() {
        return Err(bad(format!(
            "{path}: a box of rank {} for shape {shape:?}",
            start.len()
        )));
    }
    if start
        .iter()
        .zip(count)
        .zip(shape)
        .any(|((s, c), d)| s.checked_add(*c).is_none_or(|e| e > *d))
    {
        return Err(bad(format!(
            "{path}: {start:?} + {count:?} is outside shape {shape:?}"
        )));
    }
    Ok(())
}

/// The bytes to store for `values` in a dataset of type `dtype`.
fn element_bytes(path: &str, dtype: &Datatype, values: &Array) -> Result<Vec<u8>, Error> {
    let (big_endian, half) = match dtype {
        Datatype::Int { big_endian, .. } | Datatype::PackedInt { big_endian, .. } => {
            (*big_endian, false)
        }
        Datatype::Float { size, big_endian } => (*big_endian, *size == 2),
        Datatype::Enum { base, .. } => match base.as_ref() {
            Datatype::Int { big_endian, .. } => (*big_endian, false),
            _ => (false, false),
        },
        Datatype::FixedString { size: 1, .. } => {
            return match values {
                Array::U8(a) => Ok(a.iter().copied().collect()),
                _ => Err(bad(format!("{path}: characters are written as uint8"))),
            };
        }
        other => {
            return Err(Error::Unsupported(format!(
                "{path}: writing values of type {other:?} in place"
            )));
        }
    };
    let want = dtype.value_dtype();
    if want != Some(values.dtype()) {
        return Err(bad(format!(
            "{path}: values of type {} for a dataset of {}",
            values.dtype(),
            want.map_or_else(|| "another type".to_owned(), |t| t.to_string())
        )));
    }
    crate::writer::file_bytes(values, big_endian, half)
        .ok_or_else(|| Error::Unsupported(format!("{path}: writing {} data", values.dtype())))
}

/// The first element of every chunk (of extents `cdims`) the box touches, in row-major order.
fn chunks_in(cdims: &[usize], start: &[usize], count: &[usize]) -> Vec<Vec<u64>> {
    let ranges: Vec<(usize, usize)> = start
        .iter()
        .zip(count)
        .zip(cdims)
        .map(|((s, c), d)| {
            let d = (*d).max(1);
            (
                s.checked_div(d).unwrap_or(0),
                s.saturating_add(*c).div_ceil(d),
            )
        })
        .collect();
    let mut out = Vec::new();
    let mut idx: Vec<usize> = ranges.iter().map(|r| r.0).collect();
    if ranges.iter().any(|(a, b)| a >= b) {
        return out;
    }
    loop {
        out.push(
            idx.iter()
                .zip(cdims)
                .map(|(i, d)| i.saturating_mul(*d) as u64)
                .collect(),
        );
        // Next index, last axis fastest.
        let mut k = idx.len();
        loop {
            if k == 0 {
                return out;
            }
            k = k.saturating_sub(1);
            if let (Some(i), Some(r)) = (idx.get_mut(k), ranges.get(k)) {
                *i = i.saturating_add(1);
                if *i < r.1 {
                    break;
                }
                *i = r.0;
            }
        }
    }
}

/// Copy where two boxes overlap: `src` (extents `sdims`, first element `sorigin`) into
/// `dst` (extents `ddims`, first element `dorigin`), both in global coordinates.
fn copy_between(
    src: &[u8],
    sdims: &[usize],
    sorigin: &[usize],
    dst: &mut [u8],
    ddims: &[usize],
    dorigin: &[usize],
    esize: usize,
) -> Result<(), Error> {
    let mut from = Vec::with_capacity(sdims.len());
    let mut to = Vec::with_capacity(sdims.len());
    for k in 0..sdims.len() {
        let s0 = sorigin.get(k).copied().unwrap_or(0);
        let d0 = dorigin.get(k).copied().unwrap_or(0);
        let lo = s0.max(d0);
        let hi = s0
            .saturating_add(sdims.get(k).copied().unwrap_or(0))
            .min(d0.saturating_add(ddims.get(k).copied().unwrap_or(0)));
        if lo >= hi {
            return Ok(());
        }
        from.push(lo);
        to.push(hi);
    }
    crate::copy_box(src, sdims, sorigin, dst, ddims, dorigin, &from, &to, esize)
}

/// Write the box `start .. start + count` (bytes in C order) into `data` of `shape`.
fn copy_into(
    bytes: &[u8],
    count: &[usize],
    start: &[usize],
    data: &mut [u8],
    shape: &[usize],
    esize: usize,
) -> Result<(), Error> {
    copy_between(
        bytes,
        count,
        start,
        data,
        shape,
        &vec![0; shape.len()],
        esize,
    )
}

fn copy_out(
    data: &[u8],
    shape: &[usize],
    out: &mut [u8],
    start: &[usize],
    count: &[usize],
    esize: usize,
) -> Result<(), Error> {
    copy_between(data, shape, &vec![0; shape.len()], out, count, start, esize)
}

/// The rows (runs along the last axis) of the box in C-order storage of `shape`:
/// `(offset in the box's bytes, offset in the storage, length)`.
fn rows(
    shape: &[usize],
    start: &[usize],
    count: &[usize],
    esize: usize,
) -> Result<Vec<(usize, usize, usize)>, Error> {
    let rank = shape.len();
    if rank == 0 {
        return Ok(vec![(0, 0, esize)]);
    }
    let last = rank.saturating_sub(1);
    let run = count
        .get(last)
        .copied()
        .unwrap_or(0)
        .checked_mul(esize)
        .ok_or_else(|| bad("box too large"))?;
    let outer: Vec<u64> = count.iter().take(last).map(|c| *c as u64).collect();
    let nrows = outer
        .iter()
        .try_fold(1u64, |a, c| a.checked_mul(*c))
        .ok_or_else(|| bad("box too large"))?;
    let mut out = Vec::new();
    for r in 0..nrows {
        let idx = crate::unravel(r, &outer).unwrap_or_default();
        let mut flat = Ck::from(0usize);
        for k in 0..rank {
            let i = idx.get(k).map_or(0, |x| *x as usize);
            let g = start.get(k).copied().unwrap_or(0).saturating_add(i);
            flat = flat * shape.get(k).copied().unwrap_or(0) + g;
        }
        let dst = (flat * esize)
            .usize()
            .ok_or_else(|| bad("offset overflow"))?;
        let src = (Ck::from(r as usize) * run)
            .usize()
            .ok_or_else(|| bad("offset overflow"))?;
        out.push((src, dst, run));
    }
    Ok(out)
}
