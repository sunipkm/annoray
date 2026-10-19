//! HDF5 indexing structures: local and global heaps, v1 and v2 B-trees, fractal heaps, and
//! the fixed-array and extensible-array chunk indexes.

use crate::Error;

use crate::ck::Ck;
use crate::cursor::{Cursor, enc_size};
use crate::{File, bad};

/// Recursion limit for tree traversals (guards against cycles in corrupt files).
const MAX_DEPTH: usize = 64;

fn log2_floor(n: u64) -> u32 {
    63u32.saturating_sub(n.leading_zeros())
}

// ----- local heap (old-style group names) -----

/// The data segment of a local heap.
pub(crate) fn local_heap(file: &File, addr: u64) -> Result<Vec<u8>, Error> {
    let mut c = file.cursor(addr)?;
    c.expect(b"HEAP")?;
    c.skip(4)?; // version, reserved
    let size = c.len_usize()?;
    c.length()?; // free list
    let data = c.addr()?.ok_or_else(|| bad("undefined local heap data"))?;
    Ok(file.cursor(data)?.bytes(size)?.to_vec())
}

fn heap_str(seg: &[u8], off: u64) -> Result<String, Error> {
    let off = usize::try_from(off).map_err(|_| bad("bad heap offset"))?;
    let rest = seg.get(off..).ok_or_else(|| bad("heap offset past end"))?;
    let end = rest.iter().position(|b| *b == 0).unwrap_or(rest.len());
    Ok(String::from_utf8_lossy(rest.get(..end).unwrap_or_default()).into_owned())
}

// ----- v1 B-trees -----

/// Every `(name, target)` of an old-style group: object header addresses, or soft links.
pub(crate) fn group_btree(
    file: &File,
    addr: u64,
    heap: &[u8],
) -> Result<Vec<(String, crate::format::Target)>, Error> {
    let mut out = Vec::new();
    group_node(file, addr, heap, &mut out, 0)?;
    Ok(out)
}

fn group_node(
    file: &File,
    addr: u64,
    heap: &[u8],
    out: &mut Vec<(String, crate::format::Target)>,
    depth: usize,
) -> Result<(), Error> {
    if depth > MAX_DEPTH {
        return Err(bad("group B-tree too deep"));
    }
    let mut c = file.cursor(addr)?;
    c.expect(b"TREE")?;
    if c.u8()? != 0 {
        return Err(bad("expected a group B-tree node"));
    }
    let level = c.u8()?;
    let n = usize::from(c.u16()?);
    c.addr()?;
    c.addr()?;
    let mut children = Vec::with_capacity(n);
    for _ in 0..n {
        c.length()?; // key
        children.push(c.addr()?.ok_or_else(|| bad("undefined B-tree child"))?);
    }
    for child in children {
        if level > 0 {
            group_node(file, child, heap, out, depth.saturating_add(1))?;
        } else {
            let mut s = file.cursor(child)?;
            s.expect(b"SNOD")?;
            s.skip(2)?;
            let m = s.u16()?;
            for _ in 0..m {
                let name_off = s.length()?;
                let obj = s.addr()?;
                let cache = s.u32()?;
                s.skip(4)?; // reserved
                // Scratch pad (16 bytes): for a soft link, the link value's heap offset.
                let soft = s.u32()?;
                s.skip(12)?;
                let target = match (cache, obj) {
                    (2, _) => crate::format::Target::Soft(heap_str(heap, u64::from(soft))?),
                    (_, Some(a)) => crate::format::Target::Hard(a),
                    (_, None) => crate::format::Target::Unknown(0),
                };
                out.push((heap_str(heap, name_off)?, target));
            }
        }
    }
    Ok(())
}

/// One stored chunk.
#[derive(Debug, Clone)]
pub(crate) struct ChunkRef {
    /// Offset of the chunk's first element, per dimension.
    pub(crate) offset: Vec<u64>,
    pub(crate) addr: u64,
    /// Stored (possibly compressed) size in bytes.
    pub(crate) size: u64,
    /// Filters skipped for this chunk (bit i = filter i).
    pub(crate) mask: u32,
}

/// Every chunk of a v1 chunk B-tree.
pub(crate) fn chunk_btree(file: &File, addr: u64, rank: usize) -> Result<Vec<ChunkRef>, Error> {
    let mut out = Vec::new();
    chunk_node(file, addr, rank, &mut out, 0)?;
    Ok(out)
}

fn chunk_node(
    file: &File,
    addr: u64,
    rank: usize,
    out: &mut Vec<ChunkRef>,
    depth: usize,
) -> Result<(), Error> {
    if depth > MAX_DEPTH {
        return Err(bad("chunk B-tree too deep"));
    }
    let mut c = file.cursor(addr)?;
    c.expect(b"TREE")?;
    if c.u8()? != 1 {
        return Err(bad("expected a chunk B-tree node"));
    }
    let level = c.u8()?;
    let n = usize::from(c.u16()?);
    c.addr()?;
    c.addr()?;
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let size = u64::from(c.u32()?);
        let mask = c.u32()?;
        let offs = (0..=rank).map(|_| c.u64()).collect::<Result<Vec<_>, _>>()?;
        let child = c.addr()?.ok_or_else(|| bad("undefined chunk"))?;
        entries.push((size, mask, offs, child));
    }
    for (size, mask, mut offs, child) in entries {
        if level > 0 {
            chunk_node(file, child, rank, out, depth.saturating_add(1))?;
        } else {
            offs.truncate(rank);
            out.push(ChunkRef {
                offset: offs,
                addr: child,
                size,
                mask,
            });
        }
    }
    Ok(())
}

// ----- v2 B-trees -----

/// All records of a v2 B-tree: `(type, records)`.
pub(crate) fn btree2(file: &File, addr: u64) -> Result<(u8, Vec<Vec<u8>>), Error> {
    let mut c = file.cursor(addr)?;
    c.expect(b"BTHD")?;
    c.skip(1)?;
    let typ = c.u8()?;
    let node_size = u64::from(c.u32()?);
    let rec_size = usize::from(c.u16()?);
    let depth = usize::from(c.u16()?);
    c.skip(2)?;
    let root = c.addr()?;
    let root_nrec = usize::from(c.u16()?);
    if rec_size == 0 || depth > MAX_DEPTH {
        return Err(bad("bad v2 B-tree header"));
    }
    // Field widths, as HDF5 computes them (H5B2__hdr_init).
    let so = u64::from(file.sizes.offset);
    let rec = u64::try_from(rec_size).map_err(|_| bad("bad record size"))?;
    let leaf_max = (Ck::from(node_size) - 10u64) / rec;
    let leaf_max: u64 = leaf_max
        .to()
        .ok_or_else(|| bad("bad v2 B-tree node size"))?;
    let max_nrec_size = enc_size(leaf_max);
    let mut cum_max = vec![leaf_max];
    let mut cum_size = vec![0usize];
    for d in 1..=depth {
        let prev_size = if d > 1 {
            cum_size.get(d.saturating_sub(1)).copied().unwrap_or(0)
        } else {
            0
        };
        let ptr = Ck::from(so) + max_nrec_size + prev_size;
        let max_nrec: u64 = ((Ck::from(node_size) - (Ck::from(10u64) + ptr))
            / (Ck::from(rec) + ptr))
            .to()
            .ok_or_else(|| bad("bad v2 B-tree"))?;
        let prev = cum_max.get(d.saturating_sub(1)).copied().unwrap_or(0);
        let cum = (Ck::from(max_nrec) + 1u64) * prev + max_nrec;
        let cum: u64 = cum.to().unwrap_or(u64::MAX);
        cum_max.push(cum);
        cum_size.push(enc_size(cum));
    }
    let mut out = Vec::new();
    if let Some(root) = root {
        bt2_node(
            file,
            root,
            root_nrec,
            depth,
            rec_size,
            max_nrec_size,
            &cum_size,
            &mut out,
            0,
        )?;
    }
    Ok((typ, out))
}

#[allow(clippy::too_many_arguments)]
fn bt2_node(
    file: &File,
    addr: u64,
    nrec: usize,
    depth: usize,
    rec_size: usize,
    max_nrec_size: usize,
    cum_size: &[usize],
    out: &mut Vec<Vec<u8>>,
    guard: usize,
) -> Result<(), Error> {
    if guard > MAX_DEPTH {
        return Err(bad("v2 B-tree too deep"));
    }
    let mut c = file.cursor(addr)?;
    c.expect(if depth == 0 { b"BTLF" } else { b"BTIN" })?;
    c.skip(2)?;
    let records = (0..nrec)
        .map(|_| c.bytes(rec_size).map(<[u8]>::to_vec))
        .collect::<Result<Vec<_>, _>>()?;
    out.extend(records);
    if depth > 0 {
        let total_size = if depth > 1 {
            cum_size.get(depth.saturating_sub(1)).copied().unwrap_or(0)
        } else {
            0
        };
        let mut children = Vec::new();
        for _ in 0..=nrec {
            let child = c.addr()?.ok_or_else(|| bad("undefined v2 B-tree child"))?;
            let n = usize::try_from(c.uvar(max_nrec_size)?).map_err(|_| bad("bad record count"))?;
            if total_size > 0 {
                c.uvar(total_size)?;
            }
            children.push((child, n));
        }
        for (child, n) in children {
            bt2_node(
                file,
                child,
                n,
                depth.saturating_sub(1),
                rec_size,
                max_nrec_size,
                cum_size,
                out,
                guard.saturating_add(1),
            )?;
        }
    }
    Ok(())
}

// ----- fractal heap -----

/// A fractal heap (dense link and attribute storage).
pub(crate) struct FractalHeap {
    width: u64,
    start_block: u64,
    max_direct_rows: u64,
    first_row_bits: u32,
    heap_off_size: usize,
    heap_len_size: usize,
    root: Option<u64>,
    root_rows: u64,
    filtered: bool,
    id_len: usize,
    huge_btree: Option<u64>,
}

impl FractalHeap {
    pub(crate) fn open(file: &File, addr: u64) -> Result<Self, Error> {
        let mut c = file.cursor(addr)?;
        c.expect(b"FRHP")?;
        c.skip(1)?;
        let id_len = usize::from(c.u16()?);
        let filter_len = c.u16()?;
        c.skip(1)?; // flags
        let max_managed = u64::from(c.u32()?);
        c.length()?; // next huge id
        let huge_btree = c.addr()?;
        c.length()?; // free space
        c.addr()?; // free space manager
        for _ in 0..8 {
            c.length()?; // managed/huge/tiny statistics
        }
        let width = u64::from(c.u16()?);
        let start_block = c.length()?;
        let max_direct = c.length()?;
        let max_heap_bits = u64::from(c.u16()?);
        c.skip(2)?; // starting rows
        let root = c.addr()?;
        let root_rows = u64::from(c.u16()?);
        if width == 0
            || start_block == 0
            || max_direct < start_block
            || !width.is_power_of_two()
            || !start_block.is_power_of_two()
        {
            return Err(bad("bad fractal heap geometry"));
        }
        let max_direct_rows =
            u64::from(log2_floor(max_direct).saturating_sub(log2_floor(start_block)))
                .saturating_add(2);
        let heap_off_size =
            usize::try_from(max_heap_bits.saturating_add(7) / 8).map_err(|_| bad("bad heap"))?;
        let dir_off_size = usize::try_from(u64::from(log2_floor(max_direct)).saturating_add(7) / 8)
            .map_err(|_| bad("bad heap"))?;
        let heap_len_size = dir_off_size.min(enc_size(max_managed));
        Ok(Self {
            width,
            start_block,
            max_direct_rows,
            first_row_bits: log2_floor(start_block).saturating_add(log2_floor(width)),
            heap_off_size,
            heap_len_size,
            root,
            root_rows,
            filtered: filter_len > 0,
            id_len,
            huge_btree,
        })
    }

    fn row_size(&self, r: u64) -> Option<u64> {
        if r <= 1 {
            Some(self.start_block)
        } else {
            self.start_block
                .checked_shl(u32::try_from(r.saturating_sub(1)).ok()?)
        }
    }

    /// The bytes of the object with heap ID `id`.
    pub(crate) fn get(&self, file: &File, id: &[u8]) -> Result<Vec<u8>, Error> {
        let mut c = Cursor::new(id, 0, file.sizes);
        let flags = c.u8()?;
        match (flags >> 4) & 3 {
            0 => {
                if self.filtered {
                    return Err(bad("filtered fractal heaps are not supported"));
                }
                let off = c.uvar(self.heap_off_size)?;
                let len = usize::try_from(c.uvar(self.heap_len_size)?)
                    .map_err(|_| bad("object too large"))?;
                let at = self.locate(file, off)?;
                Ok(file.cursor(at)?.bytes(len)?.to_vec())
            }
            2 => {
                let len = usize::from(flags & 0x0F).saturating_add(1);
                Ok(c.bytes(len)?.to_vec())
            }
            1 => self.huge(file, &mut c),
            _ => Err(bad("bad fractal heap ID")),
        }
    }

    /// A huge object: stored outside the heap's blocks, at an address the heap ID holds
    /// (when it is long enough) or a v2 B-tree maps the ID's key to (H5HF__huge_init).
    fn huge(&self, file: &File, c: &mut Cursor<'_>) -> Result<Vec<u8>, Error> {
        if self.filtered {
            return Err(Error::Unsupported("filtered fractal heap objects".into()));
        }
        let so = usize::from(file.sizes.offset);
        let sl = usize::from(file.sizes.length);
        let (addr, len) = if self.id_len >= 1usize.saturating_add(so).saturating_add(sl) {
            (c.addr()?, c.length()?)
        } else {
            let key_len = self.id_len.saturating_sub(1).min(sl);
            let key = c.uvar(key_len)?;
            let tree = self
                .huge_btree
                .ok_or_else(|| bad("huge object without a B-tree"))?;
            let (typ, records) = btree2(file, tree)?;
            if typ != 1 {
                return Err(bad(format!("huge object B-tree of type {typ}")));
            }
            let mut found = None;
            for r in &records {
                let mut rc = Cursor::new(r, 0, file.sizes);
                let addr = rc.addr()?;
                let len = rc.length()?;
                if rc.length()? == key {
                    found = Some((addr, len));
                    break;
                }
            }
            found.ok_or_else(|| bad("huge object not in its B-tree"))?
        };
        let addr = addr.ok_or_else(|| bad("huge object at an undefined address"))?;
        let len = usize::try_from(len).map_err(|_| bad("object too large"))?;
        Ok(file.cursor(addr)?.bytes(len)?.to_vec())
    }

    fn locate(&self, file: &File, off: u64) -> Result<u64, Error> {
        let root = self.root.ok_or_else(|| bad("empty fractal heap"))?;
        if self.root_rows == 0 {
            return root.checked_add(off).ok_or_else(|| bad("bad heap offset"));
        }
        self.find(file, root, self.root_rows, 0, off, 0)
    }

    fn find(
        &self,
        file: &File,
        iblock: u64,
        nrows: u64,
        base: u64,
        off: u64,
        depth: usize,
    ) -> Result<u64, Error> {
        if depth > MAX_DEPTH {
            return Err(bad("fractal heap too deep"));
        }
        let mut c = file.cursor(iblock)?;
        c.expect(b"FHIB")?;
        c.skip(1)?;
        c.addr()?;
        c.uvar(self.heap_off_size)?;
        let mut cur = base;
        for r in 0..nrows {
            let size = self.row_size(r).ok_or_else(|| bad("bad heap row"))?;
            let direct = r < self.max_direct_rows;
            for _ in 0..self.width {
                let child = c.addr()?;
                let end = cur
                    .checked_add(size)
                    .ok_or_else(|| bad("bad heap offset"))?;
                if off >= cur && off < end {
                    let child = child.ok_or_else(|| bad("heap object in an unallocated block"))?;
                    return if direct {
                        child
                            .checked_add(off.saturating_sub(cur))
                            .ok_or_else(|| bad("bad heap offset"))
                    } else {
                        let rows = u64::from(log2_floor(size).saturating_sub(self.first_row_bits))
                            .saturating_add(1);
                        self.find(file, child, rows, cur, off, depth.saturating_add(1))
                    };
                }
                cur = end;
            }
        }
        Err(bad("heap offset not found"))
    }
}

// ----- fixed and extensible arrays (version 4 chunk indexes) -----

/// One chunk-index element: address (undefined = not allocated), stored size, filter mask.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Element {
    pub(crate) addr: Option<u64>,
    pub(crate) size: Option<u64>,
    pub(crate) mask: u32,
}

fn element(c: &mut Cursor<'_>, filtered: bool, esize: usize, so: usize) -> Result<Element, Error> {
    let addr = c.addr()?;
    if filtered {
        let size_len = esize
            .checked_sub(so)
            .and_then(|x| x.checked_sub(4))
            .ok_or_else(|| bad("bad element size"))?;
        let size = c.uvar(size_len)?;
        let mask = c.u32()?;
        Ok(Element {
            addr,
            size: Some(size),
            mask,
        })
    } else {
        Ok(Element {
            addr,
            size: None,
            mask: 0,
        })
    }
}

fn bit(bitmap: &[u8], i: u64) -> bool {
    let byte = usize::try_from(i / 8)
        .ok()
        .and_then(|b| bitmap.get(b))
        .copied()
        .unwrap_or(0);
    byte & (0x80u8 >> (i % 8)) != 0
}

/// All elements of a fixed array (index = linear chunk number).
pub(crate) fn fixed_array(file: &File, addr: u64) -> Result<Vec<Element>, Error> {
    let so = usize::from(file.sizes.offset);
    let mut c = file.cursor(addr)?;
    c.expect(b"FAHD")?;
    c.skip(1)?;
    let filtered = c.u8()? == 1;
    let esize = usize::from(c.u8()?);
    let page_bits = u32::from(c.u8()?);
    let n = c.length()?;
    let dblock = c.addr()?;
    let undefined = Element {
        addr: None,
        size: None,
        mask: 0,
    };
    let n_us = usize::try_from(n).map_err(|_| bad("fixed array too large"))?;
    let Some(dblock) = dblock else {
        return Ok(vec![undefined; n_us]);
    };
    let mut c = file.cursor(dblock)?;
    c.expect(b"FADB")?;
    c.skip(2)?;
    c.addr()?;
    let page = 1u64.checked_shl(page_bits).unwrap_or(u64::MAX);
    let mut out = Vec::with_capacity(n_us.min(1 << 24));
    if n <= page {
        for _ in 0..n {
            out.push(element(&mut c, filtered, esize, so)?);
        }
        return Ok(out);
    }
    let npages = n.div_ceil(page);
    let bitmap = c
        .bytes(usize::try_from(npages.div_ceil(8)).map_err(|_| bad("bad bitmap"))?)?
        .to_vec();
    c.skip(4)?; // checksum
    for p in 0..npages {
        let count = page.min(n.saturating_sub(p.saturating_mul(page)));
        if bit(&bitmap, p) {
            for _ in 0..count {
                out.push(element(&mut c, filtered, esize, so)?);
            }
            c.skip(4)?;
        } else {
            for _ in 0..count {
                out.push(undefined);
            }
            // Uninitialised pages still occupy space in the data block.
            c.skip(
                usize::try_from(count)
                    .ok()
                    .and_then(|k| k.checked_mul(esize))
                    .and_then(|k| k.checked_add(4))
                    .ok_or_else(|| bad("bad page"))?,
            )?;
        }
    }
    Ok(out)
}

/// Elements of an extensible array, up to `n` (index = linear chunk number).
pub(crate) fn extensible_array(file: &File, addr: u64, n: u64) -> Result<Vec<Element>, Error> {
    let so = usize::from(file.sizes.offset);
    let mut c = file.cursor(addr)?;
    c.expect(b"EAHD")?;
    c.skip(1)?;
    let filtered = c.u8()? == 1;
    let esize = usize::from(c.u8()?);
    let max_bits = u32::from(c.u8()?);
    let iblock_elmts = u64::from(c.u8()?);
    let dblk_min = u64::from(c.u8()?);
    let sblk_min_ptrs = u64::from(c.u8()?);
    let page_bits = u32::from(c.u8()?);
    for _ in 0..6 {
        c.length()?; // statistics
    }
    let iblock = c.addr()?;
    let undefined = Element {
        addr: None,
        size: None,
        mask: 0,
    };
    let n_us = usize::try_from(n).map_err(|_| bad("extensible array too large"))?;
    let Some(iblock) = iblock else {
        return Ok(vec![undefined; n_us]);
    };
    if dblk_min == 0
        || !dblk_min.is_power_of_two()
        || sblk_min_ptrs == 0
        || !sblk_min_ptrs.is_power_of_two()
    {
        return Err(bad("bad extensible array parameters"));
    }
    let blk_off_size =
        usize::try_from(u64::from(max_bits).saturating_add(7) / 8).map_err(|_| bad("bad array"))?;
    // Super block geometry (H5EA__hdr_init).
    let nsblks = u64::from(max_bits.saturating_sub(log2_floor(dblk_min))).saturating_add(1);
    let mut sblk: Vec<(u64, u64, u64, u64)> = Vec::new(); // (ndblks, dblk_nelmts, start_idx, start_dblk)
    let (mut start_idx, mut start_dblk) = (Ck::from(0u64), Ck::from(0u64));
    for u in 0..nsblks {
        let ndblks = 1u64
            .checked_shl(u32::try_from(u / 2).unwrap_or(64))
            .unwrap_or(0);
        let nel = 1u64
            .checked_shl(u32::try_from(u.saturating_add(1) / 2).unwrap_or(64))
            .unwrap_or(0)
            .saturating_mul(dblk_min);
        let (si, sd) = (
            start_idx.to::<u64>().unwrap_or(u64::MAX),
            start_dblk.to::<u64>().unwrap_or(u64::MAX),
        );
        sblk.push((ndblks, nel, si, sd));
        start_idx = start_idx + Ck::from(ndblks) * nel;
        start_dblk = start_dblk + ndblks;
    }
    let iblock_nsblks = 2u64.saturating_mul(u64::from(log2_floor(sblk_min_ptrs)));
    let ndblk_addrs = 2u64.saturating_mul(sblk_min_ptrs.saturating_sub(1));
    let nsblk_addrs = nsblks.saturating_sub(iblock_nsblks);
    let page_nelmts = 1u64.checked_shl(page_bits).unwrap_or(u64::MAX);

    let mut c = file.cursor(iblock)?;
    c.expect(b"EAIB")?;
    c.skip(2)?;
    c.addr()?;
    let mut out = Vec::with_capacity(n_us.min(1 << 24));
    for _ in 0..iblock_elmts {
        out.push(element(&mut c, filtered, esize, so)?);
    }
    let dblk_addrs = (0..ndblk_addrs)
        .map(|_| c.addr())
        .collect::<Result<Vec<_>, _>>()?;
    let sblk_addrs = (0..nsblk_addrs)
        .map(|_| c.addr())
        .collect::<Result<Vec<_>, _>>()?;

    // Data blocks in order: those listed in the index block, then those of each super block.
    let read_dblock = |addr: Option<u64>, nel: u64, out: &mut Vec<Element>| -> Result<(), Error> {
        let Some(a) = addr else {
            out.extend(std::iter::repeat_n(
                undefined,
                usize::try_from(nel).unwrap_or(0),
            ));
            return Ok(());
        };
        let mut d = file.cursor(a)?;
        d.expect(b"EADB")?;
        d.skip(2)?;
        d.addr()?;
        d.uvar(blk_off_size)?;
        if nel > page_nelmts {
            // Paged: every page is present on disk with its own checksum.
            d.skip(4)?;
            let npages = nel.div_ceil(page_nelmts);
            for _ in 0..npages {
                for _ in 0..page_nelmts {
                    out.push(element(&mut d, filtered, esize, so)?);
                }
                d.skip(4)?;
            }
        } else {
            for _ in 0..nel {
                out.push(element(&mut d, filtered, esize, so)?);
            }
        }
        Ok(())
    };
    let mut dblk_iter = dblk_addrs.into_iter();
    for (u, (ndblks, nel, _, _)) in sblk.iter().enumerate() {
        if (out.len() as u64) >= n {
            break;
        }
        let u = u as u64;
        if u < iblock_nsblks {
            for _ in 0..*ndblks {
                read_dblock(dblk_iter.next().flatten(), *nel, &mut out)?;
            }
        } else {
            let sb = sblk_addrs
                .get(usize::try_from(u.saturating_sub(iblock_nsblks)).unwrap_or(usize::MAX))
                .copied()
                .flatten();
            match sb {
                None => out.extend(std::iter::repeat_n(
                    undefined,
                    usize::try_from(ndblks.saturating_mul(*nel)).unwrap_or(0),
                )),
                Some(sa) => {
                    let mut s = file.cursor(sa)?;
                    s.expect(b"EASB")?;
                    s.skip(2)?;
                    s.addr()?;
                    s.uvar(blk_off_size)?;
                    if *nel > page_nelmts {
                        let bitmap_len =
                            usize::try_from(nel.div_ceil(page_nelmts).div_ceil(8)).unwrap_or(0);
                        s.skip(bitmap_len.saturating_mul(usize::try_from(*ndblks).unwrap_or(0)))?;
                    }
                    let addrs = (0..*ndblks)
                        .map(|_| s.addr())
                        .collect::<Result<Vec<_>, _>>()?;
                    for a in addrs {
                        read_dblock(a, *nel, &mut out)?;
                    }
                }
            }
        }
    }
    out.truncate(n_us);
    if out.len() < n_us {
        out.extend(std::iter::repeat_n(
            undefined,
            n_us.saturating_sub(out.len()),
        ));
    }
    Ok(out)
}
