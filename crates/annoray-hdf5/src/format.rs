//! HDF5 on-disk structures: superblock, object headers and the messages they contain.
//! (HDF5 File Format Specification, version 3.)

use crate::Error;

use crate::ck::Ck;
use crate::cursor::{Cursor, Sizes, align8};
use crate::dtype::Datatype;
use crate::{File, bad};

pub(crate) const SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";

/// What the superblock tells us.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Superblock {
    pub(crate) version: u8,
    pub(crate) sizes: Sizes,
    pub(crate) base: u64,
    pub(crate) root: u64,
}

/// Find and parse the superblock (at 0, 512, 1024, 2048, …).
pub(crate) fn superblock(buf: &[u8]) -> Result<Superblock, Error> {
    let mut at = 0usize;
    let start = loop {
        match buf.get(at..at.checked_add(8).ok_or_else(|| bad("not an HDF5 file"))?) {
            Some(s) if s == SIGNATURE => break at,
            Some(_) => {
                at = if at == 0 {
                    512
                } else {
                    at.checked_mul(2).ok_or_else(|| bad("not an HDF5 file"))?
                }
            }
            None => return Err(bad("not an HDF5 file")),
        }
    };
    let probe = Sizes {
        offset: 8,
        length: 8,
    };
    let mut c = Cursor::new(
        buf,
        start.checked_add(8).ok_or_else(|| bad("bad superblock"))?,
        probe,
    );
    let version = c.u8()?;
    let sb = match version {
        0 | 1 => {
            c.skip(4)?; // free-space, root group symbol table, reserved, shared header versions
            let offset = c.u8()?;
            let length = c.u8()?;
            c.skip(1)?;
            c.skip(4)?; // group leaf/internal node K
            c.skip(4)?; // file consistency flags
            if version == 1 {
                c.skip(4)?; // indexed storage K + reserved
            }
            c.sizes = Sizes { offset, length };
            let base = c.addr()?.unwrap_or(0);
            c.addr()?; // free space info
            c.addr()?; // end of file
            c.addr()?; // driver info
            // Root group symbol table entry: link name offset, object header address, ….
            c.length()?;
            let root = c.addr()?.ok_or_else(|| bad("no root group"))?;
            Superblock {
                version,
                sizes: c.sizes,
                base,
                root,
            }
        }
        2 | 3 => {
            let offset = c.u8()?;
            let length = c.u8()?;
            c.skip(1)?; // consistency flags
            c.sizes = Sizes { offset, length };
            let base = c.addr()?.unwrap_or(0);
            c.addr()?; // superblock extension
            c.addr()?; // end of file
            let root = c.addr()?.ok_or_else(|| bad("no root group"))?;
            Superblock {
                version,
                sizes: c.sizes,
                base,
                root,
            }
        }
        v => return Err(bad(format!("unsupported superblock version {v}"))),
    };
    if !matches!(sb.sizes.offset, 2 | 4 | 8) || !matches!(sb.sizes.length, 2 | 4 | 8) {
        return Err(bad("unsupported offset/length size"));
    }
    Ok(sb)
}

/// A raw header message.
#[derive(Debug, Clone)]
pub(crate) struct Message {
    pub(crate) kind: u16,
    pub(crate) flags: u8,
    /// Creation order, when the object header tracks it (attribute order).
    pub(crate) order: Option<u16>,
    pub(crate) data: Vec<u8>,
}

pub(crate) mod kind {
    pub(crate) const DATASPACE: u16 = 0x0001;
    pub(crate) const LINK_INFO: u16 = 0x0002;
    pub(crate) const DATATYPE: u16 = 0x0003;
    pub(crate) const FILL_OLD: u16 = 0x0004;
    pub(crate) const FILL: u16 = 0x0005;
    pub(crate) const LINK: u16 = 0x0006;
    pub(crate) const LAYOUT: u16 = 0x0008;
    pub(crate) const FILTERS: u16 = 0x000B;
    pub(crate) const ATTRIBUTE: u16 = 0x000C;
    pub(crate) const CONTINUATION: u16 = 0x0010;
    pub(crate) const SYMBOL_TABLE: u16 = 0x0011;
    pub(crate) const ATTRIBUTE_INFO: u16 = 0x0015;
}

/// All messages of the object header at `addr`, following continuation blocks.
pub(crate) fn object_header(file: &File, addr: u64) -> Result<Vec<Message>, Error> {
    let mut c = file.cursor(addr)?;
    let mut out = Vec::new();
    let mut pending: Vec<(u64, usize)> = Vec::new();
    let first = c.rest().get(..4).unwrap_or_default();
    let v2 = first == b"OHDR";
    let mut flags = 0u8;
    if v2 {
        c.skip(4)?;
        let version = c.u8()?;
        if version != 2 {
            return Err(bad(format!("object header version {version}")));
        }
        flags = c.u8()?;
        if flags & 0x20 != 0 {
            c.skip(16)?; // times
        }
        if flags & 0x10 != 0 {
            c.skip(4)?; // attribute phase change
        }
        let size = c.uvar(1usize << (flags & 3))?;
        let size = usize::try_from(size).map_err(|_| bad("object header too large"))?;
        let block = c.bytes(size)?;
        read_v2_messages(block, file.sizes, flags, &mut out, &mut pending)?;
    } else {
        let version = c.u8()?;
        if version != 1 {
            return Err(bad(format!("object header version {version}")));
        }
        c.skip(1)?;
        let _nmsgs = c.u16()?;
        c.skip(4)?; // reference count
        let size = usize::try_from(c.u32()?).map_err(|_| bad("object header too large"))?;
        c.skip(4)?; // pad to 8 (header prefix is 16 bytes)
        let block = c.bytes(size)?;
        read_v1_messages(block, file.sizes, &mut out, &mut pending)?;
    }
    // Continuation blocks (bounded, in case of cycles in corrupt files).
    let mut guard = 0u32;
    while let Some((addr, len)) = pending.pop() {
        guard = guard.saturating_add(1);
        if guard > 10_000 {
            return Err(bad("too many object header continuation blocks"));
        }
        let mut c = file.cursor(addr)?;
        if v2 {
            c.expect(b"OCHK")?;
            // The length covers the signature and the trailing checksum.
            let body = c.bytes(
                len.checked_sub(8)
                    .ok_or_else(|| bad("bad continuation block"))?,
            )?;
            read_v2_messages(body, file.sizes, flags, &mut out, &mut pending)?;
        } else {
            read_v1_messages(c.bytes(len)?, file.sizes, &mut out, &mut pending)?;
        }
    }
    Ok(out)
}

fn push(
    kind: u16,
    flags: u8,
    order: Option<u16>,
    data: &[u8],
    out: &mut Vec<Message>,
    pending: &mut Vec<(u64, usize)>,
    sizes: Sizes,
) -> Result<(), Error> {
    if kind == kind::CONTINUATION {
        let mut c = Cursor::new(data, 0, sizes);
        let addr = c
            .addr()?
            .ok_or_else(|| bad("undefined continuation address"))?;
        let len = c.len_usize()?;
        pending.push((addr, len));
    } else {
        out.push(Message {
            kind,
            flags,
            order,
            data: data.to_vec(),
        });
    }
    Ok(())
}

fn read_v1_messages(
    block: &[u8],
    s: Sizes,
    out: &mut Vec<Message>,
    pending: &mut Vec<(u64, usize)>,
) -> Result<(), Error> {
    let mut c = Cursor::new(block, 0, s);
    while c.rest().len() >= 8 {
        let kind = c.u16()?;
        let size = usize::from(c.u16()?);
        let flags = c.u8()?;
        c.skip(3)?;
        let data = c.bytes(size)?;
        push(kind, flags, None, data, out, pending, s)?;
    }
    Ok(())
}

fn read_v2_messages(
    block: &[u8],
    s: Sizes,
    hflags: u8,
    out: &mut Vec<Message>,
    pending: &mut Vec<(u64, usize)>,
) -> Result<(), Error> {
    let mut c = Cursor::new(block, 0, s);
    let prefix = if hflags & 0x04 != 0 { 6 } else { 4 };
    while c.rest().len() >= prefix {
        let kind = u16::from(c.u8()?);
        let size = usize::from(c.u16()?);
        let flags = c.u8()?;
        let order = if hflags & 0x04 != 0 {
            Some(c.u16()?)
        } else {
            None
        };
        let data = c.bytes(size)?;
        push(kind, flags, order, data, out, pending, s)?;
    }
    Ok(())
}

// ----- messages -----

/// A dataspace: current and maximum dimensions (`u64::MAX` = unlimited).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Space {
    pub(crate) dims: Vec<u64>,
    pub(crate) max: Vec<u64>,
}

/// Dataspace: `None` for the null dataspace; empty dims for scalars.
pub(crate) fn dataspace(data: &[u8], s: Sizes) -> Result<Option<Space>, Error> {
    let mut c = Cursor::new(data, 0, s);
    let version = c.u8()?;
    let rank = usize::from(c.u8()?);
    let flags = c.u8()?;
    let null = match version {
        1 => {
            c.skip(5)?;
            false
        }
        2 => c.u8()? == 2,
        v => return Err(bad(format!("dataspace version {v}"))),
    };
    if null {
        return Ok(None);
    }
    let dims = (0..rank)
        .map(|_| c.length())
        .collect::<Result<Vec<u64>, _>>()?;
    let max = if flags & 1 != 0 {
        let undef = crate::cursor::undefined(s.length);
        (0..rank)
            .map(|_| c.length().map(|m| if m == undef { u64::MAX } else { m }))
            .collect::<Result<Vec<u64>, _>>()?
    } else {
        dims.clone()
    };
    Ok(Some(Space { dims, max }))
}

/// Fill value bytes, if defined.
pub(crate) fn fill_value(kind_: u16, data: &[u8], s: Sizes) -> Result<Option<Vec<u8>>, Error> {
    let mut c = Cursor::new(data, 0, s);
    if kind_ == kind::FILL_OLD {
        let n = usize::try_from(c.u32()?).map_err(|_| bad("fill too large"))?;
        return (n > 0).then(|| c.bytes(n).map(<[u8]>::to_vec)).transpose();
    }
    let version = c.u8()?;
    let defined = match version {
        1 | 2 => {
            c.skip(2)?; // allocation time, write time
            let defined = c.u8()? != 0;
            if version == 2 && !defined {
                return Ok(None);
            }
            defined
        }
        3 => {
            let flags = c.u8()?;
            if flags & 0x20 == 0 {
                return Ok(None);
            }
            true
        }
        v => return Err(bad(format!("fill value version {v}"))),
    };
    let n = usize::try_from(c.u32()?).map_err(|_| bad("fill too large"))?;
    (defined && n > 0)
        .then(|| c.bytes(n).map(<[u8]>::to_vec))
        .transpose()
}

/// How a dataset's chunks are indexed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChunkIndex {
    BTreeV1,
    Single {
        filtered_size: Option<u64>,
        mask: u32,
    },
    Implicit,
    FixedArray,
    ExtensibleArray,
    BTreeV2,
}

/// Data layout.
#[derive(Debug, Clone)]
pub(crate) enum Layout {
    Compact(Vec<u8>),
    Contiguous {
        addr: Option<u64>,
        size: u64,
    },
    Chunked {
        addr: Option<u64>,
        chunk: Vec<u64>,
        index: ChunkIndex,
    },
}

pub(crate) fn layout(data: &[u8], s: Sizes) -> Result<Layout, Error> {
    let mut c = Cursor::new(data, 0, s);
    let version = c.u8()?;
    match version {
        1 | 2 => {
            let rank = usize::from(c.u8()?);
            let class = c.u8()?;
            c.skip(5)?;
            let addr = if class == 0 { None } else { c.addr()? };
            let dims = (0..rank)
                .map(|_| c.u32().map(u64::from))
                .collect::<Result<Vec<_>, _>>()?;
            match class {
                0 => {
                    let n = usize::try_from(c.u32()?).map_err(|_| bad("compact too large"))?;
                    Ok(Layout::Compact(c.bytes(n)?.to_vec()))
                }
                1 => {
                    let size = dims
                        .iter()
                        .try_fold(Ck::from(1u64), |a, d| Some(a * *d))
                        .and_then(|x| x.to())
                        .unwrap_or(0);
                    Ok(Layout::Contiguous { addr, size })
                }
                _ => Ok(Layout::Chunked {
                    addr,
                    chunk: dims,
                    index: ChunkIndex::BTreeV1,
                }),
            }
        }
        3 => match c.u8()? {
            0 => {
                let n = usize::from(c.u16()?);
                Ok(Layout::Compact(c.bytes(n)?.to_vec()))
            }
            1 => Ok(Layout::Contiguous {
                addr: c.addr()?,
                size: c.length()?,
            }),
            2 => {
                let rank = usize::from(c.u8()?);
                let addr = c.addr()?;
                let chunk = (0..rank)
                    .map(|_| c.u32().map(u64::from))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Layout::Chunked {
                    addr,
                    chunk,
                    index: ChunkIndex::BTreeV1,
                })
            }
            k => Err(bad(format!("layout class {k}"))),
        },
        // Version 5 (HDF5 2.0) has the same message format as 4; it only widens the sizes of
        // filtered chunks inside the chunk indexes, which are decoded from their element sizes.
        4 | 5 => match c.u8()? {
            0 => {
                let n = usize::from(c.u16()?);
                Ok(Layout::Compact(c.bytes(n)?.to_vec()))
            }
            1 => Ok(Layout::Contiguous {
                addr: c.addr()?,
                size: c.length()?,
            }),
            2 => {
                let flags = c.u8()?;
                let rank = usize::from(c.u8()?);
                let enc = usize::from(c.u8()?);
                let chunk = (0..rank)
                    .map(|_| c.uvar(enc))
                    .collect::<Result<Vec<_>, _>>()?;
                let index = match c.u8()? {
                    1 => {
                        let (filtered_size, mask) = if flags & 2 != 0 {
                            (Some(c.length()?), c.u32()?)
                        } else {
                            (None, 0)
                        };
                        ChunkIndex::Single {
                            filtered_size,
                            mask,
                        }
                    }
                    2 => ChunkIndex::Implicit,
                    3 => {
                        c.skip(1)?; // page bits
                        ChunkIndex::FixedArray
                    }
                    4 => {
                        c.skip(5)?; // max bits, index elements, min pointers, min elements, page bits
                        ChunkIndex::ExtensibleArray
                    }
                    5 => {
                        c.skip(6)?; // node size, split %, merge %
                        ChunkIndex::BTreeV2
                    }
                    t => return Err(bad(format!("chunk index type {t}"))),
                };
                Ok(Layout::Chunked {
                    addr: c.addr()?,
                    chunk,
                    index,
                })
            }
            3 => Err(bad("virtual datasets are not supported")),
            k => Err(bad(format!("layout class {k}"))),
        },
        v => Err(bad(format!("layout version {v}"))),
    }
}

/// One filter of a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Filter {
    pub(crate) id: u16,
    pub(crate) client: Vec<u32>,
}

pub(crate) fn filters(data: &[u8], s: Sizes) -> Result<Vec<Filter>, Error> {
    let mut c = Cursor::new(data, 0, s);
    let version = c.u8()?;
    let n = c.u8()?;
    if version == 1 {
        c.skip(6)?;
    }
    let mut out = Vec::new();
    for _ in 0..n {
        let id = c.u16()?;
        let name_len = if version == 1 || id >= 256 {
            usize::from(c.u16()?)
        } else {
            0
        };
        let _flags = c.u16()?;
        let ncd = usize::from(c.u16()?);
        if version == 1 {
            c.skip(align8(name_len).ok_or_else(|| bad("bad filter name"))?)?;
        } else {
            c.skip(name_len)?;
        }
        let client = (0..ncd).map(|_| c.u32()).collect::<Result<Vec<_>, _>>()?;
        if version == 1 && ncd % 2 == 1 {
            c.skip(4)?;
        }
        out.push(Filter { id, client });
    }
    Ok(out)
}

/// An attribute message.
#[derive(Debug, Clone)]
pub(crate) struct AttributeMsg {
    pub(crate) name: String,
    pub(crate) dtype: Datatype,
    pub(crate) dims: Option<Vec<u64>>,
    pub(crate) data: Vec<u8>,
}

pub(crate) fn attribute(file: &File, data: &[u8]) -> Result<AttributeMsg, Error> {
    let mut c = Cursor::new(data, 0, file.sizes);
    let version = c.u8()?;
    let flags = if version == 1 {
        c.skip(1)?;
        0
    } else {
        c.u8()?
    };
    let name_len = usize::from(c.u16()?);
    let dt_len = usize::from(c.u16()?);
    let ds_len = usize::from(c.u16()?);
    if version == 3 {
        c.skip(1)?; // name character set
    }
    let pad = |n: usize| if version == 1 { align8(n) } else { Some(n) };
    let name_raw = c.bytes(pad(name_len).ok_or_else(|| bad("bad attribute"))?)?;
    let name_bytes = name_raw.get(..name_len).unwrap_or_default();
    let name = String::from_utf8_lossy(name_bytes.split(|b| *b == 0).next().unwrap_or_default())
        .into_owned();
    let dt_bytes = c.bytes(pad(dt_len).ok_or_else(|| bad("bad attribute"))?)?;
    let dtype = if flags & 1 != 0 {
        crate::dtype::shared_datatype(file, dt_bytes)?
    } else {
        Datatype::parse(dt_bytes)?
    };
    let ds_bytes = c.bytes(pad(ds_len).ok_or_else(|| bad("bad attribute"))?)?;
    let dims = dataspace(ds_bytes, file.sizes)?.map(|sp| sp.dims);
    Ok(AttributeMsg {
        name,
        dtype,
        dims,
        data: c.rest().to_vec(),
    })
}

/// A link message.
#[derive(Debug, Clone)]
pub(crate) struct Link {
    pub(crate) name: String,
    /// `None` for soft and external links (not followed).
    pub(crate) target: Option<u64>,
    pub(crate) order: Option<u64>,
}

pub(crate) fn link(data: &[u8], s: Sizes) -> Result<Link, Error> {
    let mut c = Cursor::new(data, 0, s);
    let version = c.u8()?;
    if version != 1 {
        return Err(bad(format!("link version {version}")));
    }
    let flags = c.u8()?;
    let ltype = if flags & 0x08 != 0 { c.u8()? } else { 0 };
    let order = if flags & 0x04 != 0 {
        Some(c.u64()?)
    } else {
        None
    };
    if flags & 0x10 != 0 {
        c.skip(1)?; // charset
    }
    let n =
        usize::try_from(c.uvar(1usize << (flags & 3))?).map_err(|_| bad("link name too long"))?;
    let name = String::from_utf8_lossy(c.bytes(n)?).into_owned();
    let target = if ltype == 0 { c.addr()? } else { None };
    Ok(Link {
        name,
        target,
        order,
    })
}

/// Link info or attribute info: where dense storage lives.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseInfo {
    pub(crate) heap: Option<u64>,
    pub(crate) name_index: Option<u64>,
}

pub(crate) fn link_info(data: &[u8], s: Sizes) -> Result<DenseInfo, Error> {
    let mut c = Cursor::new(data, 0, s);
    c.skip(1)?;
    let flags = c.u8()?;
    if flags & 1 != 0 {
        c.skip(8)?;
    }
    Ok(DenseInfo {
        heap: c.addr()?,
        name_index: c.addr()?,
    })
}

pub(crate) fn attribute_info(data: &[u8], s: Sizes) -> Result<DenseInfo, Error> {
    let mut c = Cursor::new(data, 0, s);
    c.skip(1)?;
    let flags = c.u8()?;
    if flags & 1 != 0 {
        c.skip(2)?;
    }
    Ok(DenseInfo {
        heap: c.addr()?,
        name_index: c.addr()?,
    })
}

/// Symbol table message: (B-tree address, local heap address).
pub(crate) fn symbol_table(data: &[u8], s: Sizes) -> Result<(u64, u64), Error> {
    let mut c = Cursor::new(data, 0, s);
    let bt = c
        .addr()?
        .ok_or_else(|| bad("undefined symbol table B-tree"))?;
    let heap = c
        .addr()?
        .ok_or_else(|| bad("undefined symbol table heap"))?;
    Ok((bt, heap))
}
