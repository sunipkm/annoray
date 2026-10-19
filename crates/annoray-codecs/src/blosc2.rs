//! Blosc2: chunks (`README_CHUNK_FORMAT`), contiguous frames holding them
//! (`README_CFRAME_FORMAT`), and the `b2nd` metalayer that tiles a chunk into
//! n-dimensional blocks (`README_B2ND_METALAYER`), as the HDF5 Blosc2 filter (32026) stores
//! each HDF5 chunk: a frame of one Blosc2 chunk, block-tiled when the HDF5 chunk has more
//! than one dimension.
//!
//! Codecs: BloscLZ, LZ4 and LZ4HC, zlib, Zstandard (LZ4 and Zstandard need their features).
//! Filters: byte and bit shuffles, delta and precision truncation (which needs no undoing).
//! Zstandard dictionaries, user-defined codecs and filters, and lazy chunks are not
//! supported.

use crate::bits::{corrupt, le32};
use crate::ck::Ck;
use crate::shuffle::unshuffle;
use crate::{Error, bitshuffle};

const CODEC: &str = "blosc2";
const EXTENDED_HEADER: usize = 32;
const MEMCPYED: u8 = 0x02;
const DONT_SPLIT: u8 = 0x10;

fn err() -> Error {
    corrupt(CODEC)
}

fn z(c: Ck) -> Result<usize, Error> {
    c.usize().ok_or_else(err)
}

fn byte(b: &[u8], at: usize) -> Result<u8, Error> {
    b.get(at).copied().ok_or_else(err)
}

fn le_i32(b: &[u8], at: usize) -> Result<i32, Error> {
    let v = le32(b, at, CODEC)?;
    Ok(i32::from_le_bytes(v.to_le_bytes()))
}

/// The decompressed size a chunk declares.
///
/// # Errors
/// [`Error::Corrupt`] for a chunk too short to have a header.
pub fn chunk_nbytes(chunk: &[u8]) -> Result<usize, Error> {
    usize::try_from(le32(chunk, 4, CODEC)?).map_err(|_| err())
}

/// Decompress one Blosc2 chunk into at most `limit` bytes.
///
/// # Errors
/// [`Error::Corrupt`] for malformed chunks; [`Error::TooLarge`] past `limit`;
/// [`Error::Unsupported`] for features listed in the module documentation.
pub fn decompress_chunk(chunk: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    let version = byte(chunk, 0)?;
    let flags = byte(chunk, 2)?;
    let typesize = usize::from(byte(chunk, 3)?);
    let nbytes = chunk_nbytes(chunk)?;
    let blocksize = usize::try_from(le32(chunk, 8, CODEC)?).map_err(|_| err())?;
    if nbytes > limit {
        return Err(Error::TooLarge(CODEC.into()));
    }
    // A Blosc1-style chunk (no extended header): the version 1 decoder handles it.
    if flags & 0x05 != 0x05 {
        return crate::blosc::decompress(chunk, limit);
    }
    if chunk.len() < EXTENDED_HEADER {
        return Err(err());
    }
    let filters: [u8; 6] = chunk
        .get(16..22)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(err)?;
    let filters_meta: [u8; 6] = chunk
        .get(24..30)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(err)?;
    let flags2 = byte(chunk, 30)?;
    let blosc2_flags = byte(chunk, 31)?;
    if flags2 & 0x01 != 0 {
        return Err(Error::Unsupported(
            "Blosc2 chunks with variable-length blocks".into(),
        ));
    }
    if blosc2_flags & 0x01 != 0 {
        return Err(Error::Unsupported("Blosc2 dictionaries".into()));
    }
    if blosc2_flags & 0x08 != 0 {
        return Err(Error::Unsupported("lazy Blosc2 chunks".into()));
    }
    // Whole-chunk special values.
    match (blosc2_flags >> 4) & 0x07 {
        0 => {}
        // Zeros, and values never initialised (zeros here).
        1 | 4 => return Ok(vec![0u8; nbytes]),
        2 => {
            let nan: &[u8] = match typesize {
                4 => &[0x00, 0x00, 0xC0, 0x7F],
                8 => &[0, 0, 0, 0, 0, 0, 0xF8, 0x7F],
                _ => return Err(err()),
            };
            return Ok(nan.repeat(nbytes.checked_div(typesize).unwrap_or(0)));
        }
        3 => {
            let value = chunk
                .get(EXTENDED_HEADER..z(Ck::from(EXTENDED_HEADER) + typesize)?)
                .ok_or_else(err)?;
            return Ok(value.repeat(nbytes.checked_div(typesize).unwrap_or(0)));
        }
        k => return Err(Error::Unsupported(format!("Blosc2 special value {k}"))),
    }
    if flags & MEMCPYED != 0 {
        return Ok(chunk
            .get(EXTENDED_HEADER..z(Ck::from(nbytes) + EXTENDED_HEADER)?)
            .ok_or_else(err)?
            .to_vec());
    }
    if blocksize == 0 || typesize == 0 {
        return Err(err());
    }
    let codec = flags >> 5;
    if codec >= 5 {
        return Err(Error::Unsupported(format!(
            "Blosc2 codec {codec} (user-defined or stored elsewhere)"
        )));
    }
    let nblocks = nbytes.div_ceil(blocksize);
    let leftover = nbytes.checked_rem(blocksize).ok_or_else(err)?;
    let mut out: Vec<u8> = Vec::with_capacity(nbytes);
    for j in 0..nblocks {
        let last_partial = z(Ck::from(j) + 1usize)? == nblocks && leftover > 0;
        let bsize = if last_partial { leftover } else { blocksize };
        let mut ip = usize::try_from(le_i32(chunk, z(Ck::from(j) * 4usize + EXTENDED_HEADER)?)?)
            .map_err(|_| err())?;
        let nstreams = if flags & DONT_SPLIT == 0 && !last_partial {
            typesize
        } else {
            1
        };
        let neblock = bsize.checked_div(nstreams).ok_or_else(err)?;
        if neblock == 0 {
            return Err(err());
        }
        if bsize > nbytes {
            return Err(err());
        }
        let mut block = Vec::with_capacity(bsize);
        for _ in 0..nstreams {
            let csize = le_i32(chunk, ip)?;
            ip = z(Ck::from(ip) + 4usize)?;
            match csize {
                0 => block.resize(z(Ck::from(block.len()) + neblock)?, 0),
                c if c < 0 => {
                    // A run of one byte (the token says so).
                    let token = byte(chunk, ip)?;
                    ip = z(Ck::from(ip) + 1usize)?;
                    if token & 0x01 == 0 || c < -255 {
                        return Err(Error::Unsupported(format!("Blosc2 stream token {token}")));
                    }
                    let value = u8::try_from(c.unsigned_abs()).map_err(|_| err())?;
                    block.resize(z(Ck::from(block.len()) + neblock)?, value);
                }
                c => {
                    let c = usize::try_from(c).map_err(|_| err())?;
                    let data = chunk
                        .get(ip..ip.checked_add(c).ok_or_else(err)?)
                        .ok_or_else(err)?;
                    if c == neblock {
                        block.extend_from_slice(data);
                    } else {
                        block.extend(crate::blosc::stream(codec, data, neblock)?);
                    }
                    ip = z(Ck::from(ip) + c)?;
                }
            }
        }
        if block.len() != bsize {
            return Err(err());
        }
        // The filters, last first. Delta needs the whole reference block decoded (block 0).
        for i in (0..6).rev() {
            let f = filters.get(i).copied().unwrap_or(0);
            let meta = filters_meta.get(i).copied().unwrap_or(0);
            block = match f {
                0 | 4 => block,
                1 => unshuffle(
                    &block,
                    if meta == 0 {
                        typesize
                    } else {
                        usize::from(meta)
                    },
                ),
                2 => {
                    let n = bsize.checked_div(typesize).ok_or_else(err)?;
                    let whole = if version <= 2 && !n.is_multiple_of(8) {
                        0
                    } else {
                        n.saturating_sub(n % 8)
                    };
                    let cut = z(Ck::from(whole) * typesize)?;
                    let mut b = bitshuffle::untranspose(
                        block.get(..cut).ok_or_else(err)?,
                        whole,
                        typesize,
                    )?;
                    b.extend_from_slice(block.get(cut..).unwrap_or_default());
                    b
                }
                3 => {
                    undo_delta(&mut block, &out, j == 0, typesize);
                    block
                }
                other => {
                    return Err(Error::Unsupported(format!("Blosc2 filter {other}")));
                }
            };
        }
        out.extend(block);
    }
    Ok(out)
}

/// Undo the delta filter on one block: the reference block (block 0) XORs each item with the
/// one before it; the others XOR with the reference block's items (`reference`, decoded).
fn undo_delta(block: &mut [u8], reference: &[u8], first: bool, typesize: usize) {
    let width = match typesize {
        1 | 2 | 4 | 8 => typesize,
        t if t % 8 == 0 => 8,
        _ => 1,
    };
    if first {
        for i in width..block.len() {
            let prev = block.get(i.saturating_sub(width)).copied().unwrap_or(0);
            if let Some(b) = block.get_mut(i) {
                *b ^= prev;
            }
        }
    } else {
        for (b, r) in block.iter_mut().zip(reference) {
            *b ^= *r;
        }
    }
}

// ----- frames -----

/// The parts of a frame needed to decode it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct B2nd {
    /// The array's shape.
    pub shape: Vec<usize>,
    /// Its chunks' shape.
    pub chunkshape: Vec<usize>,
    /// Its blocks' shape (inside chunks).
    pub blockshape: Vec<usize>,
}

/// A tiny msgpack reader, for the frame's header and metalayers.
struct Msg<'a> {
    b: &'a [u8],
    at: usize,
}

#[derive(Debug)]
enum Value {
    Int(i64),
    Str(String),
    Bin(Vec<u8>),
    Array(Vec<Value>),
    Other,
}

impl Msg<'_> {
    fn u8(&mut self) -> Result<u8, Error> {
        let v = byte(self.b, self.at)?;
        self.at = z(Ck::from(self.at) + 1usize)?;
        Ok(v)
    }

    fn be(&mut self, n: usize) -> Result<u64, Error> {
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 8) | u64::from(self.u8()?);
        }
        Ok(v)
    }

    fn bytes(&mut self, n: usize) -> Result<&[u8], Error> {
        let end = z(Ck::from(self.at) + n)?;
        let s = self.b.get(self.at..end).ok_or_else(err)?;
        self.at = end;
        Ok(s)
    }

    fn signed(v: u64, bits: u32) -> i64 {
        let shift = 64u32.saturating_sub(bits);
        i64::from_ne_bytes((v << shift).to_ne_bytes()) >> shift
    }

    fn value(&mut self, depth: usize) -> Result<Value, Error> {
        if depth > 16 {
            return Err(err());
        }
        let len = |v: u64| usize::try_from(v).map_err(|_| err());
        let t = self.u8()?;
        Ok(match t {
            0x00..=0x7f => Value::Int(i64::from(t)),
            0xe0..=0xff => Value::Int(i64::from(i8::from_ne_bytes([t]))),
            0x90..=0x9f => self.array(usize::from(t & 0x0f), depth)?,
            0xa0..=0xbf => self.string(usize::from(t & 0x1f))?,
            0xcc => Value::Int(i64::try_from(self.be(1)?).map_err(|_| err())?),
            0xcd => Value::Int(i64::try_from(self.be(2)?).map_err(|_| err())?),
            0xce => Value::Int(i64::try_from(self.be(4)?).map_err(|_| err())?),
            0xcf => Value::Int(i64::try_from(self.be(8)?).map_err(|_| err())?),
            0xd0 => Value::Int(Self::signed(self.be(1)?, 8)),
            0xd1 => Value::Int(Self::signed(self.be(2)?, 16)),
            0xd2 => Value::Int(Self::signed(self.be(4)?, 32)),
            0xd3 => Value::Int(Self::signed(self.be(8)?, 64)),
            0xd9 => {
                let n = len(self.be(1)?)?;
                self.string(n)?
            }
            0xda => {
                let n = len(self.be(2)?)?;
                self.string(n)?
            }
            0xdb => {
                let n = len(self.be(4)?)?;
                self.string(n)?
            }
            0xc4..=0xc6 => {
                let n = len(self.be(1usize << t.saturating_sub(0xc4))?)?;
                Value::Bin(self.bytes(n)?.to_vec())
            }
            0xdc => {
                let n = len(self.be(2)?)?;
                self.array(n, depth)?
            }
            0xdd => {
                let n = len(self.be(4)?)?;
                self.array(n, depth)?
            }
            0xc0 | 0xc2 | 0xc3 => Value::Other,
            _ => {
                return Err(Error::Unsupported(format!(
                    "msgpack type {t:#x} in a Blosc2 frame"
                )));
            }
        })
    }

    fn array(&mut self, n: usize, depth: usize) -> Result<Value, Error> {
        if n > self.b.len() {
            return Err(err());
        }
        (0..n)
            .map(|_| self.value(depth.saturating_add(1)))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array)
    }

    fn string(&mut self, n: usize) -> Result<Value, Error> {
        Ok(Value::Str(
            String::from_utf8_lossy(self.bytes(n)?).into_owned(),
        ))
    }
}

fn dims(v: &Value) -> Result<Vec<usize>, Error> {
    match v {
        Value::Array(a) => a
            .iter()
            .map(|x| match x {
                Value::Int(i) => usize::try_from(*i).map_err(|_| err()),
                _ => Err(err()),
            })
            .collect(),
        _ => Err(err()),
    }
}

/// The `b2nd` metalayer of a frame, if it has one.
fn b2nd(frame: &[u8], header_len: usize) -> Result<Option<B2nd>, Error> {
    // The metalayers: [idx, {name: offset}, [contents]] after the fixed header (0x57).
    let header = frame.get(..header_len).ok_or_else(err)?;
    if header.len() <= 0x57 {
        return Ok(None);
    }
    // Each content is a bin32 at an absolute offset named in the map; find "b2nd" (or its
    // predecessor "caterva") by name.
    let mut m = Msg {
        b: header,
        at: 0x57,
    };
    if m.u8()? != 0x93 {
        return Ok(None);
    }
    let _idx = m.value(0)?;
    let map_len = match m.u8()? {
        0xde => usize::try_from(m.be(2)?).map_err(|_| err())?,
        t @ 0x80..=0x8f => usize::from(t & 0x0f),
        _ => return Err(err()),
    };
    let mut offset = None;
    for _ in 0..map_len {
        let Value::Str(name) = m.value(0)? else {
            return Err(err());
        };
        let Value::Int(at) = m.value(0)? else {
            return Err(err());
        };
        if name == "b2nd" || name == "caterva" {
            offset = Some(usize::try_from(at).map_err(|_| err())?);
        }
    }
    let Some(at) = offset else {
        return Ok(None);
    };
    let mut c = Msg { b: header, at };
    let Value::Bin(content) = c.value(0)? else {
        return Err(err());
    };
    let mut c = Msg { b: &content, at: 0 };
    let Value::Array(fields) = c.value(0)? else {
        return Err(err());
    };
    let (Some(shape), Some(chunkshape), Some(blockshape)) =
        (fields.get(2), fields.get(3), fields.get(4))
    else {
        return Err(err());
    };
    Ok(Some(B2nd {
        shape: dims(shape)?,
        chunkshape: dims(chunkshape)?,
        blockshape: dims(blockshape)?,
    }))
}

/// Decompress a frame of one chunk into at most `limit` bytes: the chunk's bytes, in C order
/// of the `b2nd` shape when there is one (the chunk's blocks untiled), and its metalayer.
///
/// # Errors
/// [`Error::Corrupt`] for malformed frames; [`Error::Unsupported`] for frames of other than
/// one chunk, and as for [`decompress_chunk`].
pub fn decompress_frame(frame: &[u8], limit: usize) -> Result<(Vec<u8>, Option<B2nd>), Error> {
    if frame.get(2..10) != Some(b"b2frame\0") {
        return Err(err());
    }
    let be = |at: usize, n: usize| -> Result<u64, Error> {
        let mut m = Msg { b: frame, at };
        m.be(n)
    };
    if byte(frame, 0x0a)? != 0xd2 || byte(frame, 0x0f)? != 0xcf {
        return Err(err());
    }
    let header_len = usize::try_from(be(0x0b, 4)?).map_err(|_| err())?;
    let general = byte(frame, 0x19)?;
    if (general >> 4) & 0x03 != 1 {
        return Err(Error::Unsupported(
            "Blosc2 frames without 64-bit offsets".into(),
        ));
    }
    if byte(frame, 0x1a)? & 0x0f != 0 {
        return Err(Error::Unsupported("sparse Blosc2 frames".into()));
    }
    if byte(frame, 0x1d)? != 0xd3 || byte(frame, 0x26)? != 0xd3 {
        return Err(err());
    }
    let nbytes = usize::try_from(be(0x1e, 8)?).map_err(|_| err())?;
    let cbytes = usize::try_from(be(0x27, 8)?).map_err(|_| err())?;
    if nbytes > limit {
        return Err(Error::TooLarge(CODEC.into()));
    }
    // The offsets chunk follows the data chunks.
    let offsets_at = z(Ck::from(header_len) + cbytes)?;
    let offsets_chunk = frame.get(offsets_at..).ok_or_else(err)?;
    let offsets = decompress_chunk(offsets_chunk, 8 * 1024)?;
    let first: [u8; 8] = offsets
        .get(..8)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(err)?;
    if offsets.len() != 8 {
        return Err(Error::Unsupported(
            "Blosc2 frames of more than one chunk".into(),
        ));
    }
    let data = if first[7] & 0x80 != 0 {
        // A whole chunk of special values (zeros, NaN or uninitialised).
        let typesize = usize::try_from(be(0x30, 4)?).map_err(|_| err())?;
        match first[7] & 0x07 {
            1 | 4 => vec![0u8; nbytes],
            2 => {
                let nan: &[u8] = match typesize {
                    4 => &[0x00, 0x00, 0xC0, 0x7F],
                    8 => &[0, 0, 0, 0, 0, 0, 0xF8, 0x7F],
                    _ => return Err(err()),
                };
                nan.repeat(nbytes.checked_div(typesize).unwrap_or(0))
            }
            k => return Err(Error::Unsupported(format!("Blosc2 special chunk {k}"))),
        }
    } else {
        let at = usize::try_from(i64::from_le_bytes(first)).map_err(|_| err())?;
        let chunk = frame.get(z(Ck::from(header_len) + at)?..).ok_or_else(err)?;
        decompress_chunk(chunk, limit)?
    };
    let meta = b2nd(frame, header_len)?;
    let Some(meta) = meta else {
        return Ok((data, None));
    };
    let typesize = usize::try_from(be(0x30, 4)?).map_err(|_| err())?;
    let untiled = untile(&data, &meta, typesize)?;
    Ok((untiled, Some(meta)))
}

/// The chunk's items (blocks of `blockshape`, block after block in C order of the block
/// grid, each in C order) in C order of `shape` (the chunk is the whole array).
fn untile(data: &[u8], meta: &B2nd, typesize: usize) -> Result<Vec<u8>, Error> {
    let nd = meta.shape.len();
    if meta.chunkshape.len() != nd || meta.blockshape.len() != nd {
        return Err(err());
    }
    if nd <= 1 && meta.blockshape == meta.chunkshape {
        return Ok(data.to_vec());
    }
    let shape = &meta.shape;
    let block = &meta.blockshape;
    if block.contains(&0) {
        return Ok(Vec::new());
    }
    // Blocks per dimension (over the chunk rounded up to whole blocks).
    let grid: Vec<usize> = meta
        .chunkshape
        .iter()
        .zip(block)
        .map(|(c, b)| c.div_ceil(*b))
        .collect();
    let block_items = block
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(err)?;
    let n = shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(err)?;
    // The blocks hold at least the shape's items (a corrupt shape cannot ask for more).
    let size = z(Ck::from(n) * typesize)?;
    if size > data.len() {
        return Err(err());
    }
    let mut out = vec![0u8; size];
    let nblocks = grid
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(err)?;
    // The data holds every block whole.
    if z(Ck::from(nblocks) * block_items * typesize)? > data.len() {
        return Err(err());
    }
    let last = nd.saturating_sub(1);
    let row = z(Ck::from(block.get(last).copied().unwrap_or(1)) * typesize)?;
    let rows_per_block = block_items
        .checked_div(block.get(last).copied().unwrap_or(1))
        .unwrap_or(0);
    for b in 0..nblocks {
        // The block's position in the grid.
        let mut rem = b;
        let mut bpos = vec![0usize; nd];
        for (slot, g) in bpos.iter_mut().zip(&grid).rev() {
            *slot = rem.checked_rem(*g).unwrap_or(0);
            rem = rem.checked_div(*g).unwrap_or(0);
        }
        let base = z(Ck::from(b) * block_items)?;
        for r in 0..rows_per_block {
            // The row's index inside the block (all but the last dimension).
            let mut rem = r;
            let mut inner = vec![0usize; nd];
            for k in (0..last).rev() {
                let bk = block.get(k).copied().unwrap_or(1);
                if let Some(slot) = inner.get_mut(k) {
                    *slot = rem.checked_rem(bk).unwrap_or(0);
                }
                rem = rem.checked_div(bk).unwrap_or(0);
            }
            // Global coordinates of the row's first item; rows past the shape are padding.
            let mut flat = Ck::from(0usize);
            let mut inside = true;
            for k in 0..nd {
                let g = bpos
                    .get(k)
                    .copied()
                    .unwrap_or(0)
                    .saturating_mul(block.get(k).copied().unwrap_or(1))
                    .saturating_add(inner.get(k).copied().unwrap_or(0));
                let d = shape.get(k).copied().unwrap_or(0);
                if g >= d {
                    inside = false;
                    break;
                }
                flat = flat * d + g;
            }
            if !inside {
                continue;
            }
            // Items of the row inside the shape.
            let start_last = bpos
                .get(last)
                .copied()
                .unwrap_or(0)
                .saturating_mul(block.get(last).copied().unwrap_or(1));
            let keep = shape
                .get(last)
                .copied()
                .unwrap_or(0)
                .saturating_sub(start_last)
                .min(block.get(last).copied().unwrap_or(1));
            let len = z(Ck::from(keep) * typesize)?;
            // The row starts `r` rows into the block's items.
            let src = z(Ck::from(base) * typesize + z(Ck::from(r) * row)?)?;
            let dst = z(flat * typesize)?;
            let (Some(d), Some(s)) = (
                out.get_mut(dst..dst.saturating_add(len)),
                data.get(src..src.saturating_add(len)),
            ) else {
                return Err(err());
            };
            d.copy_from_slice(s);
        }
    }
    Ok(out)
}
