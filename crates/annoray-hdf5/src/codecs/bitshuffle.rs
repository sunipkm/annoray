//! The Bitshuffle HDF5 filter (32008) with no compression, LZ4 or Zstandard (the bit
//! transposition itself is `annoray_codecs::bitshuffle`).

use crate::Error;

use super::{be, corrupt};
use crate::ck::Ck;
use annoray_codecs::bitshuffle::untranspose;

const CODEC: &str = "bitshuffle";

fn err() -> Error {
    corrupt(CODEC)
}

/// A checked size.
fn z(c: Ck) -> Result<usize, Error> {
    c.usize().ok_or_else(err)
}

/// Untranspose `data` in blocks of `block` elements; within the last block, elements past the
/// last multiple of 8 are stored as they are.
fn untranspose_blocks(data: &[u8], size: usize, block: usize) -> Result<Vec<u8>, Error> {
    let n = data.len().checked_div(size).ok_or_else(err)?;
    if block == 0 {
        return Err(err());
    }
    let mut out = Vec::with_capacity(data.len());
    let mut start = 0usize;
    while start < n {
        let len = block.min(n.saturating_sub(start));
        let whole = len.saturating_sub(len % 8);
        let from = z(Ck::from(start) * size)?;
        let mid = z(Ck::from(whole) * size + from)?;
        let end = z(Ck::from(len) * size + from)?;
        out.extend(untranspose(
            data.get(from..mid).ok_or_else(err)?,
            whole,
            size,
        )?);
        // The tail of the final block is not transposed.
        out.extend_from_slice(data.get(mid..end).ok_or_else(err)?);
        start = z(Ck::from(start) + len)?;
    }
    out.extend_from_slice(data.get(z(Ck::from(n) * size)?..).unwrap_or_default());
    Ok(out)
}

/// One compressed block of `want` bytes (2 = LZ4, 3 = Zstandard).
fn block_codec(comp: usize, src: &[u8], want: usize) -> Result<Vec<u8>, Error> {
    match comp {
        2 => Ok(annoray_codecs::lz4::block_decompress(src, want)?),
        3 => {
            let out = annoray_codecs::zstd::decompress(src, want)?;
            if out.len() != want {
                return Err(err());
            }
            Ok(out)
        }
        other => Err(Error::Unsupported(format!(
            "bitshuffle compression {other}"
        ))),
    }
}

/// bitshuffle's default block: 8 KiB of elements, a multiple of 8, at least 128.
fn default_block(size: usize) -> usize {
    (8192usize.checked_div(size).unwrap_or(0) / 8)
        .saturating_mul(8)
        .max(128)
}

/// Decode a chunk written by the Bitshuffle filter. `params` are its client values:
/// version (2), element size, block size (0 = default), compression (0 none, 2 LZ4,
/// 3 Zstandard), [level].
pub(crate) fn decompress(data: &[u8], params: &[u32], limit: usize) -> Result<Vec<u8>, Error> {
    let p = |i: usize| {
        params
            .get(i)
            .copied()
            .map(|v| usize::try_from(v).unwrap_or(0))
    };
    let size = p(2).filter(|s| *s > 0).ok_or_else(err)?;
    let comp = p(4).unwrap_or(0);
    if comp == 0 {
        let block = p(3)
            .filter(|b| *b > 0)
            .unwrap_or_else(|| default_block(size));
        if data.len() > limit {
            return Err(err());
        }
        return untranspose_blocks(data, size, block);
    }
    // Compressed: total bytes (8, big-endian), block bytes (4), then each block as a 4-byte
    // compressed size and its data; leftover elements follow uncompressed.
    let total = usize::try_from(be(data, 0, 8, CODEC)?).map_err(|_| err())?;
    let block_bytes = usize::try_from(be(data, 8, 4, CODEC)?).map_err(|_| err())?;
    if total > limit || block_bytes == 0 || block_bytes.checked_rem(size) != Some(0) {
        return Err(err());
    }
    let block = block_bytes.checked_div(size).ok_or_else(err)?;
    let n = total.checked_div(size).ok_or_else(err)?;
    let mut out = Vec::with_capacity(total);
    let mut ip = 12usize;
    let mut done = 0usize;
    while done < n {
        let len = block.min(n.saturating_sub(done));
        let whole = len.saturating_sub(len % 8);
        if whole > 0 {
            let csize = usize::try_from(be(data, ip, 4, CODEC)?).map_err(|_| err())?;
            ip = z(Ck::from(ip) + 4usize)?;
            let end = ip.checked_add(csize).ok_or_else(err)?;
            let src = data.get(ip..end).ok_or_else(err)?;
            let raw = block_codec(comp, src, z(Ck::from(whole) * size)?)?;
            out.extend(untranspose(&raw, whole, size)?);
            ip = end;
        }
        if whole < len {
            // The final block's leftover elements, stored as they are.
            let end = z(Ck::from(len.saturating_sub(whole)) * size + ip)?;
            out.extend_from_slice(data.get(ip..end).ok_or_else(err)?);
            ip = end;
        }
        done = z(Ck::from(done) + len)?;
    }
    // Trailing bytes that do not make a whole element.
    out.extend_from_slice(
        data.get(ip..)
            .unwrap_or_default()
            .get(..total.saturating_sub(out.len()))
            .unwrap_or_default(),
    );
    Ok(out)
}
