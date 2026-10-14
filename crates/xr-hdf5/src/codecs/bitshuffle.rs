//! Bit-transposition (the bitshuffle library's layout), and the Bitshuffle HDF5 filter (32008)
//! with no compression, LZ4 or Zstandard.

use xr_io::IoError;

use super::{be, corrupt};
use crate::ck::Ck;

const CODEC: &str = "bitshuffle";

fn err() -> IoError {
    corrupt(CODEC)
}

/// A checked size.
fn z(c: Ck) -> Result<usize, IoError> {
    c.usize().ok_or_else(err)
}

/// Transpose an 8×8 bit matrix held in a `u64` (byte `r` is row `r`, bit `c` column `c`).
fn transpose8(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^ t ^ (t << 28)
}

/// The bitshuffle layout, eight bytes at a time: for byte position `b` of the elements and
/// element group `jj` (8 elements), the 8 bit-plane bytes `(b * 8 + k) * n / 8 + jj` and the 8
/// element bytes `(jj * 8 + e) * size + b` are each other's bit transpose.
fn regroup(data: &[u8], n: usize, size: usize, planes_in: bool) -> Result<Vec<u8>, IoError> {
    let total = n.checked_mul(size).ok_or_else(err)?;
    if !n.is_multiple_of(8) || data.len() < total {
        return Err(err());
    }
    let rows = n / 8;
    let mut out = vec![0u8; total];
    let plane = |b: usize, k: usize, jj: usize| z((Ck::from(b) * 8usize + k) * rows + jj);
    let elem = |b: usize, e: usize, jj: usize| z((Ck::from(jj) * 8usize + e) * size + b);
    for b in 0..size {
        for jj in 0..rows {
            let mut x = 0u64;
            for r in 0..8u32 {
                let k = usize::try_from(r).map_err(|_| err())?;
                let at = if planes_in {
                    plane(b, k, jj)?
                } else {
                    elem(b, k, jj)?
                };
                x |= u64::from(*data.get(at).ok_or_else(err)?).wrapping_shl(r.wrapping_mul(8));
            }
            let y = transpose8(x);
            for r in 0..8u32 {
                let k = usize::try_from(r).map_err(|_| err())?;
                let at = if planes_in {
                    elem(b, k, jj)?
                } else {
                    plane(b, k, jj)?
                };
                *out.get_mut(at).ok_or_else(err)? =
                    y.wrapping_shr(r.wrapping_mul(8)).to_le_bytes()[0];
            }
        }
    }
    Ok(out)
}

/// Undo bitshuffle's transpose of `n` elements of `size` bytes (`n` a multiple of 8): bit
/// `k` of byte `b` of element `j` is stored at bit `j % 8` of byte
/// `(b * 8 + k) * n / 8 + j / 8`.
pub(crate) fn untranspose(data: &[u8], n: usize, size: usize) -> Result<Vec<u8>, IoError> {
    regroup(data, n, size, true)
}

/// bitshuffle's transpose (the inverse of [`untranspose`]).
pub(crate) fn transpose(data: &[u8], n: usize, size: usize) -> Result<Vec<u8>, IoError> {
    regroup(data, n, size, false)
}

/// Untranspose `data` in blocks of `block` elements; within the last block, elements past the
/// last multiple of 8 are stored as they are.
fn untranspose_blocks(data: &[u8], size: usize, block: usize) -> Result<Vec<u8>, IoError> {
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
fn block_codec(comp: usize, src: &[u8], want: usize) -> Result<Vec<u8>, IoError> {
    match comp {
        #[cfg(feature = "lz4")]
        2 => {
            let mut buf = vec![0u8; want];
            let got = lz4_flex::block::decompress_into(src, &mut buf)
                .map_err(|e| crate::bad(format!("bitshuffle lz4: {e}")))?;
            if got != want {
                return Err(err());
            }
            Ok(buf)
        }
        #[cfg(feature = "zstd")]
        3 => crate::filters::zstd_decompress(src, want),
        other => {
            let _ = (src, want);
            Err(IoError::Unsupported(format!(
                "bitshuffle compression {other} (needs the `lz4`/`zstd` features)"
            )))
        }
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
pub(crate) fn decompress(data: &[u8], params: &[u32], limit: usize) -> Result<Vec<u8>, IoError> {
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

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, reason = "small test values")]
mod tests {
    use super::*;

    /// The forward transpose, straight from the definition.
    fn reference(data: &[u8], n: usize, size: usize) -> Vec<u8> {
        let mut out = vec![0u8; n * size];
        for j in 0..n {
            for b in 0..size {
                for k in 0..8 {
                    if data[j * size + b] >> k & 1 == 1 {
                        out[(b * 8 + k) * (n / 8) + j / 8] |= 1 << (j % 8);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn untranspose_inverts_transpose() {
        let data: Vec<u8> = (0..96u32).map(|i| (i * 37 % 251) as u8).collect();
        for size in [1, 2, 4, 12] {
            let n = data.len() / size / 8 * 8;
            let part = &data[..n * size];
            assert_eq!(transpose(part, n, size).unwrap(), reference(part, n, size));
            assert_eq!(
                untranspose(&reference(part, n, size), n, size).unwrap(),
                part
            );
        }
        assert!(untranspose(&[1, 2, 3], 3, 1).is_err());
    }
}
