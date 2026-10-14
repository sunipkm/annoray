//! Chunk filters.
//!
//! Decoding: deflate, shuffle, fletcher32, N-bit, scale-offset, LZF, bzip2, Blosc (BloscLZ,
//! LZ4, zlib, Zstandard) and Bitshuffle; with the `zstd` and `lz4` features also Zstandard and
//! LZ4 on their own and inside Blosc and Bitshuffle. Encoding: deflate, shuffle, fletcher32.

use xr_io::IoError;

use crate::bad;
use crate::format::Filter;

pub(crate) const DEFLATE: u16 = 1;
pub(crate) const SHUFFLE: u16 = 2;
pub(crate) const FLETCHER32: u16 = 3;
pub(crate) const NBIT: u16 = 5;
pub(crate) const SCALEOFFSET: u16 = 6;
pub(crate) const BZIP2: u16 = 307;
pub(crate) const BLOSC: u16 = 32001;
pub(crate) const BITSHUFFLE: u16 = 32008;
pub(crate) const LZF: u16 = 32000;
pub(crate) const LZ4: u16 = 32004;
pub(crate) const ZSTD: u16 = 32015;

/// Undo the pipeline on one chunk. `mask` bit `i` set means filter `i` was skipped.
/// `expected` bounds decompressed sizes so a corrupt chunk cannot exhaust memory.
pub(crate) fn decode(
    pipeline: &[Filter],
    mask: u32,
    mut data: Vec<u8>,
    expected: usize,
) -> Result<Vec<u8>, IoError> {
    for (i, f) in pipeline.iter().enumerate().rev() {
        if i < 32 && mask & (1u32 << i) != 0 {
            continue;
        }
        data = match f.id {
            DEFLATE => miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&data, expected)
                .map_err(|e| bad(format!("deflate: {e:?}")))?,
            SHUFFLE => {
                let size = f
                    .client
                    .first()
                    .copied()
                    .map_or(1, |s| usize::try_from(s).unwrap_or(1));
                unshuffle(&data, size)
            }
            FLETCHER32 => {
                let n = data
                    .len()
                    .checked_sub(4)
                    .ok_or_else(|| bad("fletcher32: chunk too short"))?;
                let stored: [u8; 4] = data
                    .get(n..)
                    .and_then(|s| s.try_into().ok())
                    .ok_or_else(|| bad("fletcher32"))?;
                let sum = fletcher32(data.get(..n).unwrap_or_default());
                // Some HDF5 versions wrote the checksum byte-swapped; accept both, as the library does.
                if sum != u32::from_le_bytes(stored) && sum != u32::from_be_bytes(stored) {
                    return Err(bad("fletcher32 checksum mismatch (corrupt chunk)"));
                }
                data.truncate(n);
                data
            }
            LZF => lzf_decompress(&data, expected)?,
            NBIT => crate::codecs::nbit::decompress(&data, &f.client, expected)?,
            SCALEOFFSET => crate::codecs::scaleoffset::decompress(&data, &f.client, expected)?,
            BZIP2 => crate::codecs::bzip2::decompress(&data, expected)?,
            BLOSC => crate::codecs::blosc::decompress(&data, expected)?,
            BITSHUFFLE => crate::codecs::bitshuffle::decompress(&data, &f.client, expected)?,
            #[cfg(feature = "zstd")]
            ZSTD => zstd_decompress(&data, expected)?,
            #[cfg(feature = "lz4")]
            LZ4 => lz4_decompress(&data, expected)?,
            #[cfg(not(feature = "zstd"))]
            ZSTD => {
                return Err(IoError::Unsupported(
                    "Zstandard-compressed data (enable the `zstd` feature)".into(),
                ));
            }
            #[cfg(not(feature = "lz4"))]
            LZ4 => {
                return Err(IoError::Unsupported(
                    "LZ4-compressed data (enable the `lz4` feature)".into(),
                ));
            }
            id => return Err(IoError::Unsupported(format!("HDF5 filter {id}"))),
        };
    }
    Ok(data)
}

/// LZF (liblzf, as h5py writes it), producing at most `limit` bytes.
pub(crate) fn lzf_decompress(input: &[u8], limit: usize) -> Result<Vec<u8>, IoError> {
    let corrupt = || bad("lzf: corrupt chunk");
    let mut out: Vec<u8> = Vec::with_capacity(limit);
    let mut ip = 0usize;
    let next = |ip: &mut usize| -> Result<u8, IoError> {
        let b = input.get(*ip).copied().ok_or_else(corrupt)?;
        *ip = ip.checked_add(1).ok_or_else(corrupt)?;
        Ok(b)
    };
    while ip < input.len() {
        let ctrl = usize::from(next(&mut ip)?);
        if ctrl < 32 {
            // A run of ctrl + 1 literal bytes.
            let end = ip.checked_add(ctrl + 1).ok_or_else(corrupt)?;
            let lit = input.get(ip..end).ok_or_else(corrupt)?;
            out.extend_from_slice(lit);
            ip = end;
        } else {
            // A back-reference of len + 2 bytes.
            let mut len = ctrl >> 5;
            if len == 7 {
                len = len
                    .checked_add(usize::from(next(&mut ip)?))
                    .ok_or_else(corrupt)?;
            }
            let back = ((ctrl & 0x1f) << 8)
                .checked_add(usize::from(next(&mut ip)?))
                .and_then(|b| b.checked_add(1))
                .ok_or_else(corrupt)?;
            let mut from = out.len().checked_sub(back).ok_or_else(corrupt)?;
            for _ in 0..len.checked_add(2).ok_or_else(corrupt)? {
                let b = out.get(from).copied().ok_or_else(corrupt)?;
                out.push(b);
                from = from.checked_add(1).ok_or_else(corrupt)?;
            }
        }
        if out.len() > limit {
            return Err(bad("lzf: chunk larger than expected"));
        }
    }
    Ok(out)
}

/// One Zstandard frame (the HDF5 zstd filter's chunk format), producing at most `limit`
/// bytes.
#[cfg(feature = "zstd")]
pub(crate) fn zstd_decompress(input: &[u8], limit: usize) -> Result<Vec<u8>, IoError> {
    use std::io::Read;
    let mut decoder =
        ruzstd::decoding::StreamingDecoder::new(input).map_err(|e| bad(format!("zstd: {e}")))?;
    let mut out = Vec::with_capacity(limit);
    let cap = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    (&mut decoder)
        .take(cap)
        .read_to_end(&mut out)
        .map_err(|e| bad(format!("zstd: {e}")))?;
    if out.len() > limit {
        return Err(bad("zstd: chunk larger than expected"));
    }
    Ok(out)
}

/// The HDF5 LZ4 filter's chunk format: the original size (8 bytes) and block size (4 bytes),
/// big-endian, then blocks, each a 4-byte compressed size and LZ4 block data (stored raw
/// when that size equals the block's).
#[cfg(feature = "lz4")]
pub(crate) fn lz4_decompress(input: &[u8], limit: usize) -> Result<Vec<u8>, IoError> {
    let corrupt = || bad("lz4: corrupt chunk");
    let be = |at: usize, n: usize| -> Result<u64, IoError> {
        let b = input
            .get(at..at.checked_add(n).ok_or_else(corrupt)?)
            .ok_or_else(corrupt)?;
        Ok(b.iter().fold(0u64, |acc, x| (acc << 8) | u64::from(*x)))
    };
    let total = usize::try_from(be(0, 8)?).map_err(|_| corrupt())?;
    let block = usize::try_from(be(8, 4)?).map_err(|_| corrupt())?;
    if total > limit || (block == 0 && total > 0) {
        return Err(corrupt());
    }
    let mut out = Vec::with_capacity(total);
    let mut ip = 12usize;
    while out.len() < total {
        let want = block.min(total.saturating_sub(out.len()));
        let size = usize::try_from(be(ip, 4)?).map_err(|_| corrupt())?;
        ip = ip.checked_add(4).ok_or_else(corrupt)?;
        let end = ip.checked_add(size).ok_or_else(corrupt)?;
        let data = input.get(ip..end).ok_or_else(corrupt)?;
        if size == want {
            out.extend_from_slice(data);
        } else {
            let start = out.len();
            out.resize(start.checked_add(want).ok_or_else(corrupt)?, 0);
            let dst = out.get_mut(start..).ok_or_else(corrupt)?;
            let n = lz4_flex::block::decompress_into(data, dst)
                .map_err(|e| bad(format!("lz4: {e}")))?;
            if n != want {
                return Err(corrupt());
            }
        }
        ip = end;
    }
    Ok(out)
}

/// One Zstandard frame (at Zstandard's fastest level, the one `ruzstd` implements), with its
/// content size in the header: the HDF5 zstd plugin sizes its output buffer from it, and
/// `ruzstd` leaves it out.
#[cfg(feature = "zstd")]
pub(crate) fn zstd_compress(data: &[u8]) -> Vec<u8> {
    let mut frame =
        ruzstd::encoding::compress_to_vec(data, ruzstd::encoding::CompressionLevel::Fastest);
    // Header: magic (4), descriptor (1), window descriptor (1, as single-segment is off),
    // then the content size if the descriptor's top two bits say so (3 = 8 bytes).
    let plain = frame.get(..4) == Some(&[0x28, 0xB5, 0x2F, 0xFD][..])
        && frame.get(4).is_some_and(|d| d & 0xE3 == 0); // no content size, window, no dictionary
    if plain {
        if let Some(d) = frame.get_mut(4) {
            *d |= 0xC0;
        }
        let size = (data.len() as u64).to_le_bytes();
        frame.splice(6..6, size);
    }
    frame
}

/// The HDF5 LZ4 filter's chunk format, as one block (stored raw if LZ4 does not shrink it).
#[cfg(feature = "lz4")]
pub(crate) fn lz4_compress(data: &[u8]) -> Result<Vec<u8>, IoError> {
    let n = u32::try_from(data.len()).map_err(|_| bad("lz4: chunk too large"))?;
    let c = lz4_flex::block::compress(data);
    let body: &[u8] = if c.len() < data.len() { &c } else { data };
    let mut out = Vec::with_capacity(body.len().saturating_add(16));
    out.extend_from_slice(&(data.len() as u64).to_be_bytes());
    out.extend_from_slice(&n.to_be_bytes());
    out.extend_from_slice(
        &u32::try_from(body.len())
            .map_err(|_| bad("lz4: chunk too large"))?
            .to_be_bytes(),
    );
    out.extend_from_slice(body);
    Ok(out)
}

/// Bytes grouped by position within each element → element-interleaved.
pub(crate) fn unshuffle(data: &[u8], size: usize) -> Vec<u8> {
    let Some(n) = data.len().checked_div(size).filter(|_| size > 1) else {
        return data.to_vec();
    };
    let mut out = data.to_vec(); // trailing partial element (if any) is left in place
    for (b, plane) in data.chunks(n.max(1)).take(size).enumerate() {
        for (e, byte) in plane.iter().enumerate().take(n) {
            if let Some(slot) = e
                .checked_mul(size)
                .and_then(|x| x.checked_add(b))
                .and_then(|i| out.get_mut(i))
            {
                *slot = *byte;
            }
        }
    }
    out
}

/// Element-interleaved → bytes grouped by position within each element.
pub(crate) fn shuffle(data: &[u8], size: usize) -> Vec<u8> {
    let Some(n) = data.len().checked_div(size).filter(|_| size > 1) else {
        return data.to_vec();
    };
    let mut out = data.to_vec();
    for (e, el) in data.chunks_exact(size).enumerate() {
        for (b, byte) in el.iter().enumerate() {
            if let Some(slot) = b
                .checked_mul(n)
                .and_then(|x| x.checked_add(e))
                .and_then(|i| out.get_mut(i))
            {
                *slot = *byte;
            }
        }
    }
    out
}

/// HDF5's Fletcher-32 checksum (H5_checksum_fletcher32), stored big-endian after the data.
pub(crate) fn fletcher32(data: &[u8]) -> u32 {
    let (mut sum1, mut sum2) = (0u32, 0u32);
    let mut words = data.chunks(2);
    let mut remaining = data.len() / 2;
    while remaining > 0 {
        let tlen = remaining.min(360);
        remaining = remaining.saturating_sub(tlen);
        for _ in 0..tlen {
            let w = words.next().unwrap_or_default();
            let v = (u32::from(w.first().copied().unwrap_or(0)) << 8)
                | u32::from(w.get(1).copied().unwrap_or(0));
            sum1 = sum1.wrapping_add(v);
            sum2 = sum2.wrapping_add(sum1);
        }
        sum1 = (sum1 & 0xffff).wrapping_add(sum1 >> 16);
        sum2 = (sum2 & 0xffff).wrapping_add(sum2 >> 16);
    }
    if data.len() % 2 == 1 {
        let last = data.last().copied().unwrap_or(0);
        sum1 = sum1.wrapping_add(u32::from(last) << 8);
        sum2 = sum2.wrapping_add(sum1);
        sum1 = (sum1 & 0xffff).wrapping_add(sum1 >> 16);
        sum2 = (sum2 & 0xffff).wrapping_add(sum2 >> 16);
    }
    sum1 = (sum1 & 0xffff).wrapping_add(sum1 >> 16);
    sum2 = (sum2 & 0xffff).wrapping_add(sum2 >> 16);
    (sum2 << 16) | sum1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_frames_carry_their_content_size() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 13) as u8).collect();
        let frame = zstd_compress(&data);
        assert_eq!(frame[4] & 0xC0, 0xC0);
        assert_eq!(frame[6..14], (data.len() as u64).to_le_bytes());
        assert_eq!(zstd_decompress(&frame, data.len()).unwrap(), data);
    }

    #[test]
    fn lzf_decodes_literals_and_back_references() {
        // "abcabcabcd": 3 literals, a back-reference of 6 bytes at distance 3, 1 literal.
        let enc = [2, b'a', b'b', b'c', 4 << 5, 2, 0, b'd'];
        assert_eq!(lzf_decompress(&enc, 100).unwrap(), b"abcabcabcd");
        assert!(lzf_decompress(&enc, 5).is_err(), "the size limit holds");
        assert!(
            lzf_decompress(&[1 << 5, 9], 100).is_err(),
            "reference before the start"
        );
        assert!(
            lzf_decompress(&[5, b'a'], 100).is_err(),
            "truncated literal run"
        );
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_round_trips_through_the_hdf5_framing() {
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 7) as u8).collect();
        let mut framed = Vec::new();
        framed.extend_from_slice(&(data.len() as u64).to_be_bytes());
        framed.extend_from_slice(&4096u32.to_be_bytes());
        for block in data.chunks(4096) {
            let c = lz4_flex::block::compress(block);
            framed.extend_from_slice(&(c.len() as u32).to_be_bytes());
            framed.extend_from_slice(&c);
        }
        assert_eq!(lz4_decompress(&framed, data.len()).unwrap(), data);
        assert!(lz4_decompress(&framed, 10).is_err());
    }

    #[test]
    fn shuffle_round_trips() {
        let data: Vec<u8> = (0..24).collect();
        for size in [1, 2, 4, 8, 3] {
            assert_eq!(unshuffle(&shuffle(&data, size), size), data, "size {size}");
        }
        assert_eq!(shuffle(&[1, 2, 3, 4], 2), vec![1, 3, 2, 4]);
    }
}
