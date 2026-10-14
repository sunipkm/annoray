//! Chunk filters: deflate, shuffle, fletcher32 (decoding and encoding).

use xr_io::IoError;

use crate::bad;
use crate::format::Filter;

pub(crate) const DEFLATE: u16 = 1;
pub(crate) const SHUFFLE: u16 = 2;
pub(crate) const FLETCHER32: u16 = 3;

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
            id => return Err(IoError::Unsupported(format!("HDF5 filter {id}"))),
        };
    }
    Ok(data)
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

    #[test]
    fn shuffle_round_trips() {
        let data: Vec<u8> = (0..24).collect();
        for size in [1, 2, 4, 8, 3] {
            assert_eq!(unshuffle(&shuffle(&data, size), size), data, "size {size}");
        }
        assert_eq!(shuffle(&[1, 2, 3, 4], 2), vec![1, 3, 2, 4]);
    }
}
