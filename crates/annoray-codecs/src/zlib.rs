//! zlib (RFC 1950) and gzip (RFC 1952) streams around raw deflate (RFC 1951), via
//! `miniz_oxide`. Decoding is bounded by a limit; gzip members may be concatenated, and
//! their CRCs and lengths are checked.

use miniz_oxide::inflate::TINFLStatus;
use miniz_oxide::inflate::core::{DecompressorOxide, decompress, inflate_flags};

use crate::Error;
use crate::bits::le32;
use crate::crc::crc32;

/// Decompress a zlib stream into at most `limit` bytes.
///
/// # Errors
/// [`Error::Corrupt`] for malformed input; [`Error::TooLarge`] beyond `limit`.
pub fn zlib_decompress(data: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(data, limit).map_err(|e| {
        if e.status == TINFLStatus::HasMoreOutput {
            Error::TooLarge("zlib".into())
        } else {
            Error::Corrupt(format!("zlib ({:?})", e.status))
        }
    })
}

/// A zlib stream of `data` at `level` (0–9).
#[must_use]
pub fn zlib_compress(data: &[u8], level: u8) -> Vec<u8> {
    miniz_oxide::deflate::compress_to_vec_zlib(data, level.min(9))
}

/// Raw deflate from `data` into `out` (appending at most `room` bytes): the bytes of `data`
/// consumed.
fn inflate_raw(data: &[u8], out: &mut Vec<u8>, room: usize) -> Result<usize, Error> {
    let start = out.len();
    out.resize(
        start
            .checked_add(room)
            .ok_or_else(|| Error::TooLarge("gzip".into()))?,
        0,
    );
    let mut state = DecompressorOxide::new();
    let flags = inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
    let (status, read, written) = decompress(&mut state, data, out, start, flags);
    out.truncate(start.saturating_add(written));
    match status {
        TINFLStatus::Done => Ok(read),
        TINFLStatus::HasMoreOutput => Err(Error::TooLarge("gzip".into())),
        other => Err(Error::Corrupt(format!("gzip ({other:?})"))),
    }
}

/// Decompress gzip data (one or more members) into at most `limit` bytes.
///
/// # Errors
/// [`Error::Corrupt`] for malformed members or a CRC/length mismatch; [`Error::TooLarge`]
/// beyond `limit`.
pub fn gzip_decompress(data: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    let bad = || Error::Corrupt("gzip".into());
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < data.len() {
        let head = data
            .get(at..at.checked_add(10).ok_or_else(bad)?)
            .ok_or_else(bad)?;
        if head.get(..3) != Some(&[0x1F, 0x8B, 8][..]) {
            return Err(bad());
        }
        let flg = head.get(3).copied().unwrap_or(0);
        let mut p = at.checked_add(10).ok_or_else(bad)?;
        if flg & 0x04 != 0 {
            let xlen = data
                .get(p..p.checked_add(2).ok_or_else(bad)?)
                .ok_or_else(bad)?;
            let n = usize::from(u16::from_le_bytes([
                xlen.first().copied().unwrap_or(0),
                xlen.get(1).copied().unwrap_or(0),
            ]));
            p = p
                .checked_add(2)
                .and_then(|q| q.checked_add(n))
                .ok_or_else(bad)?;
        }
        for bit in [0x08u8, 0x10] {
            if flg & bit != 0 {
                let rest = data.get(p..).ok_or_else(bad)?;
                let z = rest.iter().position(|b| *b == 0).ok_or_else(bad)?;
                p = p
                    .checked_add(z)
                    .and_then(|q| q.checked_add(1))
                    .ok_or_else(bad)?;
            }
        }
        if flg & 0x02 != 0 {
            p = p.checked_add(2).ok_or_else(bad)?;
        }
        let body = data.get(p..).ok_or_else(bad)?;
        let before = out.len();
        let room = limit.saturating_sub(before);
        let read = inflate_raw(body, &mut out, room)?;
        let trailer = p.checked_add(read).ok_or_else(bad)?;
        let crc = le32(data, trailer, "gzip")?;
        let size = le32(data, trailer.checked_add(4).ok_or_else(bad)?, "gzip")?;
        let member = out.get(before..).unwrap_or_default();
        #[allow(clippy::cast_possible_truncation)] // ISIZE is the length modulo 2^32
        if crc32(member) != crc || member.len() as u32 != size {
            return Err(Error::Corrupt("gzip: CRC or length mismatch".into()));
        }
        at = trailer.checked_add(8).ok_or_else(bad)?;
    }
    Ok(out)
}

/// A single-member gzip stream of `data` at `level` (0–9), without a file name or time.
#[must_use]
pub fn gzip_compress(data: &[u8], level: u8) -> Vec<u8> {
    let body = miniz_oxide::deflate::compress_to_vec(data, level.min(9));
    let mut out = Vec::with_capacity(body.len().saturating_add(18));
    out.extend_from_slice(&[0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 0xFF]);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(data).to_le_bytes());
    #[allow(clippy::cast_possible_truncation)] // ISIZE is the length modulo 2^32
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_limits() {
        let data: Vec<u8> = (0..20_000u32).map(|i| (i % 251) as u8).collect();
        let z = zlib_compress(&data, 6);
        assert_eq!(zlib_decompress(&z, data.len()).unwrap(), data);
        assert!(matches!(zlib_decompress(&z, 10), Err(Error::TooLarge(_))));
        let g = gzip_compress(&data, 6);
        assert_eq!(gzip_decompress(&g, data.len()).unwrap(), data);
        assert!(matches!(gzip_decompress(&g, 10), Err(Error::TooLarge(_))));
        // Two members.
        let mut two = g.clone();
        two.extend_from_slice(&gzip_compress(b"tail", 1));
        let mut want = data.clone();
        want.extend_from_slice(b"tail");
        assert_eq!(gzip_decompress(&two, want.len()).unwrap(), want);
        // A flipped byte in the body or the CRC is caught.
        let mut bad = g.clone();
        let n = bad.len();
        bad[n - 6] ^= 1;
        assert!(gzip_decompress(&bad, data.len()).is_err());
        assert!(gzip_decompress(&[0x1F, 0x8B], 100).is_err());
    }
}
