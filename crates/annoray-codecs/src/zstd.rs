//! Zstandard frames (pure Rust, `ruzstd`), behind the `zstd` feature; without it, both
//! functions report [`Error::Unsupported`].

use crate::Error;

/// One Zstandard frame (or several, concatenated), producing at most `limit` bytes.
///
/// # Errors
/// [`Error::Corrupt`] for malformed frames; [`Error::TooLarge`] beyond `limit`.
#[cfg(feature = "zstd")]
pub fn decompress(input: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    use std::io::Read;
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(input)
        .map_err(|e| Error::Corrupt(format!("zstd: {e}")))?;
    let mut out = Vec::with_capacity(limit);
    let cap = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    (&mut decoder)
        .take(cap)
        .read_to_end(&mut out)
        .map_err(|e| Error::Corrupt(format!("zstd: {e}")))?;
    if out.len() > limit {
        return Err(Error::TooLarge("zstd".into()));
    }
    Ok(out)
}

/// Without the `zstd` feature: [`Error::Unsupported`].
///
/// # Errors
/// Always.
#[cfg(not(feature = "zstd"))]
pub fn decompress(_input: &[u8], _limit: usize) -> Result<Vec<u8>, Error> {
    Err(Error::Unsupported(
        "Zstandard (enable the `zstd` feature)".into(),
    ))
}

/// One Zstandard frame (at Zstandard's fastest level, the one `ruzstd` implements), with its
/// content size in the header: decoders that size their output from it (the HDF5 zstd plugin,
/// numcodecs) need it, and `ruzstd` leaves it out.
#[cfg(feature = "zstd")]
#[must_use]
pub fn compress(data: &[u8]) -> Vec<u8> {
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

/// Without the `zstd` feature: empty (callers check [`available`] first).
#[cfg(not(feature = "zstd"))]
#[must_use]
pub fn compress(_data: &[u8]) -> Vec<u8> {
    Vec::new()
}

/// Whether Zstandard is compiled in.
#[must_use]
pub const fn available() -> bool {
    cfg!(feature = "zstd")
}

#[cfg(all(test, feature = "zstd"))]
mod tests {
    use super::*;

    #[test]
    fn frames_carry_their_content_size() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i % 13) as u8).collect();
        let frame = compress(&data);
        assert_eq!(frame[4] & 0xC0, 0xC0);
        assert_eq!(frame[6..14], (data.len() as u64).to_le_bytes());
        assert_eq!(decompress(&frame, data.len()).unwrap(), data);
        assert!(decompress(&frame, 10).is_err());
    }
}
