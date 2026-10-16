//! LZ4 blocks (pure Rust, `lz4_flex`), behind the `lz4` feature, and numcodecs' `LZ4` framing
//! (a little-endian `u32` with the decoded size, then one block). Without the feature, every
//! function reports [`Error::Unsupported`].

use crate::Error;

#[cfg(not(feature = "lz4"))]
fn unsupported() -> Error {
    Error::Unsupported("LZ4 (enable the `lz4` feature)".into())
}

/// One LZ4 block that decodes to exactly `want` bytes.
///
/// # Errors
/// [`Error::Corrupt`] for malformed input or another size; [`Error::Unsupported`] without
/// the feature.
pub fn block_decompress(src: &[u8], want: usize) -> Result<Vec<u8>, Error> {
    #[cfg(feature = "lz4")]
    {
        let mut buf = vec![0u8; want];
        let got = lz4_flex::block::decompress_into(src, &mut buf)
            .map_err(|e| Error::Corrupt(format!("lz4: {e}")))?;
        if got != want {
            return Err(Error::Corrupt("lz4: wrong size".into()));
        }
        Ok(buf)
    }
    #[cfg(not(feature = "lz4"))]
    {
        let _ = (src, want);
        Err(unsupported())
    }
}

/// One LZ4 block of `data`.
///
/// # Errors
/// [`Error::Unsupported`] without the feature.
pub fn block_compress(data: &[u8]) -> Result<Vec<u8>, Error> {
    #[cfg(feature = "lz4")]
    {
        Ok(lz4_flex::block::compress(data))
    }
    #[cfg(not(feature = "lz4"))]
    {
        let _ = data;
        Err(unsupported())
    }
}

/// numcodecs' `LZ4`: the decoded size, then one block; at most `limit` bytes.
///
/// # Errors
/// As for [`block_decompress`]; [`Error::TooLarge`] beyond `limit`.
pub fn numcodecs_decompress(src: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    let n = usize::try_from(crate::bits::le32(src, 0, "lz4")?)
        .map_err(|_| Error::TooLarge("lz4".into()))?;
    if n > limit {
        return Err(Error::TooLarge("lz4".into()));
    }
    block_decompress(src.get(4..).unwrap_or_default(), n)
}

/// numcodecs' `LZ4` framing of `data`.
///
/// # Errors
/// [`Error::TooLarge`] beyond 4 GiB; [`Error::Unsupported`] without the feature.
pub fn numcodecs_compress(data: &[u8]) -> Result<Vec<u8>, Error> {
    let n = u32::try_from(data.len()).map_err(|_| Error::TooLarge("lz4".into()))?;
    let block = block_compress(data)?;
    let mut out = Vec::with_capacity(block.len().saturating_add(4));
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&block);
    Ok(out)
}

/// Whether LZ4 is compiled in.
#[must_use]
pub const fn available() -> bool {
    cfg!(feature = "lz4")
}

#[cfg(all(test, feature = "lz4"))]
mod tests {
    use super::*;

    #[test]
    fn numcodecs_framing_round_trips() {
        let data: Vec<u8> = (0..9000u32).map(|i| (i % 7) as u8).collect();
        let c = numcodecs_compress(&data).unwrap();
        assert_eq!(numcodecs_decompress(&c, data.len()).unwrap(), data);
        assert!(numcodecs_decompress(&c, 10).is_err());
    }
}
