//! Blosc (version 1 frames, HDF5 filter 32001) with its BloscLZ, LZ4, zlib and Zstandard
//! codecs and its byte- and bit-shuffles.

use xr_io::IoError;

use super::{bitshuffle, corrupt, le32};
use crate::ck::Ck;
use crate::filters::unshuffle;

const CODEC: &str = "blosc";
const MEMCPYED: u8 = 0x02;
const SHUFFLE: u8 = 0x01;
const BITSHUFFLE: u8 = 0x04;
const DONT_SPLIT: u8 = 0x10;
const MAX_SPLITS: usize = 16;
const MIN_BUFFERSIZE: usize = 128;
const MAX_DISTANCE: usize = 8191;

fn err() -> IoError {
    corrupt(CODEC)
}

/// A checked size.
fn z(c: Ck) -> Result<usize, IoError> {
    c.usize().ok_or_else(err)
}

/// BloscLZ (a FastLZ derivative), producing exactly `want` bytes.
pub(crate) fn blosclz(input: &[u8], want: usize) -> Result<Vec<u8>, IoError> {
    let mut out: Vec<u8> = Vec::with_capacity(want);
    let mut ip = 0usize;
    let next = |ip: &mut usize| -> Result<usize, IoError> {
        let b = input.get(*ip).copied().ok_or_else(err)?;
        *ip = z(Ck::from(*ip) + 1usize)?;
        Ok(usize::from(b))
    };
    let mut ctrl = next(&mut ip)? & 31;
    loop {
        if ctrl >= 32 {
            // A match: length and a 13-bit (or, escaped, 16-bit) distance.
            let mut len = Ck::from(ctrl >> 5) - 1usize;
            let ofs = Ck::from((ctrl & 31) << 8);
            if z(len)? == 6 {
                loop {
                    let code = next(&mut ip)?;
                    len = len + code;
                    if code != 255 {
                        break;
                    }
                }
            }
            let code = next(&mut ip)?;
            let len = z(len + 3usize)?;
            let mut ofs = z(ofs + code)?;
            if code == 255 && ofs == (31 << 8) + 255 {
                let hi = next(&mut ip)?;
                let lo = next(&mut ip)?;
                ofs = z(Ck::from(hi) * 256usize + lo + MAX_DISTANCE)?;
            }
            // The reference is `ofs + 1` bytes back.
            let from = z(Ck::from(out.len()) - ofs - 1usize)?;
            if z(Ck::from(out.len()) + len)? > want {
                return Err(err());
            }
            for i in 0..len {
                let b = *out.get(z(Ck::from(from) + i)?).ok_or_else(err)?;
                out.push(b);
            }
        } else {
            // A run of ctrl + 1 literals.
            let end = z(Ck::from(ip) + ctrl + 1usize)?;
            let lit = input.get(ip..end).ok_or_else(err)?;
            if z(Ck::from(out.len()) + lit.len())? > want {
                return Err(err());
            }
            out.extend_from_slice(lit);
            ip = end;
        }
        if ip >= input.len() {
            break;
        }
        ctrl = next(&mut ip)?;
    }
    if out.len() != want {
        return Err(err());
    }
    Ok(out)
}

/// One compressed stream of `want` bytes with the frame's codec.
fn stream(codec: u8, src: &[u8], want: usize) -> Result<Vec<u8>, IoError> {
    let out = match codec {
        0 => blosclz(src, want)?,
        #[cfg(feature = "lz4")]
        1 => {
            let mut buf = vec![0u8; want];
            let got = lz4_flex::block::decompress_into(src, &mut buf)
                .map_err(|e| crate::bad(format!("blosc lz4: {e}")))?;
            buf.truncate(got);
            buf
        }
        3 => miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(src, want)
            .map_err(|e| crate::bad(format!("blosc zlib: {e:?}")))?,
        #[cfg(feature = "zstd")]
        4 => crate::filters::zstd_decompress(src, want)?,
        2 => return Err(IoError::Unsupported("Blosc with Snappy".into())),
        other => {
            return Err(IoError::Unsupported(format!(
                "Blosc codec {other} (LZ4 and Zstandard need the `lz4`/`zstd` features)"
            )));
        }
    };
    if out.len() != want {
        return Err(err());
    }
    Ok(out)
}

/// Decompress a Blosc frame into at most `limit` bytes.
pub(crate) fn decompress(src: &[u8], limit: usize) -> Result<Vec<u8>, IoError> {
    let header = src.get(..16).ok_or_else(err)?;
    let version = header.first().copied().ok_or_else(err)?;
    let flags = header.get(2).copied().ok_or_else(err)?;
    let typesize = usize::from(header.get(3).copied().ok_or_else(err)?);
    let nbytes = usize::try_from(le32(src, 4, CODEC)?).map_err(|_| err())?;
    let blocksize = usize::try_from(le32(src, 8, CODEC)?).map_err(|_| err())?;
    if nbytes > limit {
        return Err(crate::bad("blosc: chunk larger than expected"));
    }
    if flags & MEMCPYED != 0 {
        return Ok(src
            .get(16..z(Ck::from(nbytes) + 16usize)?)
            .ok_or_else(err)?
            .to_vec());
    }
    if blocksize == 0 || typesize == 0 {
        return Err(err());
    }
    let codec = flags >> 5;
    let nblocks = nbytes.div_ceil(blocksize);
    let leftover = nbytes.checked_rem(blocksize).ok_or_else(err)?;
    let mut out = Vec::with_capacity(nbytes);
    for j in 0..nblocks {
        let last_partial = z(Ck::from(j) + 1usize)? == nblocks && leftover > 0;
        let bsize = if last_partial { leftover } else { blocksize };
        let mut ip = usize::try_from(le32(src, z(Ck::from(j) * 4usize + 16usize)?, CODEC)?)
            .map_err(|_| err())?;
        let split = flags & DONT_SPLIT == 0
            && typesize <= MAX_SPLITS
            && bsize.checked_div(typesize).unwrap_or(0) >= MIN_BUFFERSIZE
            && !last_partial;
        let nstreams = if split { typesize } else { 1 };
        let neblock = bsize.checked_div(nstreams).ok_or_else(err)?;
        let mut block = Vec::with_capacity(bsize);
        for _ in 0..nstreams {
            let csize = usize::try_from(le32(src, ip, CODEC)?).map_err(|_| err())?;
            ip = z(Ck::from(ip) + 4usize)?;
            let data = src
                .get(ip..ip.checked_add(csize).ok_or_else(err)?)
                .ok_or_else(err)?;
            if csize == neblock {
                block.extend_from_slice(data);
            } else {
                block.extend(stream(codec, data, neblock)?);
            }
            ip = z(Ck::from(ip) + csize)?;
        }
        // Undo the shuffle of this block.
        let block = if flags & BITSHUFFLE != 0 && typesize > 0 {
            let n = bsize.checked_div(typesize).ok_or_else(err)?;
            // Format version 2 bit-shuffles only blocks of a multiple of 8 elements; later
            // versions bit-shuffle the multiple of 8 and copy the rest.
            let whole = if version <= 2 && !n.is_multiple_of(8) {
                0
            } else {
                n.saturating_sub(n % 8)
            };
            let cut = z(Ck::from(whole) * typesize)?;
            let mut b =
                bitshuffle::untranspose(block.get(..cut).ok_or_else(err)?, whole, typesize)?;
            b.extend_from_slice(block.get(cut..).unwrap_or_default());
            b
        } else if flags & SHUFFLE != 0 && typesize > 1 {
            unshuffle(&block, typesize)
        } else {
            block
        };
        out.extend(block);
    }
    Ok(out)
}

/// A Blosc frame of `data` (elements of `typesize` bytes) as one block and one stream, with
/// `codec` (1 LZ4, 3 zlib, 4 Zstandard) after the given shuffle; stored raw when that does
/// not shrink it.
pub(crate) fn compress(
    data: &[u8],
    typesize: usize,
    codec: u8,
    level: u8,
    shuffle: xr_core::BloscShuffle,
) -> Result<Vec<u8>, IoError> {
    let big = || crate::bad("blosc: chunk too large");
    let nbytes = u32::try_from(data.len()).map_err(|_| big())?;
    let ts = u8::try_from(typesize).map_err(|_| big())?;
    let shuffled = match shuffle {
        xr_core::BloscShuffle::Byte if typesize > 1 => crate::filters::shuffle(data, typesize),
        xr_core::BloscShuffle::Bit if typesize > 0 => {
            // Format version 2 (what this writes): only whole multiples of 8 elements.
            let n = data.len().checked_div(typesize).unwrap_or(0);
            let whole = if n.is_multiple_of(8) { n } else { 0 };
            let cut = z(Ck::from(whole) * typesize)?;
            let mut b =
                bitshuffle::transpose(data.get(..cut).unwrap_or_default(), whole, typesize)?;
            b.extend_from_slice(data.get(cut..).unwrap_or_default());
            b
        }
        _ => data.to_vec(),
    };
    let packed: Vec<u8> = match codec {
        #[cfg(feature = "lz4")]
        1 => lz4_flex::block::compress(&shuffled),
        3 => miniz_oxide::deflate::compress_to_vec_zlib(&shuffled, level.min(9)),
        #[cfg(feature = "zstd")]
        4 => crate::filters::zstd_compress(&shuffled),
        other => {
            return Err(IoError::Unsupported(format!(
                "writing Blosc codec {other} (LZ4 and Zstandard need the `lz4`/`zstd` features)"
            )));
        }
    };
    let mut flags = DONT_SPLIT | (codec << 5);
    flags |= match shuffle {
        xr_core::BloscShuffle::Byte => SHUFFLE,
        xr_core::BloscShuffle::Bit => BITSHUFFLE,
        _ => 0,
    };
    let mut out = Vec::with_capacity(packed.len().saturating_add(28));
    if packed.len() >= data.len() || data.is_empty() {
        // Not worth compressing: a copy of the data.
        out.extend_from_slice(&[2, 1, MEMCPYED | (codec << 5), ts]);
        out.extend_from_slice(&nbytes.to_le_bytes());
        out.extend_from_slice(&nbytes.to_le_bytes());
        out.extend_from_slice(&nbytes.saturating_add(16).to_le_bytes());
        out.extend_from_slice(data);
        return Ok(out);
    }
    let csize = u32::try_from(packed.len()).map_err(|_| big())?;
    // Header, one block start (just past it), the stream's size, the stream.
    out.extend_from_slice(&[2, 1, flags, ts]);
    out.extend_from_slice(&nbytes.to_le_bytes());
    out.extend_from_slice(&nbytes.to_le_bytes());
    out.extend_from_slice(&csize.saturating_add(24).to_le_bytes());
    out.extend_from_slice(&20u32.to_le_bytes());
    out.extend_from_slice(&csize.to_le_bytes());
    out.extend_from_slice(&packed);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blosclz_literals_and_matches() {
        // 3 literals "abc", then a match of length (4 - 1) + 3 = 6 at distance 3 (ofs 2 means
        // 3 back), then one literal "d".
        let enc = [2, b'a', b'b', b'c', 4 << 5, 2, 0, b'd'];
        assert_eq!(blosclz(&enc, 10).unwrap(), b"abcabcabcd");
        assert!(blosclz(&enc, 9).is_err());
        assert!(blosclz(&[1 << 5, 9], 10).is_err());
    }

    #[test]
    fn rejects_garbage_without_panicking() {
        for input in [
            &[0u8; 3][..],
            &[2, 1, 0, 8, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0xFF; 40],
        ] {
            let _ = decompress(input, 1 << 20);
        }
    }
}
