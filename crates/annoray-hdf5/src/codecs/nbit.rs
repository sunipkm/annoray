//! HDF5's N-bit filter (5): only each value's `precision` significant bits are stored, packed
//! most significant first. Atomic (integer and float) types are supported.

use crate::Error;

use super::{BitReader, corrupt};

const CODEC: &str = "n-bit";
const ATOMIC: u32 = 1;
const NOOP: u32 = 4;

fn err() -> Error {
    corrupt(CODEC)
}

/// Decode an N-bit chunk into at most `limit` bytes. `cd`: parameter count, "needs no
/// compression" flag, element count, then the type: class, size, byte order, precision,
/// offset.
pub(crate) fn decompress(data: &[u8], cd: &[u32], limit: usize) -> Result<Vec<u8>, Error> {
    let get = |i: usize| -> Result<usize, Error> {
        cd.get(i)
            .copied()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(err)
    };
    if get(1)? == 1 || cd.get(3) == Some(&NOOP) {
        // Full-precision data is stored as it is.
        return Ok(data.to_vec());
    }
    if cd.get(3) != Some(&ATOMIC) {
        return Err(Error::Unsupported(
            "N-bit compression of array or compound types".into(),
        ));
    }
    let (n, size, big_endian, precision, offset) =
        (get(2)?, get(4)?, get(5)? == 1, get(6)?, get(7)?);
    let total = n.checked_mul(size).ok_or_else(err)?;
    if total > limit
        || size == 0
        || size > 8
        || precision == 0
        || precision.checked_add(offset) > size.checked_mul(8)
    {
        return Err(err());
    }
    let mut r = BitReader::new(data, CODEC);
    let mut out = Vec::with_capacity(total);
    let precision = u32::try_from(precision).map_err(|_| err())?;
    for _ in 0..n {
        // The significant bits, put back at their offset (other bits are zero).
        let v = r.bits(precision)? << offset;
        let bytes = v.to_le_bytes();
        let b = bytes.get(..size).ok_or_else(err)?;
        if big_endian {
            out.extend(b.iter().rev());
        } else {
            out.extend_from_slice(b);
        }
    }
    Ok(out)
}
