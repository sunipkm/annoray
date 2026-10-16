//! Bit-transposition in the bitshuffle library's layout (used by Blosc's bit-shuffle and the
//! Bitshuffle HDF5 filter).

use crate::Error;
use crate::bits::corrupt;
use crate::ck::Ck;

const CODEC: &str = "bitshuffle";

fn err() -> Error {
    corrupt(CODEC)
}

/// A checked size.
fn z(c: Ck) -> Result<usize, Error> {
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
fn regroup(data: &[u8], n: usize, size: usize, planes_in: bool) -> Result<Vec<u8>, Error> {
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
/// # Errors
/// [`Error::Corrupt`] if `n` is not a multiple of 8 or `data` is shorter than `n * size`.
pub fn untranspose(data: &[u8], n: usize, size: usize) -> Result<Vec<u8>, Error> {
    regroup(data, n, size, true)
}

/// bitshuffle's transpose (the inverse of [`untranspose`]).
///
/// # Errors
/// As for [`untranspose`].
pub fn transpose(data: &[u8], n: usize, size: usize) -> Result<Vec<u8>, Error> {
    regroup(data, n, size, false)
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
