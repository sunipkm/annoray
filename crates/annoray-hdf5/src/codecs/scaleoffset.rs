//! HDF5's scale-offset filter (6): values stored as `minbits`-bit offsets from a minimum,
//! for integers, and for floats scaled by `10^D` (the "D-scale" method).

use annoray_io::IoError;

use super::{BitReader, corrupt, le32};

const CODEC: &str = "scale-offset";
const HEADER: usize = 21;

fn err() -> IoError {
    corrupt(CODEC)
}

/// The filter's client values.
struct Params {
    scale_type: u32,
    scale: i32,
    n: usize,
    float: bool,
    size: usize,
    signed: bool,
    big_endian: bool,
    fill: Option<u64>,
}

impl Params {
    fn parse(cd: &[u32]) -> Result<Self, IoError> {
        let get = |i: usize| cd.get(i).copied().ok_or_else(err);
        let size = usize::try_from(get(4)?).map_err(|_| err())?;
        if !matches!(size, 1 | 2 | 4 | 8) {
            return Err(err());
        }
        // The fill value's bytes, in the writer's (little-endian) memory order.
        let fill = if get(7)? == 1 {
            let bytes: Vec<u8> = cd
                .get(8..)
                .unwrap_or_default()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let b = bytes.get(..size).ok_or_else(err)?;
            Some(
                b.iter()
                    .rev()
                    .fold(0u64, |acc, x| (acc << 8) | u64::from(*x)),
            )
        } else {
            None
        };
        Ok(Self {
            scale_type: get(0)?,
            scale: i32::from_ne_bytes(get(1)?.to_ne_bytes()),
            n: usize::try_from(get(2)?).map_err(|_| err())?,
            float: get(3)? == 1,
            size,
            signed: get(5)? == 1,
            big_endian: get(6)? == 1,
            fill,
        })
    }

    fn mask(&self) -> u64 {
        let bits = u32::try_from(self.size.saturating_mul(8)).unwrap_or(64);
        1u64.checked_shl(bits)
            .map_or(u64::MAX, |v| v.wrapping_sub(1))
    }

    /// A value of `size` bytes, sign-extended if signed.
    fn signed_value(&self, v: u64) -> i64 {
        let bits = u32::try_from(self.size.saturating_mul(8)).unwrap_or(64);
        if self.signed && bits < 64 && v.checked_shr(bits.saturating_sub(1)).unwrap_or(0) & 1 == 1 {
            i64::from_ne_bytes((v | !self.mask()).to_ne_bytes())
        } else {
            i64::from_ne_bytes(v.to_ne_bytes())
        }
    }

    fn put(&self, out: &mut Vec<u8>, v: u64) {
        let bytes = v.to_le_bytes();
        let b = bytes.get(..self.size).unwrap_or_default();
        if self.big_endian {
            out.extend(b.iter().rev());
        } else {
            out.extend_from_slice(b);
        }
    }
}

/// Decode a scale-offset chunk into at most `limit` bytes.
pub(crate) fn decompress(data: &[u8], cd: &[u32], limit: usize) -> Result<Vec<u8>, IoError> {
    let p = Params::parse(cd)?;
    let total = p.n.checked_mul(p.size).ok_or_else(err)?;
    if total > limit {
        return Err(crate::bad("scale-offset: chunk larger than expected"));
    }
    let minbits = le32(data, 0, CODEC)?;
    let minval_size = usize::from(*data.get(4).ok_or_else(err)?).min(8);
    let minval = data
        .get(5..5usize.saturating_add(minval_size))
        .ok_or_else(err)?
        .iter()
        .rev()
        .fold(0u64, |acc, x| (acc << 8) | u64::from(*x));
    let body = data.get(HEADER..).ok_or_else(err)?;
    let full = u32::try_from(p.size.saturating_mul(8)).map_err(|_| err())?;
    if minbits > full {
        return Err(err());
    }
    let mut out = Vec::with_capacity(total);
    if minbits == full {
        // Stored without compression, in the dataset's byte order.
        out.extend_from_slice(body.get(..total).ok_or_else(err)?);
        return Ok(out);
    }
    let fill_code = 1u64
        .checked_shl(minbits)
        .map_or(u64::MAX, |v| v.wrapping_sub(1));
    let mut r = BitReader::new(body, CODEC);
    for _ in 0..p.n {
        let packed = if minbits == 0 { 0 } else { r.bits(minbits)? };
        let v = match p.fill {
            Some(f) if packed == fill_code => f,
            _ if p.float => {
                if p.scale_type != 0 {
                    return Err(IoError::Unsupported("scale-offset E-scaling".into()));
                }
                // D-scale, as HDF5 computes it in the data's own type:
                // value = offset / 10^D + minimum.
                #[allow(clippy::cast_precision_loss)] // as HDF5 does
                if p.size == 4 {
                    let min =
                        f32::from_bits(u32::try_from(minval & 0xFFFF_FFFF).map_err(|_| err())?);
                    let x = p.signed_value(packed) as f32 / 10f32.powi(p.scale) + min;
                    u64::from(x.to_bits())
                } else {
                    let x = p.signed_value(packed) as f64 / 10f64.powi(p.scale)
                        + f64::from_bits(minval);
                    x.to_bits()
                }
            }
            _ => packed.wrapping_add(minval) & p.mask(),
        };
        p.put(&mut out, v);
    }
    Ok(out)
}
