//! HDF5's szip filter (4), decoded as libaec decodes it: Adaptive Entropy Coding (CCSDS
//! 121.0-B) behind the szip library's interface. A chunk is its decoded size (`u32`,
//! little-endian) followed by the coded stream. The filter's parameters: the szip options
//! mask, pixels per block, bits per pixel and pixels per scanline.
//!
//! The stream is a sequence of coded data sets (CDS), one per block of samples; a reference
//! sample interval (RSI) groups `ceil(pixels per scanline / pixels per block)` blocks. With
//! the nearest-neighbour option, the first sample of each RSI is a reference and the others
//! are mapped prediction differences. Pixels of 32 or 64 bits are coded as bytes, the bytes
//! of each position in a pixel together; scanlines that do not fill whole blocks are padded.

use crate::Error;

use super::corrupt;

const CODEC: &str = "szip";

/// The szip options-mask bits libaec honours (`szlib.h`).
const MSB: u32 = 16;
const NN: u32 = 32;

/// A zero-block count of 5 stands for "to the end of the segment" (Remainder Of Segment).
const ROS: u32 = 5;

/// The second-extension option codes pairs `(a, b)` as `(a + b)(a + b + 1) / 2 + b`, for
/// `a + b` up to 12: entries up to this index.
const SE_TABLE_SIZE: u32 = 90;

fn err() -> Error {
    corrupt(CODEC)
}

/// Bits, most significant first, through a 64-bit window.
struct Bits<'a> {
    data: &'a [u8],
    next: usize,
    /// Valid bits at the top of `acc`.
    acc: u64,
    n: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            next: 0,
            acc: 0,
            n: 0,
        }
    }

    fn refill(&mut self) {
        while self.n <= 56 {
            let Some(b) = self.data.get(self.next) else {
                return;
            };
            self.acc |= u64::from(*b) << 56u32.wrapping_sub(self.n);
            self.n = self.n.wrapping_add(8);
            self.next = self.next.saturating_add(1);
        }
    }

    /// `k` bits (at most 32) as a number.
    fn get(&mut self, k: u32) -> Result<u32, Error> {
        if k == 0 {
            return Ok(0);
        }
        if self.n < k {
            self.refill();
            if self.n < k {
                return Err(err());
            }
        }
        let v = self.acc >> 64u32.wrapping_sub(k);
        self.acc = self.acc.checked_shl(k).unwrap_or(0);
        self.n = self.n.wrapping_sub(k);
        u32::try_from(v).map_err(|_| err())
    }

    /// A fundamental sequence: the number of zeros before the next one.
    fn fs(&mut self) -> Result<u32, Error> {
        let mut count: u32 = 0;
        loop {
            if self.n == 0 {
                self.refill();
                if self.n == 0 {
                    return Err(err());
                }
            }
            let lz = self.acc.leading_zeros();
            if lz < self.n {
                // lz < n ≤ 64.
                let used = lz.wrapping_add(1);
                self.acc = self.acc.checked_shl(used).unwrap_or(0);
                self.n = self.n.wrapping_sub(used);
                return count.checked_add(lz).ok_or_else(err);
            }
            count = count.checked_add(self.n).ok_or_else(err)?;
            self.acc = 0;
            self.n = 0;
        }
    }
}

/// The second-extension table: for each code `m`, the pair's sum and the code of its first
/// pair (`(sum, first)`), as libaec builds it.
fn se_table() -> Vec<(u32, u32)> {
    let mut t = Vec::with_capacity(91);
    let mut k: u32 = 0;
    for i in 0..13u32 {
        let first = k;
        for _ in 0..=i {
            t.push((i, first));
            k = k.saturating_add(1);
        }
    }
    t
}

/// Decode `samples` samples of `bps` bits (at most 32) from an AEC stream; blocks of
/// `block` samples, `rsi` blocks per reference sample interval, `pp`: the samples are
/// prediction-coded. Each RSI is undone as it completes.
fn aec_decode(
    input: &[u8],
    bps: u32,
    block: usize,
    rsi: usize,
    pp: bool,
    samples: usize,
) -> Result<Vec<u32>, Error> {
    if bps == 0 || bps > 32 || block == 0 || rsi == 0 {
        return Err(err());
    }
    let id_len: u32 = if bps > 16 {
        5
    } else if bps > 8 {
        4
    } else {
        3
    };
    let uncompressed = (1u32 << id_len).wrapping_sub(1);
    let xmax: u32 = if bps == 32 {
        u32::MAX
    } else {
        (1u32 << bps).wrapping_sub(1)
    };
    let rsi_size = rsi.checked_mul(block).ok_or_else(err)?;
    let se = se_table();
    let mut bits = Bits::new(input);
    let mut out: Vec<u32> = Vec::with_capacity(samples);
    // Where the current RSI starts in `out`.
    let mut start = 0usize;
    let finish_rsi = |out: &mut Vec<u32>, start: usize| {
        if pp {
            undo_prediction(out.get_mut(start..).unwrap_or_default(), xmax);
        }
    };
    while out.len() < samples {
        let used = out.len().saturating_sub(start);
        let reference = pp && used == 0;
        let id = bits.get(id_len)?;
        if id == 0 {
            // Low entropy: second extension or zero blocks.
            let second = bits.get(1)? == 1;
            if reference {
                out.push(bits.get(bps)?);
            }
            if second {
                let mut i = usize::from(reference);
                while i < block {
                    let m = bits.fs()?;
                    if m > SE_TABLE_SIZE {
                        return Err(err());
                    }
                    let (sum, first) = se
                        .get(usize::try_from(m).map_err(|_| err())?)
                        .copied()
                        .ok_or_else(err)?;
                    let d1 = m.checked_sub(first).ok_or_else(err)?;
                    if i & 1 == 0 {
                        out.push(sum.checked_sub(d1).ok_or_else(err)?);
                        i = i.saturating_add(1);
                    }
                    out.push(d1);
                    i = i.saturating_add(1);
                }
            } else {
                let mut blocks = bits.fs()?.checked_add(1).ok_or_else(err)?;
                if blocks == ROS {
                    let b = used.checked_div(block).unwrap_or(0);
                    let left = rsi.saturating_sub(b);
                    let to_64 = 64usize.saturating_sub(b % 64);
                    blocks = u32::try_from(left.min(to_64)).map_err(|_| err())?;
                } else if blocks > ROS {
                    blocks = blocks.saturating_sub(1);
                }
                let zeros = usize::try_from(blocks)
                    .ok()
                    .and_then(|b| b.checked_mul(block))
                    .and_then(|z| z.checked_sub(usize::from(reference)))
                    .ok_or_else(err)?;
                if rsi_size.saturating_sub(used) < zeros {
                    return Err(err());
                }
                out.resize(out.len().saturating_add(zeros), 0);
            }
        } else if id == uncompressed {
            for _ in 0..block {
                out.push(bits.get(bps)?);
            }
        } else {
            // Sample splitting: the high bits as fundamental sequences, then `k` low bits each.
            let k = id.saturating_sub(1);
            if reference {
                out.push(bits.get(bps)?);
            }
            let n = block.saturating_sub(usize::from(reference));
            let from = out.len();
            for _ in 0..n {
                let hi = bits.fs()?;
                out.push(hi.checked_shl(k).ok_or_else(err)?);
            }
            if k > 0 {
                for v in out.get_mut(from..).unwrap_or_default() {
                    *v = v.wrapping_add(bits.get(k)?);
                }
            }
        }
        if out.len().saturating_sub(start) >= rsi_size {
            finish_rsi(&mut out, start);
            start = out.len();
        }
    }
    if start < out.len() {
        finish_rsi(&mut out, start);
    }
    out.truncate(samples);
    Ok(out)
}

/// Undo the unit-delay predictor and its mapping on one RSI (unsigned samples, libaec's
/// `FLUSH` with `xmin == 0`): the first sample is the reference.
fn undo_prediction(rsi: &mut [u32], xmax: u32) {
    let Some((first, rest)) = rsi.split_first_mut() else {
        return;
    };
    let mut data = *first;
    let med = (xmax / 2).wrapping_add(1);
    for d in rest {
        let delta = *d;
        let half = (delta >> 1).wrapping_add(delta & 1);
        let mask = if data & med != 0 { xmax } else { 0 };
        if half <= (mask ^ data) {
            // Even: +delta/2; odd: -(delta+1)/2.
            data = data.wrapping_add((delta >> 1) ^ (!((delta & 1).wrapping_sub(1))));
        } else {
            data = mask ^ delta;
        }
        *d = data;
    }
}

/// Decode an szip chunk into at most `limit` bytes. `cd`: options mask, pixels per block,
/// bits per pixel, pixels per scanline (`H5Z_SZIP_PARM_*`).
pub(crate) fn decompress(data: &[u8], cd: &[u32], limit: usize) -> Result<Vec<u8>, Error> {
    let param = |i: usize| cd.get(i).copied().ok_or_else(err);
    let (mask, ppb, bpp, ppsl) = (param(0)?, param(1)?, param(2)?, param(3)?);
    let size = usize::try_from(super::le32(data, 0, CODEC)?).map_err(|_| err())?;
    if size > limit {
        return Err(err());
    }
    let stream = data.get(4..).ok_or_else(err)?;
    let ppb = usize::try_from(ppb).map_err(|_| err())?;
    let ppsl = usize::try_from(ppsl).map_err(|_| err())?;
    if ppb == 0 || ppsl == 0 {
        return Err(err());
    }
    let interleave = bpp == 32 || bpp == 64;
    let bps = if interleave { 8 } else { bpp };
    let pixel: usize = if bps > 16 {
        4
    } else if bps > 8 {
        2
    } else {
        1
    };
    let rsi = ppsl.div_ceil(ppb);
    let padded = ppsl.checked_rem(ppb).is_some_and(|r| r != 0);
    // Samples to decode: whole padded scanlines, or the decoded size.
    let (samples, scanlines) = if padded {
        let lines = size.checked_div(pixel).unwrap_or(0).div_ceil(ppsl);
        let n = rsi
            .checked_mul(ppb)
            .and_then(|x| x.checked_mul(lines))
            .ok_or_else(err)?;
        (n, lines)
    } else {
        (size.checked_div(pixel).unwrap_or(0), 0)
    };
    // A corrupt header cannot ask for more than the chunk holds, padded.
    if samples
        > limit
            .saturating_mul(2)
            .saturating_add(ppb.saturating_mul(rsi))
    {
        return Err(err());
    }
    let values = aec_decode(stream, bps, ppb, rsi, mask & NN != 0, samples)?;
    let msb = mask & MSB != 0;
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(pixel));
    for v in &values {
        match (pixel, msb) {
            (1, _) => bytes.push(v.to_le_bytes()[0]),
            (2, false) => bytes.extend_from_slice(&v.to_le_bytes()[..2]),
            (2, true) => bytes.extend_from_slice(&v.to_be_bytes()[2..]),
            (_, false) => bytes.extend_from_slice(&v.to_le_bytes()),
            (_, true) => bytes.extend_from_slice(&v.to_be_bytes()),
        }
    }
    if padded {
        let line = ppsl.saturating_mul(pixel);
        let padded_line = rsi.saturating_mul(ppb).saturating_mul(pixel);
        let mut kept = Vec::with_capacity(scanlines.saturating_mul(line));
        for chunk in bytes.chunks(padded_line.max(1)) {
            kept.extend_from_slice(chunk.get(..line.min(chunk.len())).unwrap_or_default());
        }
        bytes = kept;
    }
    bytes.truncate(size);
    if bytes.len() != size {
        return Err(err());
    }
    if interleave {
        // Bytes were coded position by position: byte j of every pixel together.
        let width = usize::try_from(bpp / 8).map_err(|_| err())?;
        let n = size.checked_div(width).unwrap_or(0);
        if n == 0 {
            return Ok(bytes);
        }
        let mut out = vec![0u8; size];
        for (j, plane) in bytes.chunks(n).enumerate().take(width) {
            for (i, b) in plane.iter().enumerate() {
                if let Some(slot) = out.get_mut(i.saturating_mul(width).saturating_add(j)) {
                    *slot = *b;
                }
            }
        }
        bytes = out;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_extension_table() {
        let t = se_table();
        assert_eq!(t.len(), 91);
        assert_eq!(t[0], (0, 0));
        assert_eq!(t[1], (1, 1));
        assert_eq!(t[2], (1, 1));
        assert_eq!(t[3], (2, 3));
        assert_eq!(t[90], (12, 78));
    }

    #[test]
    fn prediction_round_trip() {
        // Map a sequence of 8-bit samples as libaec's preprocessor does, then undo it.
        let xs: Vec<u32> = vec![100, 101, 99, 99, 255, 0, 3, 128];
        let xmax = 255u32;
        let mut coded = vec![xs[0]];
        for w in xs.windows(2) {
            let (prev, x) = (i64::from(w[0]), i64::from(w[1]));
            let d = x - prev;
            let theta = prev.min(i64::from(xmax) - prev);
            let m = if d >= 0 && d <= theta {
                2 * d
            } else if d < 0 && -d <= theta {
                -2 * d - 1
            } else {
                theta + d.abs()
            };
            coded.push(u32::try_from(m).unwrap());
        }
        undo_prediction(&mut coded, xmax);
        assert_eq!(coded, xs);
    }

    #[test]
    fn truncated_stream_is_an_error() {
        let cd = [NN, 16, 16, 32];
        let mut data = 64u32.to_le_bytes().to_vec();
        data.extend_from_slice(&[0xF0, 0x00]);
        assert!(decompress(&data, &cd, 64).is_err());
    }
}
