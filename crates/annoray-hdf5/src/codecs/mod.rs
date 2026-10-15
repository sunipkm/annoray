//! Our own decoders for HDF5 filters: bzip2, Blosc (with BloscLZ), Bitshuffle, scale-offset
//! and N-bit. Each treats its input as untrusted: reads are bounds-checked, arithmetic is
//! checked, and output is limited to the chunk's expected size.

pub(crate) mod bitshuffle;
pub(crate) mod blosc;
pub(crate) mod bzip2;
pub(crate) mod nbit;
pub(crate) mod scaleoffset;

use crate::Error;

use crate::bad;

/// An error in a named codec.
pub(crate) fn corrupt(codec: &str) -> Error {
    bad(format!("{codec}: corrupt chunk"))
}

/// Reads bits most significant first.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits.
    pos: usize,
    codec: &'static str,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8], codec: &'static str) -> Self {
        Self {
            data,
            pos: 0,
            codec,
        }
    }

    pub(crate) fn bit(&mut self) -> Result<bool, Error> {
        let byte = self
            .data
            .get(self.pos / 8)
            .copied()
            .ok_or_else(|| corrupt(self.codec))?;
        let b = byte.wrapping_shl(u32::try_from(self.pos % 8).unwrap_or(0)) & 0x80 != 0;
        self.pos = self.pos.checked_add(1).ok_or_else(|| corrupt(self.codec))?;
        Ok(b)
    }

    /// `n` (at most 64) bits as a big-endian number.
    pub(crate) fn bits(&mut self, n: u32) -> Result<u64, Error> {
        if n > 64 {
            return Err(corrupt(self.codec));
        }
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | u64::from(self.bit()?);
        }
        Ok(v)
    }

    /// Skip to the next byte boundary.
    pub(crate) fn align(&mut self) {
        self.pos = self.pos.div_ceil(8).saturating_mul(8);
    }

    pub(crate) fn at_end(&self) -> bool {
        self.pos / 8 >= self.data.len()
    }
}

/// A little-endian `u32` at `at`.
pub(crate) fn le32(data: &[u8], at: usize, codec: &str) -> Result<u32, Error> {
    let b = at
        .checked_add(4)
        .and_then(|end| data.get(at..end))
        .ok_or_else(|| corrupt(codec))?;
    Ok(u32::from_le_bytes([
        b.first().copied().unwrap_or(0),
        b.get(1).copied().unwrap_or(0),
        b.get(2).copied().unwrap_or(0),
        b.get(3).copied().unwrap_or(0),
    ]))
}

/// A big-endian unsigned number of `n` bytes at `at`.
pub(crate) fn be(data: &[u8], at: usize, n: usize, codec: &str) -> Result<u64, Error> {
    let b = at
        .checked_add(n)
        .and_then(|end| data.get(at..end))
        .ok_or_else(|| corrupt(codec))?;
    Ok(b.iter().fold(0u64, |acc, x| (acc << 8) | u64::from(*x)))
}
