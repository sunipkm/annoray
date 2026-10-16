//! Bit- and byte-level readers for decoders.

use crate::Error;

/// An error in a named codec.
#[must_use]
pub fn corrupt(codec: &str) -> Error {
    Error::Corrupt(codec.to_owned())
}

/// Reads bits most significant first.
#[derive(Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits.
    pos: usize,
    codec: &'static str,
}

impl<'a> BitReader<'a> {
    /// A reader at the start of `data`, naming `codec` in errors.
    #[must_use]
    pub fn new(data: &'a [u8], codec: &'static str) -> Self {
        Self {
            data,
            pos: 0,
            codec,
        }
    }

    /// The next bit.
    ///
    /// # Errors
    /// [`Error::Corrupt`] past the end.
    pub fn bit(&mut self) -> Result<bool, Error> {
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
    ///
    /// # Errors
    /// [`Error::Corrupt`] past the end or for `n > 64`.
    pub fn bits(&mut self, n: u32) -> Result<u64, Error> {
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
    pub fn align(&mut self) {
        self.pos = self.pos.div_ceil(8).saturating_mul(8);
    }

    /// Whether every byte has been read.
    #[must_use]
    pub fn at_end(&self) -> bool {
        self.pos / 8 >= self.data.len()
    }
}

/// A little-endian `u32` at `at`.
///
/// # Errors
/// [`Error::Corrupt`] past the end.
pub fn le32(data: &[u8], at: usize, codec: &str) -> Result<u32, Error> {
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
///
/// # Errors
/// [`Error::Corrupt`] past the end.
pub fn be(data: &[u8], at: usize, n: usize, codec: &str) -> Result<u64, Error> {
    let b = at
        .checked_add(n)
        .and_then(|end| data.get(at..end))
        .ok_or_else(|| corrupt(codec))?;
    Ok(b.iter().fold(0u64, |acc, x| (acc << 8) | u64::from(*x)))
}
