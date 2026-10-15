//! Bounds-checked little-endian reading of HDF5 structures.

use crate::Error;

use crate::bad;

/// Sizes of file offsets and lengths, from the superblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sizes {
    pub(crate) offset: u8,
    pub(crate) length: u8,
}

/// A position in the file. Every read checks bounds.
#[derive(Debug, Clone)]
pub(crate) struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    pub(crate) sizes: Sizes,
}

/// The "undefined address" value (all bits set) at a given width.
pub(crate) fn undefined(width: u8) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (u32::from(width) * 8)).wrapping_sub(1)
    }
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(buf: &'a [u8], pos: usize, sizes: Sizes) -> Self {
        Self { buf, pos, sizes }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| bad("size overflow"))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| bad("unexpected end of data"))?;
        self.pos = end;
        Ok(s)
    }

    pub(crate) fn skip(&mut self, n: usize) -> Result<(), Error> {
        self.bytes(n).map(drop)
    }

    pub(crate) fn rest(&self) -> &'a [u8] {
        self.buf.get(self.pos..).unwrap_or_default()
    }

    pub(crate) fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.bytes(1)?.first().copied().unwrap_or(0))
    }

    pub(crate) fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_le_bytes(
            self.bytes(2)?.try_into().map_err(|_| bad("short read"))?,
        ))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().map_err(|_| bad("short read"))?,
        ))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().map_err(|_| bad("short read"))?,
        ))
    }

    /// An unsigned little-endian integer of `n` (≤ 8) bytes.
    pub(crate) fn uvar(&mut self, n: usize) -> Result<u64, Error> {
        if n > 8 {
            return Err(bad(format!("integer field of {n} bytes")));
        }
        let mut b = [0u8; 8];
        for (dst, src) in b.iter_mut().zip(self.bytes(n)?) {
            *dst = *src;
        }
        Ok(u64::from_le_bytes(b))
    }

    /// A file address; `None` for the undefined address.
    pub(crate) fn addr(&mut self) -> Result<Option<u64>, Error> {
        let w = self.sizes.offset;
        let v = self.uvar(usize::from(w))?;
        Ok((v != undefined(w)).then_some(v))
    }

    /// A length field.
    pub(crate) fn length(&mut self) -> Result<u64, Error> {
        self.uvar(usize::from(self.sizes.length))
    }

    /// A length field that must fit in memory.
    pub(crate) fn len_usize(&mut self) -> Result<usize, Error> {
        usize::try_from(self.length()?).map_err(|_| bad("length too large"))
    }

    /// Check a 4-byte signature.
    pub(crate) fn expect(&mut self, sig: &[u8; 4]) -> Result<(), Error> {
        let got = self.bytes(4)?;
        if got == sig {
            Ok(())
        } else {
            Err(bad(format!(
                "expected {:?} signature",
                String::from_utf8_lossy(sig)
            )))
        }
    }
}

/// `x` rounded up to a multiple of 8.
pub(crate) fn align8(x: usize) -> Option<usize> {
    x.checked_add(7).map(|v| v & !7)
}

/// Bytes needed to encode `n` (HDF5's `H5VM_limit_enc_size`: ⌊log2 n⌋/8 + 1).
pub(crate) fn enc_size(n: u64) -> usize {
    let log2 = if n == 0 {
        0
    } else {
        63usize.saturating_sub(n.leading_zeros() as usize)
    };
    (log2 / 8).saturating_add(1)
}
