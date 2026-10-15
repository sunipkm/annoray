//! A bzip2 decoder (HDF5 filter 307, as netCDF-C writes it): Huffman-coded MTF/RLE2 symbols,
//! inverse Burrows–Wheeler transform, then RLE1, with every block's CRC checked.

use crate::Error;

use super::{BitReader, corrupt};

const CODEC: &str = "bzip2";
const BLOCK_MAGIC: u64 = 0x3141_5926_5359;
const END_MAGIC: u64 = 0x1772_4538_5090;
const MAX_CODE_LEN: usize = 20;
const GROUP_SIZE: u32 = 50;

fn err() -> Error {
    corrupt(CODEC)
}

/// The CRC bzip2 uses: CRC-32 with polynomial 0x04C11DB7, most significant bit first.
fn crc_update(mut crc: u32, byte: u8) -> u32 {
    crc ^= u32::from(byte) << 24;
    for _ in 0..8 {
        crc = if crc & 0x8000_0000 != 0 {
            (crc << 1) ^ 0x04C1_1DB7
        } else {
            crc << 1
        };
    }
    crc
}

/// Canonical Huffman decoding tables for one group (bzip2's limit/base/perm scheme).
struct Table {
    min_len: usize,
    limit: [i64; MAX_CODE_LEN + 2],
    base: [i64; MAX_CODE_LEN + 2],
    perm: Vec<u16>,
}

impl Table {
    fn new(lens: &[u8]) -> Result<Self, Error> {
        let min_len = usize::from(*lens.iter().min().ok_or_else(err)?);
        let max_len = usize::from(*lens.iter().max().ok_or_else(err)?);
        let mut perm = Vec::with_capacity(lens.len());
        for l in min_len..=max_len {
            for (sym, len) in lens.iter().enumerate() {
                if usize::from(*len) == l {
                    perm.push(u16::try_from(sym).map_err(|_| err())?);
                }
            }
        }
        let mut count = [0i64; MAX_CODE_LEN + 2];
        for len in lens {
            let slot = count.get_mut(usize::from(*len)).ok_or_else(err)?;
            *slot = slot.checked_add(1).ok_or_else(err)?;
        }
        let mut limit = [-1i64; MAX_CODE_LEN + 2];
        let mut base = [0i64; MAX_CODE_LEN + 2];
        let (mut code, mut index) = (0i64, 0i64);
        for l in min_len..=max_len {
            let n = count.get(l).copied().unwrap_or(0);
            // Codes of length `l` run from `code` to `code + n - 1`; `base` maps them to
            // positions in `perm`.
            let end = code.checked_add(n).ok_or_else(err)?;
            if let (Some(lim), Some(b)) = (limit.get_mut(l), base.get_mut(l)) {
                *lim = end.checked_sub(1).ok_or_else(err)?;
                *b = code.checked_sub(index).ok_or_else(err)?;
            }
            index = index.checked_add(n).ok_or_else(err)?;
            code = end.checked_mul(2).ok_or_else(err)?;
        }
        Ok(Self {
            min_len,
            limit,
            base,
            perm,
        })
    }

    fn decode(&self, r: &mut BitReader<'_>) -> Result<u16, Error> {
        let mut len = self.min_len;
        let mut code =
            i64::try_from(r.bits(u32::try_from(len).map_err(|_| err())?)?).map_err(|_| err())?;
        loop {
            if len > MAX_CODE_LEN {
                return Err(err());
            }
            if code <= self.limit.get(len).copied().unwrap_or(-1) {
                let i = code
                    .checked_sub(self.base.get(len).copied().unwrap_or(0))
                    .ok_or_else(err)?;
                return usize::try_from(i)
                    .ok()
                    .and_then(|i| self.perm.get(i))
                    .copied()
                    .ok_or_else(err);
            }
            code = code.checked_mul(2).ok_or_else(err)? | i64::from(r.bit()?);
            len = len.checked_add(1).ok_or_else(err)?;
        }
    }
}

/// Decode one block's bytes (after RLE1 decoding), appending them to `out`.
fn block(
    r: &mut BitReader<'_>,
    max_block: usize,
    out: &mut Vec<u8>,
    limit: usize,
) -> Result<(), Error> {
    let crc = u32::try_from(r.bits(32)?).map_err(|_| err())?;
    if r.bit()? {
        return Err(Error::Unsupported("randomised bzip2 blocks".into()));
    }
    let orig_ptr = usize::try_from(r.bits(24)?).map_err(|_| err())?;

    // Symbols in use.
    let used16 = r.bits(16)?;
    let mut seq: Vec<u8> = Vec::new();
    for i in 0..16u8 {
        if used16 & (0x8000 >> i) != 0 {
            let used = r.bits(16)?;
            for j in 0..16u8 {
                if used & (0x8000 >> j) != 0 {
                    seq.push(i.wrapping_mul(16) | j);
                }
            }
        }
    }
    if seq.is_empty() {
        return Err(err());
    }
    let alpha = seq.len().checked_add(2).ok_or_else(err)?;

    // Huffman groups and their selectors.
    let groups = usize::try_from(r.bits(3)?).map_err(|_| err())?;
    if !(2..=6).contains(&groups) {
        return Err(err());
    }
    let n_sel = usize::try_from(r.bits(15)?).map_err(|_| err())?;
    if n_sel == 0 {
        return Err(err());
    }
    let mut order: Vec<u8> = (0..6).collect();
    let mut selectors = Vec::with_capacity(n_sel);
    for _ in 0..n_sel {
        let mut j = 0usize;
        while r.bit()? {
            j = j.checked_add(1).ok_or_else(err)?;
            if j >= groups {
                return Err(err());
            }
        }
        let g = order.remove(j);
        order.insert(0, g);
        selectors.push(g);
    }
    let mut tables = Vec::with_capacity(groups);
    for _ in 0..groups {
        let mut len = i64::try_from(r.bits(5)?).map_err(|_| err())?;
        let mut lens = Vec::with_capacity(alpha);
        for _ in 0..alpha {
            loop {
                if !(1..=20).contains(&len) {
                    return Err(err());
                }
                if !r.bit()? {
                    break;
                }
                len = len
                    .checked_add(if r.bit()? { -1 } else { 1 })
                    .ok_or_else(err)?;
            }
            lens.push(u8::try_from(len).map_err(|_| err())?);
        }
        tables.push(Table::new(&lens)?);
    }

    // Symbols → MTF/RLE2 → block bytes.
    let eob = u16::try_from(alpha.checked_sub(1).ok_or_else(err)?).map_err(|_| err())?;
    let mut mtf: Vec<u8> = (0..=255).collect();
    let mut tt: Vec<u8> = Vec::new();
    let (mut run, mut weight) = (0usize, 1usize);
    let mut sel = selectors.iter();
    let mut table = None;
    let mut left = 0u32;
    loop {
        if left == 0 {
            table = tables.get(usize::from(*sel.next().ok_or_else(err)?));
            left = GROUP_SIZE;
        }
        left = left.checked_sub(1).ok_or_else(err)?;
        let sym = table.ok_or_else(err)?.decode(r)?;
        if sym <= 1 {
            // RUNA adds `weight`, RUNB twice that; weights double.
            run = weight
                .checked_mul(usize::from(sym) + 1)
                .and_then(|w| run.checked_add(w))
                .ok_or_else(err)?;
            weight = weight.checked_mul(2).ok_or_else(err)?;
            if run > max_block {
                return Err(err());
            }
            continue;
        }
        if run > 0 {
            let b = *seq
                .get(usize::from(*mtf.first().ok_or_else(err)?))
                .ok_or_else(err)?;
            if tt.len().checked_add(run).is_none_or(|t| t > max_block) {
                return Err(err());
            }
            tt.extend(std::iter::repeat_n(b, run));
            run = 0;
            weight = 1;
        }
        if sym == eob {
            break;
        }
        let i = usize::from(sym.checked_sub(1).ok_or_else(err)?);
        if i >= seq.len() {
            return Err(err());
        }
        let v = mtf.remove(i);
        mtf.insert(0, v);
        tt.push(*seq.get(usize::from(v)).ok_or_else(err)?);
        if tt.len() > max_block {
            return Err(err());
        }
    }

    // Inverse BWT.
    let n = tt.len();
    if orig_ptr >= n {
        return Err(err());
    }
    let mut start = [0usize; 256];
    for b in &tt {
        if let Some(s) = start.get_mut(usize::from(*b)) {
            *s = s.checked_add(1).ok_or_else(err)?;
        }
    }
    let mut sum = 0usize;
    for s in &mut start {
        let c = *s;
        *s = sum;
        sum = sum.checked_add(c).ok_or_else(err)?;
    }
    let mut next = vec![0usize; n];
    for (i, b) in tt.iter().enumerate() {
        let s = start.get_mut(usize::from(*b)).ok_or_else(err)?;
        *next.get_mut(*s).ok_or_else(err)? = i;
        *s = s.checked_add(1).ok_or_else(err)?;
    }

    // RLE1: four equal bytes are followed by a count of further copies.
    let mut p = *next.get(orig_ptr).ok_or_else(err)?;
    let mut block_crc = 0xFFFF_FFFFu32;
    let (mut last, mut same) = (None::<u8>, 0u8);
    for _ in 0..n {
        let b = *tt.get(p).ok_or_else(err)?;
        p = *next.get(p).ok_or_else(err)?;
        if same == 4 {
            for _ in 0..b {
                out.push(last.ok_or_else(err)?);
                block_crc = crc_update(block_crc, last.ok_or_else(err)?);
            }
            same = 0;
            last = None;
        } else {
            same = if Some(b) == last {
                same.saturating_add(1)
            } else {
                1
            };
            last = Some(b);
            out.push(b);
            block_crc = crc_update(block_crc, b);
        }
        if out.len() > limit {
            return Err(bad_size());
        }
    }
    if !block_crc != crc {
        return Err(crate::bad("bzip2: block CRC mismatch (corrupt chunk)"));
    }
    Ok(())
}

fn bad_size() -> Error {
    crate::bad("bzip2: chunk larger than expected")
}

/// Decompress one or more concatenated bzip2 streams into at most `limit` bytes.
pub(crate) fn decompress(input: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    let mut r = BitReader::new(input, CODEC);
    let mut out = Vec::with_capacity(limit);
    while !r.at_end() {
        if r.bits(24)? != 0x42_5A_68 {
            // "BZh"
            return Err(err());
        }
        let level = r.bits(8)?;
        if !(u64::from(b'1')..=u64::from(b'9')).contains(&level) {
            return Err(err());
        }
        let max_block = usize::try_from(level.saturating_sub(u64::from(b'0')))
            .map_err(|_| err())?
            .saturating_mul(100_000);
        loop {
            match r.bits(48)? {
                BLOCK_MAGIC => block(&mut r, max_block, &mut out, limit)?,
                END_MAGIC => {
                    r.bits(32)?; // the stream CRC (every block's was checked)
                    r.align();
                    break;
                }
                _ => return Err(err()),
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_garbage_without_panicking() {
        for input in [
            &b""[..],
            b"BZh",
            b"BZh9",
            b"BZh91AY&SY",
            b"XYZ123",
            &[0xFF; 64],
        ] {
            let _ = decompress(input, 1000);
        }
        // `bzip2 -c <<< hello`.
        let hello = [
            0x42, 0x5A, 0x68, 0x39, 0x31, 0x41, 0x59, 0x26, 0x53, 0x59, 0xC1, 0xC0, 0x80, 0xE2,
            0x00, 0x00, 0x01, 0x41, 0x00, 0x00, 0x10, 0x02, 0x44, 0xA0, 0x00, 0x30, 0xCD, 0x00,
            0xC3, 0x46, 0x29, 0x97, 0x17, 0x72, 0x45, 0x38, 0x50, 0x90, 0xC1, 0xC0, 0x80, 0xE2,
        ];
        assert_eq!(decompress(&hello, 100).unwrap(), b"hello\n");
        assert!(decompress(&hello, 3).is_err());
    }
}
