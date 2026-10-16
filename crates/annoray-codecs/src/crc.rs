//! CRC-32 (IEEE, as gzip and zlib's `crc32`) and CRC-32C (Castagnoli, Zarr v3's `crc32c`
//! codec), table-driven.

// Evaluated at compile time: an out-of-range index or overflow is a build error, not a panic.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
const fn table(poly: u32) -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        #[allow(clippy::cast_possible_truncation)]
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ poly } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

const IEEE: [u32; 256] = table(0xEDB8_8320);
const CASTAGNOLI: [u32; 256] = table(0x82F6_3B78);

fn run(t: &[u32; 256], data: &[u8]) -> u32 {
    let mut c = !0u32;
    for b in data {
        let i = usize::from((c as u8) ^ b);
        c = t.get(i).copied().unwrap_or(0) ^ (c >> 8);
    }
    !c
}

/// CRC-32 (IEEE 802.3) of `data`.
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    run(&IEEE, data)
}

/// CRC-32C (Castagnoli) of `data`.
#[must_use]
pub fn crc32c(data: &[u8]) -> u32 {
    run(&CASTAGNOLI, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_values() {
        // The standard check value: the CRC of "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32(b""), 0);
    }
}
