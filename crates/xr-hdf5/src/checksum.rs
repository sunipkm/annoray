//! Bob Jenkins' lookup3 `hashlittle`, which HDF5 uses for metadata checksums
//! (`H5_checksum_lookup3`).

/// Up to four bytes as a little-endian word, zero-padded.
fn word(k: &[u8]) -> u32 {
    let mut b = [0u8; 4];
    for (d, s) in b.iter_mut().zip(k) {
        *d = *s;
    }
    u32::from_le_bytes(b)
}

/// `hashlittle(key, initval)`.
#[must_use]
pub fn lookup3(key: &[u8], initval: u32) -> u32 {
    let len = u32::try_from(key.len()).unwrap_or(u32::MAX);
    let mut a = 0xdead_beef_u32.wrapping_add(len).wrapping_add(initval);
    let mut b = a;
    let mut c = a;
    let mut k = key;
    while k.len() > 12 {
        a = a.wrapping_add(word(k));
        b = b.wrapping_add(word(k.get(4..).unwrap_or_default()));
        c = c.wrapping_add(word(k.get(8..).unwrap_or_default()));
        // mix(a, b, c)
        a = a.wrapping_sub(c);
        a ^= c.rotate_left(4);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a);
        b ^= a.rotate_left(6);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b);
        c ^= b.rotate_left(8);
        b = b.wrapping_add(a);
        a = a.wrapping_sub(c);
        a ^= c.rotate_left(16);
        c = c.wrapping_add(b);
        b = b.wrapping_sub(a);
        b ^= a.rotate_left(19);
        a = a.wrapping_add(c);
        c = c.wrapping_sub(b);
        c ^= b.rotate_left(4);
        b = b.wrapping_add(a);
        k = k.get(12..).unwrap_or_default();
    }
    if k.is_empty() {
        return c;
    }
    // Tail: up to 12 bytes, little-endian, zero-padded.
    a = a.wrapping_add(word(k));
    b = b.wrapping_add(word(k.get(4..).unwrap_or_default()));
    c = c.wrapping_add(word(k.get(8..).unwrap_or_default()));
    // final(a, b, c)
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(14));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(11));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(25));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(16));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(4));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(14));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(24));
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        // From lookup3.c's driver: hashlittle("", 0) and "Four score and seven years ago".
        assert_eq!(lookup3(b"", 0), 0xdead_beef);
        assert_eq!(lookup3(b"Four score and seven years ago", 0), 0x1777_0551);
        assert_eq!(lookup3(b"Four score and seven years ago", 1), 0xcd62_8161);
    }
}
