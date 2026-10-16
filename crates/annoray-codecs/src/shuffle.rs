//! Byte shuffle (HDF5 filter 2, numcodecs' `Shuffle`, Blosc's byte shuffle): bytes grouped by
//! their position within each element, which helps compressors on numeric data. A trailing
//! partial element is kept as it is.

/// Bytes grouped by position within each element → element-interleaved. Writes are
/// sequential (one element at a time, gathered from the byte planes).
pub fn unshuffle(data: &[u8], size: usize) -> Vec<u8> {
    let Some(n) = data.len().checked_div(size).filter(|_| size > 1) else {
        return data.to_vec();
    };
    let body = n.saturating_mul(size);
    let (planes_data, tail) = data.split_at(body.min(data.len()));
    let planes: Vec<&[u8]> = planes_data.chunks_exact(n.max(1)).collect();
    let mut out = Vec::with_capacity(data.len());
    match size {
        2 => unshuffle_n::<2>(&planes, n, &mut out),
        4 => unshuffle_n::<4>(&planes, n, &mut out),
        8 => unshuffle_n::<8>(&planes, n, &mut out),
        _ => {
            for e in 0..n {
                out.extend(planes.iter().map(|p| p.get(e).copied().unwrap_or(0)));
            }
        }
    }
    // A trailing partial element (if any) is kept as it is.
    out.extend_from_slice(tail);
    out
}

fn unshuffle_n<const S: usize>(planes: &[&[u8]], n: usize, out: &mut Vec<u8>) {
    let p: [&[u8]; S] = std::array::from_fn(|b| planes.get(b).copied().unwrap_or(&[]));
    for e in 0..n {
        let elem: [u8; S] =
            std::array::from_fn(|b| p.get(b).and_then(|x| x.get(e)).copied().unwrap_or(0));
        out.extend_from_slice(&elem);
    }
}

/// Element-interleaved → bytes grouped by position within each element. Writes are
/// sequential (one byte plane at a time).
pub fn shuffle(data: &[u8], size: usize) -> Vec<u8> {
    let Some(n) = data.len().checked_div(size).filter(|_| size > 1) else {
        return data.to_vec();
    };
    let body = n.saturating_mul(size);
    let (elems, tail) = data.split_at(body.min(data.len()));
    let mut out = Vec::with_capacity(data.len());
    for b in 0..size {
        out.extend(
            elems
                .chunks_exact(size)
                .map(|el| el.get(b).copied().unwrap_or(0)),
        );
    }
    out.extend_from_slice(tail);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_round_trips() {
        let data: Vec<u8> = (0..24).collect();
        for size in [1, 2, 4, 8, 3] {
            assert_eq!(unshuffle(&shuffle(&data, size), size), data, "size {size}");
        }
        assert_eq!(shuffle(&[1, 2, 3, 4], 2), vec![1, 3, 2, 4]);
        // A trailing partial element stays in place.
        let odd: Vec<u8> = (0..27).collect();
        for size in [2, 4, 8, 5] {
            let s = shuffle(&odd, size);
            assert_eq!(s.len(), odd.len());
            assert_eq!(unshuffle(&s, size), odd, "size {size}");
        }
        assert_eq!(shuffle(&[1, 2, 3, 4, 9], 2), vec![1, 3, 2, 4, 9]);
    }
}
