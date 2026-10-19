//! Chunks of the Blosc2, ZFP, SZ and SZ3 plugins (`fixtures/h5/more_filters.h5`, from
//! `fixtures/gen_more_filters.py`): every dataset must read as its `_plain` copy (what HDF5
//! with the plugin reads back), whole and in slices.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::path::PathBuf;

use annoray_hdf5::{Array, Dataset, File, Node};

fn path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/h5/more_filters.h5")
}

fn datasets(f: &File) -> Vec<(String, Dataset)> {
    f.root()
        .unwrap()
        .members()
        .unwrap()
        .into_iter()
        .filter_map(|(n, node)| match node {
            Node::Dataset(d) => Some((n, *d)),
            _ => None,
        })
        .collect()
}

/// Bit patterns, so NaNs compare equal and floats compare exactly.
fn bits(t: &Array) -> Vec<u64> {
    match t {
        Array::F32(a) => a.iter().map(|v| u64::from(v.to_bits())).collect(),
        Array::F64(a) => a.iter().map(|v| v.to_bits()).collect(),
        Array::I8(a) => a.iter().map(|v| *v as u64).collect(),
        Array::I16(a) => a.iter().map(|v| *v as u64).collect(),
        Array::I32(a) => a.iter().map(|v| *v as u64).collect(),
        Array::I64(a) => a.iter().map(|v| *v as u64).collect(),
        Array::U8(a) => a.iter().map(|v| u64::from(*v)).collect(),
        Array::U16(a) => a.iter().map(|v| u64::from(*v)).collect(),
        Array::U32(a) => a.iter().map(|v| u64::from(*v)).collect(),
        Array::U64(a) => a.iter().copied().collect(),
        other => panic!("unexpected {:?}", other.dtype()),
    }
}

/// Equal to a relative 1e-9 (`double`) or 1e-5 (`float`).
fn close(got: &Array, want: &Array, name: &str) {
    let (g, w, tol): (Vec<f64>, Vec<f64>, f64) = match (got, want) {
        (Array::F32(a), Array::F32(b)) => (
            a.iter().map(|x| f64::from(*x)).collect(),
            b.iter().map(|x| f64::from(*x)).collect(),
            1e-5,
        ),
        (Array::F64(a), Array::F64(b)) => (
            a.iter().copied().collect(),
            b.iter().copied().collect(),
            1e-9,
        ),
        _ => panic!("{name}: types"),
    };
    assert_eq!(g.len(), w.len(), "{name}");
    for (x, y) in g.iter().zip(&w) {
        assert!(
            (x - y).abs() <= tol * y.abs().max(1e-30),
            "{name}: {x} vs {y}"
        );
    }
}

/// Which datasets need an optional feature.
fn available(name: &str) -> bool {
    let blosc2 = name.starts_with("blosc2");
    let zstd = name.contains("zstd")
        || name.starts_with("sz")
        || (blosc2
            && (name.ends_with("_3d") || name.ends_with("_i32") || name.ends_with("_zeros")));
    let lz4 = name.contains("lz4")
        || (blosc2 && (name.ends_with("_1d") || name.ends_with("_u8") || name.ends_with("_const")));
    (!zstd || cfg!(feature = "zstd")) && (!lz4 || cfg!(feature = "lz4"))
}

#[test]
fn every_plugin_matches_hdf5() {
    let f = File::open(path()).unwrap();
    let all = datasets(&f);
    let mut checked = Vec::new();
    for (name, d) in &all {
        if name.ends_with("_plain") || !available(name) {
            continue;
        }
        let plain = &all
            .iter()
            .find(|(n, _)| *n == format!("{name}_plain"))
            .unwrap()
            .1;
        let got = d.read().unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got.shape(), plain.shape().as_slice(), "{name}");
        let want = plain.read().unwrap();
        if name.contains("pwrel") && !cfg!(target_os = "macos") {
            // SZ's point-wise relative decoding tabulates `pow`, whose last bit depends on
            // the platform's libm (the fixtures come from macOS).
            close(&got, &want, name);
        } else {
            assert!(bits(&got) == bits(&want), "{name}");
        }
        // A slice over several chunks, including the ragged edge ones.
        let shape = d.shape();
        let start: Vec<usize> = shape.iter().map(|s| (s / 3).min(1)).collect();
        let count: Vec<usize> = shape.iter().zip(&start).map(|(s, a)| s - a).collect();
        let (gs, ws) = (
            d.read_slice(&start, &count).unwrap(),
            plain.read_slice(&start, &count).unwrap(),
        );
        if name.contains("pwrel") && !cfg!(target_os = "macos") {
            close(&gs, &ws, name);
        } else {
            assert!(bits(&gs) == bits(&ws), "{name} slice");
        }
        checked.push(name.clone());
    }
    assert!(checked.len() >= 10, "{checked:?}");
}

#[test]
fn corrupt_chunks_are_errors_not_panics() {
    let good = std::fs::read(path()).unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..120 {
        let mut bytes = good.clone();
        for _ in 0..8 {
            let at = (next() % bytes.len() as u64) as usize;
            bytes[at] = (next() & 0xFF) as u8;
        }
        let Ok(f) = File::from_bytes(bytes) else {
            continue;
        };
        let f = f.with_load_limit(1 << 24);
        let Ok(root) = f.root() else { continue };
        let Ok(members) = root.members() else {
            continue;
        };
        for (_, node) in members {
            if let Node::Dataset(d) = node {
                let _ = d.read();
            }
        }
    }
}
