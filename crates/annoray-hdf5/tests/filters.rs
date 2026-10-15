//! Compressed and packed chunks written by HDF5's own filters and plugins
//! (`fixtures/h5/filters.h5`, from `fixtures/gen_filters.py`): every dataset must read as its
//! uncompressed `_plain` copy, whole and in slices.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::path::PathBuf;

use annoray_core::{Owned, TypedArray};
use annoray_hdf5::{Dataset, File, Node};

fn file() -> File {
    File::open(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/h5/filters.h5"))
        .unwrap()
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

fn numbers(t: &TypedArray<Owned>) -> Vec<f64> {
    match t {
        TypedArray::F32(a) => a.iter().map(|v| f64::from(*v)).collect(),
        TypedArray::F64(a) => a.iter().copied().collect(),
        TypedArray::I16(a) => a.iter().map(|v| f64::from(*v)).collect(),
        TypedArray::I32(a) => a.iter().map(|v| f64::from(*v)).collect(),
        TypedArray::I64(a) => a.iter().map(|v| *v as f64).collect(),
        TypedArray::U8(a) => a.iter().map(|v| f64::from(*v)).collect(),
        TypedArray::U16(a) => a.iter().map(|v| f64::from(*v)).collect(),
        other => panic!("unexpected {:?}", other.dtype()),
    }
}

/// Which datasets need an optional feature.
fn available(name: &str) -> bool {
    let zstd = name.contains("zstd");
    let lz4 = name.contains("lz4") || name == "blosc_i16";
    (!zstd || cfg!(feature = "zstd")) && (!lz4 || cfg!(feature = "lz4"))
}

#[test]
fn every_filter_matches_hdf5() {
    let f = file();
    let all = datasets(&f);
    let mut checked = Vec::new();
    for (name, d) in &all {
        if name.ends_with("_plain") {
            continue;
        }
        let plain = &all
            .iter()
            .find(|(n, _)| *n == format!("{name}_plain"))
            .unwrap()
            .1;
        if !available(name) {
            // Without the codec only an error naming the feature is acceptable (a chunk may
            // also have been stored uncompressed, and then reads fine).
            match d.read() {
                Err(e) => {
                    assert!(e.to_string().contains("feature"), "{name}: {e}");
                    continue;
                }
                Ok(v) => assert_eq!(numbers(&v), numbers(&plain.read().unwrap()), "{name}"),
            }
            continue;
        }
        let got = d.read().unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got.shape(), plain.shape().as_slice(), "{name}");
        assert_eq!(numbers(&got), numbers(&plain.read().unwrap()), "{name}");
        // A slice over several chunks, including the ragged edge ones.
        let shape = d.shape();
        let (start, count) = ([1, 2], [shape[0] - 1, shape[1] - 3]);
        assert_eq!(
            numbers(&d.read_slice(&start, &count).unwrap()),
            numbers(&plain.read_slice(&start, &count).unwrap()),
            "{name} slice"
        );
        checked.push(name.clone());
    }
    assert!(checked.len() >= 10, "{checked:?}");
}

#[test]
fn corrupt_chunks_are_errors_not_panics() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/h5/filters.h5");
    let good = std::fs::read(path).unwrap();
    // A deterministic generator, so failures reproduce.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..300 {
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
                let _ = d.read_slice(&[1, 1], &[2, 2]);
            }
        }
    }
}

/// 1024x1024 chunks: several Blosc blocks and split streams, many Bitshuffle blocks,
/// multi-block bzip2 and LZ4 streams, ragged edges. The file is ~300 MB, so it is generated
/// on the spot: `XR_PYTHON=<python with h5py, hdf5plugin> cargo test -p annoray-hdf5
/// --all-features --release --test filters -- --ignored big_chunks --nocapture`.
#[test]
#[ignore = "needs python3 with h5py and hdf5plugin; writes ~300 MB"]
fn big_chunks() {
    let python = std::env::var("XR_PYTHON").unwrap_or_else(|_| "python3".into());
    let dir = std::env::temp_dir().join(format!("annoray-hdf5-big-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("big.h5");
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gen_filters.py");
    let out = std::process::Command::new(python)
        .arg(&script)
        .arg("--big")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let f = File::open(&path).unwrap().with_load_limit(1 << 30);
    let all = datasets(&f);
    let mut checked = 0;
    for (name, d) in &all {
        if name.ends_with("_plain") {
            continue;
        }
        let plain = &all
            .iter()
            .find(|(n, _)| *n == format!("{name}_plain"))
            .unwrap()
            .1;
        let t = std::time::Instant::now();
        let got = d.read().unwrap_or_else(|e| panic!("{name}: {e}"));
        let took = t.elapsed();
        assert_eq!(numbers(&got), numbers(&plain.read().unwrap()), "{name}");
        // A slice straddling the chunk boundary.
        assert_eq!(
            numbers(&d.read_slice(&[1000, 1000], &[100, 30]).unwrap()),
            numbers(&plain.read_slice(&[1000, 1000], &[100, 30]).unwrap()),
            "{name} slice"
        );
        println!("{name:>22}: {took:?}");
        checked += 1;
    }
    assert!(checked >= 20);
    let _ = std::fs::remove_dir_all(dir);
}
