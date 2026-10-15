//! Writing with plugin codecs (Zstandard, LZ4, Blosc): our reader must read the values back
//! and report the codec, and the HDF5 C library's own plugins must read them too.
#![cfg(all(feature = "zstd", feature = "lz4"))]
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use std::path::PathBuf;

use annoray_core::{BloscCodec, BloscShuffle, Compression, TypedArray};
use annoray_hdf5::{File, Node, WDataset, WGroup, WNode};
use ndarray::{ArrayD, IxDyn};

const SHAPE: [usize; 2] = [23, 17];

fn value(i: usize, j: usize) -> f64 {
    (i * 100 + j) as f64
}

fn data(kind: &str) -> TypedArray<annoray_core::Owned> {
    let sh = IxDyn(&SHAPE);
    match kind {
        "f64" => TypedArray::F64(ArrayD::from_shape_fn(sh, |ix| value(ix[0], ix[1]) / 7.0)),
        "f32" => TypedArray::F32(ArrayD::from_shape_fn(sh, |ix| {
            (value(ix[0], ix[1]) / 7.0) as f32
        })),
        "i32" => TypedArray::I32(ArrayD::from_shape_fn(sh, |ix| value(ix[0], ix[1]) as i32)),
        "i16" => TypedArray::I16(ArrayD::from_shape_fn(sh, |ix| value(ix[0], ix[1]) as i16)),
        _ => panic!("{kind}"),
    }
}

fn cases() -> Vec<(&'static str, &'static str, Compression)> {
    let blosc = |codec, shuffle| Compression::Blosc {
        codec,
        level: 5,
        shuffle,
    };
    vec![
        ("zstd", "f64", Compression::Zstd { level: 3 }),
        ("lz4", "i32", Compression::Lz4),
        (
            "blosc_lz4_byte",
            "f32",
            blosc(BloscCodec::Lz4, BloscShuffle::Byte),
        ),
        (
            "blosc_zstd_bit",
            "f64",
            blosc(BloscCodec::Zstd, BloscShuffle::Bit),
        ),
        (
            "blosc_zlib_none",
            "i16",
            blosc(BloscCodec::Zlib, BloscShuffle::None),
        ),
    ]
}

fn written(file: &str) -> PathBuf {
    let mut root = WGroup::default();
    for (name, kind, c) in cases() {
        let mut d = WDataset::new(data(kind));
        // Ragged edge chunks included.
        d.chunks = Some(vec![10, 6]);
        d.compression = Some(c);
        root.members.push((name.into(), WNode::Dataset(d)));
    }
    let dir = std::env::temp_dir().join(format!("annoray-hdf5-codecs-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(file);
    std::fs::write(&path, annoray_hdf5::write_file(&root).unwrap()).unwrap();
    path
}

#[test]
fn our_reader_reads_our_codecs() {
    let f = File::open(written("ours.h5")).unwrap();
    let members = f.root().unwrap().members().unwrap();
    for (name, kind, c) in cases() {
        let Some((_, Node::Dataset(d))) = members.iter().find(|(n, _)| n == name) else {
            panic!("{name} missing");
        };
        assert_eq!(d.codec(), Some(c), "{name}");
        assert_eq!(
            format!("{:?}", d.read().unwrap()),
            format!("{:?}", data(kind)),
            "{name}"
        );
    }
}

#[test]
fn blosclz_cannot_be_written() {
    let mut d = WDataset::new(data("f64"));
    d.chunks = Some(vec![10, 6]);
    d.compression = Some(Compression::Blosc {
        codec: BloscCodec::BloscLz,
        level: 5,
        shuffle: BloscShuffle::Byte,
    });
    let root = WGroup {
        members: vec![("x".into(), WNode::Dataset(d))],
        ..WGroup::default()
    };
    assert!(annoray_hdf5::write_file(&root).is_err());
}

/// The HDF5 C library's plugins (via h5py and hdf5plugin) must read our files.
/// Set `XR_PYTHON` to a Python with both installed (default `python3`).
#[test]
#[ignore = "needs python3 with h5py and hdf5plugin"]
fn hdf5_plugins_read_our_codecs() {
    let python = std::env::var("XR_PYTHON").unwrap_or_else(|_| "python3".into());
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/check_codecs.py");
    let out = std::process::Command::new(python)
        .arg(&script)
        .arg(written("plugins.h5"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "OK");
}
