//! Stores written here read back here, and (`zarr_python_reads_them`, ignored by default)
//! with zarr-python via `fixtures/check_zarr.py`. Every array holds `f(i, j) = 100 i + j`
//! (divided by 7 for floats; `"s{i}_{j}"` for text; `(i + j) odd` for booleans).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::arithmetic_side_effects
)]

use std::path::{Path, PathBuf};

use annoray_zarr::{
    Array, ArrayBuilder, ArrayData, BloscCodec, BloscShuffle, Compressor, DType, Fill, Group, Map,
    Store, Value, ZarrFormat,
};
use ndarray::{Array2, ArrayD, IxDyn};

const SHAPE: [usize; 2] = [7, 5];
const CHUNKS: [usize; 2] = [3, 2];

fn values(dtype: DType) -> ArrayData {
    let f = |(i, j): (usize, usize)| (100 * i + j) as i64;
    let a = |g: &dyn Fn((usize, usize)) -> i64| Array2::from_shape_fn(SHAPE, g).into_dyn();
    match dtype {
        DType::Bool => {
            ArrayData::Bool(Array2::from_shape_fn(SHAPE, |(i, j)| (i + j) % 2 == 1).into_dyn())
        }
        DType::I8 => ArrayData::I8(a(&|p| f(p) % 100).mapv(|v| v as i8)),
        DType::I16 => ArrayData::I16(a(&f).mapv(|v| v as i16)),
        DType::I32 => ArrayData::I32(a(&f).mapv(|v| v as i32)),
        DType::I64 => ArrayData::I64(a(&f)),
        DType::U8 => ArrayData::U8(a(&|p| f(p) % 100).mapv(|v| v as u8)),
        DType::U16 => ArrayData::U16(a(&f).mapv(|v| v as u16)),
        DType::U32 => ArrayData::U32(a(&f).mapv(|v| v as u32)),
        DType::U64 => ArrayData::U64(a(&f).mapv(|v| v as u64)),
        DType::F32 => ArrayData::F32(a(&f).mapv(|v| v as f32 / 7.0)),
        DType::F64 => ArrayData::F64(a(&f).mapv(|v| v as f64 / 7.0)),
        _ => ArrayData::Str(Array2::from_shape_fn(SHAPE, |(i, j)| format!("s{i}_{j}")).into_dyn()),
    }
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("annoray-zarr-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    d
}

fn compressors() -> Vec<(&'static str, Compressor)> {
    let mut out = vec![
        ("none", Compressor::None),
        ("gzip", Compressor::Gzip(6)),
        ("zlib", Compressor::Zlib(6)),
        (
            "blosc_zlib",
            Compressor::Blosc {
                cname: BloscCodec::Zlib,
                clevel: 5,
                shuffle: BloscShuffle::Byte,
            },
        ),
    ];
    if cfg!(feature = "zstd") {
        out.push(("zstd", Compressor::Zstd(3)));
        out.push((
            "blosc_zstd_bit",
            Compressor::Blosc {
                cname: BloscCodec::Zstd,
                clevel: 5,
                shuffle: BloscShuffle::Bit,
            },
        ));
    }
    if cfg!(feature = "lz4") {
        out.push(("lz4", Compressor::Lz4));
        out.push((
            "blosc_lz4",
            Compressor::Blosc {
                cname: BloscCodec::Lz4,
                clevel: 5,
                shuffle: BloscShuffle::Byte,
            },
        ));
    }
    out
}

const DTYPES: [(&str, DType); 12] = [
    ("bool", DType::Bool),
    ("int8", DType::I8),
    ("int16", DType::I16),
    ("int32", DType::I32),
    ("int64", DType::I64),
    ("uint8", DType::U8),
    ("uint16", DType::U16),
    ("uint32", DType::U32),
    ("uint64", DType::U64),
    ("float32", DType::F32),
    ("float64", DType::F64),
    ("string", DType::String),
];

/// Write the test suite into `root` in `format`.
fn write_suite(root: &Path, format: ZarrFormat) {
    let store = Store::new(root);
    let mut attrs = Map::new();
    attrs.insert(
        "title".into(),
        Value::String("written by annoray-zarr".into()),
    );
    let group = Group::create(&store, "", format, attrs).unwrap();
    let names = vec!["y".to_owned(), "x".to_owned()];
    for (name, dt) in DTYPES {
        let a = ArrayBuilder::new(dt, &SHAPE, &CHUNKS)
            .dimension_names(names.clone())
            .create(&store, &format!("dtype_{name}"), format)
            .unwrap();
        a.write(&values(dt)).unwrap();
    }
    for (name, c) in compressors() {
        let a = ArrayBuilder::new(DType::F64, &SHAPE, &CHUNKS)
            .compressor(c)
            .create(&store, &format!("comp_{name}"), format)
            .unwrap();
        a.write(&values(DType::F64)).unwrap();
    }
    // Fixed-width text.
    let a = ArrayBuilder::new(DType::FixedUnicode(4), &SHAPE, &CHUNKS)
        .create(&store, "fixed_unicode", format)
        .unwrap();
    a.write(&values(DType::String)).unwrap();
    // A fill value: chunks equal to it are not stored.
    let a = ArrayBuilder::new(DType::F64, &[6], &[2])
        .fill_value(Fill::Float(f64::NAN))
        .create(&store, "sparse", format)
        .unwrap();
    let v = ArrayD::from_shape_vec(
        IxDyn(&[6]),
        vec![1.0, 2.0, f64::NAN, f64::NAN, 5.0, f64::NAN],
    )
    .unwrap();
    a.write(&ArrayData::F64(v)).unwrap();
    assert!(
        store.get(&a.chunk_key(&[1])).unwrap().is_none(),
        "an all-fill chunk is skipped"
    );
    // A scalar.
    let a = ArrayBuilder::new(DType::I32, &[], &[])
        .create(&store, "scalar", format)
        .unwrap();
    a.write(&ArrayData::I32(ArrayD::from_elem(IxDyn(&[]), 42)))
        .unwrap();
    if format == ZarrFormat::V3 {
        let a = ArrayBuilder::new(DType::I32, &[8, 9], &[4, 9])
            .shards(&[2, 3])
            .create(&store, "sharded", format)
            .unwrap();
        let v = Array2::from_shape_fn(
            (8, 9),
            |(i, j)| if i >= 4 { 0 } else { (100 * i + j) as i32 },
        );
        a.write(&ArrayData::I32(v.into_dyn())).unwrap();
    }
    Group::create(&store, "sub", format, Map::new()).unwrap();
    ArrayBuilder::new(DType::I16, &[3], &[3])
        .create(&store, "sub/inner", format)
        .unwrap()
        .write(&ArrayData::I16(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1, 2, 3]).unwrap(),
        ))
        .unwrap();
    group.consolidate().unwrap();
}

#[test]
fn round_trips_here() {
    for (i, format) in [ZarrFormat::V2, ZarrFormat::V3].into_iter().enumerate() {
        let root = dir(&format!("rt{i}"));
        write_suite(&root, format);
        let store = Store::new(&root);
        for (name, dt) in DTYPES {
            let a = Array::open(&store, &format!("dtype_{name}")).unwrap();
            assert_eq!(a.read().unwrap(), values(dt), "{format:?} {name}");
            assert_eq!(a.dimension_names().unwrap(), ["y", "x"]);
        }
        for (name, _) in compressors() {
            let a = Array::open(&store, &format!("comp_{name}")).unwrap();
            assert_eq!(a.read().unwrap(), values(DType::F64), "{format:?} {name}");
        }
        let fixed = Array::open(&store, "fixed_unicode").unwrap();
        assert_eq!(fixed.read().unwrap(), values(DType::String));
        let ArrayData::F64(s) = Array::open(&store, "sparse").unwrap().read().unwrap() else {
            panic!()
        };
        assert_eq!(s[0], 1.0);
        assert!(s[2].is_nan() && s[3].is_nan());
        let ArrayData::I32(sc) = Array::open(&store, "scalar").unwrap().read().unwrap() else {
            panic!()
        };
        assert_eq!(sc[[]], 42);
        if format == ZarrFormat::V3 {
            let sh = Array::open(&store, "sharded").unwrap();
            let ArrayData::I32(v) = sh.read_region(&[1, 2], &[4, 5]).unwrap() else {
                panic!()
            };
            assert_eq!(v[[0, 0]], 102);
            assert_eq!(v[[3, 4]], 0);
        }
        let g = Group::open(&store, "").unwrap();
        assert_eq!(g.attrs()["title"], "written by annoray-zarr");
        std::fs::remove_dir_all(&root).ok();
    }
}

#[test]
fn writes_reject_wrong_shapes_and_types() {
    let root = dir("bad");
    let store = Store::new(&root);
    let a = ArrayBuilder::new(DType::F64, &[4], &[2])
        .create(&store, "a", ZarrFormat::V3)
        .unwrap();
    assert!(
        a.write(&ArrayData::F64(ArrayD::zeros(IxDyn(&[3]))))
            .is_err()
    );
    assert!(
        a.write(&ArrayData::I32(ArrayD::zeros(IxDyn(&[4]))))
            .is_err()
    );
    assert!(
        a.write_chunk(&[1], &ArrayData::F64(ArrayD::zeros(IxDyn(&[1]))))
            .is_err()
    );
    assert!(
        ArrayBuilder::new(DType::F64, &[4], &[2])
            .shards(&[1])
            .create(&store, "b", ZarrFormat::V2)
            .is_err()
    );
    std::fs::remove_dir_all(&root).ok();
}

/// Set `XR_PYTHON` to a Python with zarr-python 3 (default `python3`).
#[test]
#[ignore = "needs python3 with zarr-python 3"]
fn zarr_python_reads_them() {
    let python = std::env::var("XR_PYTHON").unwrap_or_else(|_| "python3".into());
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/check_zarr.py");
    for (i, format) in [ZarrFormat::V2, ZarrFormat::V3].into_iter().enumerate() {
        let root = dir(&format!("py{i}"));
        write_suite(&root, format);
        let out = std::process::Command::new(&python)
            .arg(&script)
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{format:?}\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        std::fs::remove_dir_all(&root).ok();
    }
}
