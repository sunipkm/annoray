//! The pure-Rust netCDF-3 backend against netCDF-C: each must read what the other writes, and
//! both must read the same values from the same file.
#![cfg(feature = "netcdf-c")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::arithmetic_side_effects
)]

use std::path::PathBuf;

use annoray::io::netcdf_c::NetcdfC;
use annoray::io::netcdf3::Netcdf3;
use annoray::io::{Backend, Format, RawDataset, RawDim, RawVariable, Storage, WriteOptions};
use annoray::{AttrValue, Name, Owned, TypedArray, Variable, variables_equal};
use ndarray::{ArrayD, IxDyn, array};
use proptest::prelude::*;

const FORMATS: [Format; 3] = [
    Format::Netcdf3Classic,
    Format::Netcdf3Offset64,
    Format::Netcdf3Data64,
];

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("annoray-netcdf3-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn var(dims: &[&str], data: TypedArray<Owned>) -> RawVariable {
    RawVariable {
        dims: dims.iter().map(|d| Name::from(*d)).collect(),
        data: data.into(),
        is_char: false,
        attrs: Default::default(),
        storage: Storage::default(),
    }
}

/// Exercises fixed and record variables (including odd sizes that need padding), scalars,
/// char data, and every attribute type the format allows.
fn sample(cdf5: bool, n_record_vars: usize) -> RawDataset {
    let mut ds = RawDataset::default();
    ds.dims.insert(
        "time".into(),
        RawDim {
            len: 3,
            unlimited: true,
        },
    );
    ds.dims.insert(
        "x".into(),
        RawDim {
            len: 3,
            unlimited: false,
        },
    );
    ds.dims.insert(
        "nchar".into(),
        RawDim {
            len: 5,
            unlimited: false,
        },
    );
    ds.attrs
        .insert("title".into(), AttrValue::from("differential"));
    ds.attrs
        .insert("shorts".into(), AttrValue::I16(vec![-1, 2, 3]));
    ds.attrs.insert("bytes".into(), AttrValue::I8(vec![-5]));
    ds.attrs
        .insert("pi".into(), AttrValue::F64(vec![std::f64::consts::PI]));
    ds.attrs.insert("e".into(), AttrValue::F32(vec![2.5, -0.0]));
    ds.attrs.insert("empty".into(), AttrValue::from(""));
    let mut a = var(
        &["x"],
        TypedArray::F64(array![1.5, -2.0, f64::NAN].into_dyn()),
    );
    // `_FillValue` first: the C backend defines it before other attributes.
    a.attrs
        .insert("_FillValue".into(), AttrValue::F64(vec![-999.0]));
    a.attrs.insert("units".into(), AttrValue::from("m"));
    ds.vars.insert("a".into(), a);
    ds.vars.insert(
        "b".into(),
        var(&["x"], TypedArray::I8(array![-128i8, 0, 127].into_dyn())),
    );
    ds.vars.insert(
        "s".into(),
        var(&[], TypedArray::I16(ArrayD::from_elem(IxDyn(&[]), 7))),
    );
    let mut c = var(
        &["x", "nchar"],
        TypedArray::U8(
            ArrayD::from_shape_vec(IxDyn(&[3, 5]), b"abc\0\0hello\0\0\0\0\0".to_vec()).unwrap(),
        ),
    );
    c.is_char = true;
    ds.vars.insert("c".into(), c);
    // Record variables: a 1-byte one (needs padding when there are several) and others.
    ds.vars.insert(
        "rb".into(),
        var(&["time"], TypedArray::I8(array![1i8, 2, 3].into_dyn())),
    );
    let mut filled = var(&["x"], TypedArray::I8(array![1i8, 2, 3].into_dyn()));
    filled
        .attrs
        .insert("_FillValue".into(), AttrValue::I8(vec![-1]));
    ds.vars.insert("filled".into(), filled);
    if n_record_vars > 1 {
        ds.vars.insert(
            "rf".into(),
            var(
                &["time", "x"],
                TypedArray::F32(
                    ArrayD::from_shape_vec(IxDyn(&[3, 3]), (0..9).map(|i| i as f32).collect())
                        .unwrap(),
                ),
            ),
        );
        ds.vars.insert(
            "rs".into(),
            var(&["time"], TypedArray::I16(array![-1i16, -2, -3].into_dyn())),
        );
    }
    if cdf5 {
        ds.vars.insert(
            "u64".into(),
            var(&["x"], TypedArray::U64(array![u64::MAX, 0, 1].into_dyn())),
        );
        ds.vars.insert(
            "i64".into(),
            var(&["x"], TypedArray::I64(array![i64::MIN, 0, 1].into_dyn())),
        );
        ds.vars.insert(
            "u8".into(),
            var(&["x"], TypedArray::U8(array![255u8, 0, 1].into_dyn())),
        );
        ds.vars.insert(
            "u16".into(),
            var(&["x"], TypedArray::U16(array![65535u16, 0, 1].into_dyn())),
        );
        ds.attrs
            .insert("big".into(), AttrValue::I64(vec![i64::MAX]));
    }
    ds
}

fn nan_eq(a: &TypedArray<Owned>, b: &TypedArray<Owned>) -> bool {
    match (a, b) {
        (TypedArray::F64(x), TypedArray::F64(y)) => {
            x.shape() == y.shape() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        (TypedArray::F32(x), TypedArray::F32(y)) => {
            x.shape() == y.shape() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        _ => variables_equal(
            &Variable::from_typed((0..a.ndim()).map(|i| format!("d{i}")), a.clone()).unwrap(),
            &Variable::from_typed((0..b.ndim()).map(|i| format!("d{i}")), b.clone()).unwrap(),
        ),
    }
}

/// Attributes compared order-independently, with NaN equal to NaN (via `Debug`).
fn attrs_key(a: &annoray::Attrs) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = a
        .iter()
        .map(|(k, x)| (k.to_string(), format!("{x:?}")))
        .collect();
    v.sort();
    v
}

fn assert_same(a: &RawDataset, b: &RawDataset, ctx: &str) {
    assert_eq!(a.dims, b.dims, "{ctx}: dims");
    assert_eq!(
        attrs_key(&a.attrs),
        attrs_key(&b.attrs),
        "{ctx}: global attrs"
    );
    assert_eq!(
        a.vars.keys().collect::<Vec<_>>(),
        b.vars.keys().collect::<Vec<_>>(),
        "{ctx}"
    );
    for (k, v) in &a.vars {
        let w = &b.vars[k];
        assert_eq!((&v.dims, v.is_char), (&w.dims, w.is_char), "{ctx}: {k}");
        assert_eq!(attrs_key(&v.attrs), attrs_key(&w.attrs), "{ctx}: {k} attrs");
        assert!(
            nan_eq(&v.data.values().unwrap(), &w.data.values().unwrap()),
            "{ctx}: {k}: {:?} vs {:?}",
            v.data,
            w.data
        );
    }
}

#[test]
fn ours_and_netcdf_c_read_each_others_files() {
    for (i, f) in FORMATS.into_iter().enumerate() {
        for n_rec in [1, 3] {
            let ds = sample(f == Format::Netcdf3Data64, n_rec);
            let opts = WriteOptions::format(f);
            // C writes, both read.
            let p = tmp(&format!("c{i}-{n_rec}.nc"));
            NetcdfC.write(&ds, &p, &opts).unwrap();
            let by_c = NetcdfC.read(&p).unwrap();
            assert_same(
                &by_c,
                &Netcdf3.read(&p).unwrap(),
                &format!("{f:?}/{n_rec}: C-written"),
            );
            // We write, both read.
            let q = tmp(&format!("r{i}-{n_rec}.nc"));
            Netcdf3.write(&ds, &q, &opts).unwrap();
            assert_same(
                &by_c,
                &NetcdfC.read(&q).unwrap(),
                &format!("{f:?}/{n_rec}: ours, read by C"),
            );
            assert_same(
                &by_c,
                &Netcdf3.read(&q).unwrap(),
                &format!("{f:?}/{n_rec}: ours, read by ours"),
            );
        }
    }
}

#[test]
fn reads_files_written_by_python_xarray() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc3.nc");
    assert_same(
        &NetcdfC.read(&p).unwrap(),
        &Netcdf3.read(&p).unwrap(),
        "xr_nc3.nc",
    );
}

#[test]
fn format_limits_are_errors() {
    let mut two_unlimited = sample(false, 1);
    two_unlimited.dims.insert(
        "t2".into(),
        RawDim {
            len: 1,
            unlimited: true,
        },
    );
    assert!(
        annoray::io::netcdf3::write_bytes(
            &two_unlimited,
            &WriteOptions::format(Format::Netcdf3Classic)
        )
        .is_err()
    );
    assert!(
        annoray::io::netcdf3::write_bytes(
            &sample(true, 1),
            &WriteOptions::format(Format::Netcdf3Classic)
        )
        .is_err()
    );
    assert!(
        annoray::io::netcdf3::write_bytes(
            &sample(false, 1),
            &WriteOptions::format(Format::Netcdf4)
        )
        .is_err()
    );
    let mut late_unlimited = sample(false, 1);
    late_unlimited.vars.insert(
        "bad".into(),
        var(
            &["x", "time"],
            TypedArray::I8(ArrayD::zeros(IxDyn(&[3, 3]))),
        ),
    );
    assert!(
        annoray::io::netcdf3::write_bytes(
            &late_unlimited,
            &WriteOptions::format(Format::Netcdf3Classic)
        )
        .is_err()
    );
    assert!(annoray::io::netcdf3::read_bytes(b"CDF\x03").is_err());
    assert!(annoray::io::netcdf3::read_bytes(b"").is_err());
}

fn valid_file() -> Vec<u8> {
    annoray::io::netcdf3::write_bytes(
        &sample(true, 3),
        &WriteOptions::format(Format::Netcdf3Data64),
    )
    .unwrap()
}

#[test]
fn every_truncation_is_an_error_not_a_panic() {
    for f in FORMATS {
        let bytes = annoray::io::netcdf3::write_bytes(
            &sample(f == Format::Netcdf3Data64, 3),
            &WriteOptions::format(f),
        )
        .unwrap();
        for n in 0..bytes.len() {
            let _ = annoray::io::netcdf3::read_bytes(&bytes[..n]);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Corrupting any bytes of a valid file never panics or over-allocates.
    #[test]
    fn corrupted_files_never_panic(edits in proptest::collection::vec((0usize..2000, any::<u8>()), 1..8)) {
        let mut bytes = valid_file();
        let len = bytes.len();
        for (pos, b) in edits {
            bytes[pos % len] = b;
        }
        let _ = annoray::io::netcdf3::read_bytes(&bytes);
    }

    /// Arbitrary bytes behind a valid magic number never panic.
    #[test]
    fn random_headers_never_panic(body in proptest::collection::vec(any::<u8>(), 0..512), v in prop_oneof![Just(1u8), Just(2), Just(5)]) {
        let mut bytes = vec![b'C', b'D', b'F', v];
        bytes.extend(body);
        let _ = annoray::io::netcdf3::read_bytes(&bytes);
    }
}

/// Our files are byte-for-byte what netCDF-C writes, padding included.
#[test]
fn byte_identical_to_netcdf_c() {
    for (i, f) in FORMATS.into_iter().enumerate() {
        for n_rec in [1, 3] {
            let ds = sample(f == Format::Netcdf3Data64, n_rec);
            let p = tmp(&format!("bc{i}-{n_rec}.nc"));
            NetcdfC.write(&ds, &p, &WriteOptions::format(f)).unwrap();
            let theirs = std::fs::read(&p).unwrap();
            let ours = annoray::io::netcdf3::write_bytes(&ds, &WriteOptions::format(f)).unwrap();
            let first_diff = theirs.iter().zip(&ours).position(|(a, b)| a != b);
            assert_eq!(
                (theirs.len(), first_diff),
                (ours.len(), None),
                "{f:?} with {n_rec} record variable(s)"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Slices read lazily from a memory-mapped file equal the same box cut from a full read.
    #[test]
    fn mapped_slices_match_full_reads(fmt in 0usize..3, n_rec in prop_oneof![Just(1usize), Just(3)], seed in any::<u64>()) {
        let f = FORMATS[fmt];
        let path = tmp(&format!("slice{fmt}-{n_rec}-{seed}.nc"));
        Netcdf3.write(&sample(f == Format::Netcdf3Data64, n_rec), &path, &WriteOptions::format(f)).unwrap();
        let file = annoray::io::netcdf3::Netcdf3File::open(&path).unwrap();
        let full = file.to_raw().unwrap();
        let mut rng = seed;
        let mut next = |n: usize| { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); if n == 0 { 0 } else { (rng >> 33) as usize % n } };
        let names: Vec<Name> = file.variables().cloned().collect();
        for name in names {
            let (_, shape, _) = file.variable(&name).unwrap();
            let shape = shape.to_vec();
            let start: Vec<usize> = shape.iter().map(|s| next(*s + 1)).collect();
            let count: Vec<usize> = shape.iter().zip(&start).map(|(s, st)| next(s - st + 1)).collect();
            let got = file.read_slice(&name, &start, &count).unwrap();
            let dims: Vec<String> = (0..shape.len()).map(|i| format!("d{i}")).collect();
            let whole = annoray::Variable::from_typed(dims.clone(), full.vars[&name].data.values().unwrap().into_owned()).unwrap();
            let idx: Vec<(String, annoray::Indexer)> = dims.iter().zip(start.iter().zip(&count))
                .map(|(d, (s, c))| (d.clone(), annoray::Indexer::Slice(annoray::PySlice::new(Some(*s as isize), Some((s + c) as isize), None))))
                .collect();
            let cut = whole.isel(idx).unwrap().to_owned();
            let key = |t: String| t.split(", strides=").next().unwrap_or_default().to_string();
            prop_assert_eq!(key(format!("{:?}", cut.typed().unwrap())), key(format!("{:?}", got)), "{:?} {:?} {:?}+{:?}", f, name, start, count);
        }
        let _ = std::fs::remove_file(&path);
    }
}
