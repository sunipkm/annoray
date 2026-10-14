//! The pure-Rust netCDF-4 backend against netCDF-C, in both directions.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::arithmetic_side_effects
)]

use std::path::PathBuf;

use ndarray::{ArrayD, IxDyn, array};
use xr_core::{AttrValue, Name, Owned, TypedArray};
use xr_io::{Backend, Format, RawDataset, RawDim, RawVariable, Storage, WriteOptions};
use xr_io_c::NetcdfC;
use xr_netcdf4::Netcdf4;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/nc")
        .join(name)
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xr-netcdf4-{}", std::process::id()));
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

fn attrs_key(a: &xr_core::Attrs) -> Vec<(String, String)> {
    a.iter()
        .map(|(k, x)| (k.to_string(), format!("{x:?}")))
        .collect()
}

fn assert_same(c: &RawDataset, ours: &RawDataset, ctx: &str) {
    assert_eq!(c.dims, ours.dims, "{ctx}: dims");
    assert_eq!(
        attrs_key(&c.attrs),
        attrs_key(&ours.attrs),
        "{ctx}: global attrs"
    );
    assert_eq!(
        c.vars.keys().collect::<Vec<_>>(),
        ours.vars.keys().collect::<Vec<_>>(),
        "{ctx}: variables"
    );
    for (k, v) in &c.vars {
        let w = &ours.vars[k];
        assert_eq!((&v.dims, v.is_char), (&w.dims, w.is_char), "{ctx}: {k}");
        assert_eq!(attrs_key(&v.attrs), attrs_key(&w.attrs), "{ctx}: {k} attrs");
        assert_eq!(
            format!("{:?}", v.data),
            format!("{:?}", w.data),
            "{ctx}: {k} data"
        );
        assert_eq!(v.storage.chunks, w.storage.chunks, "{ctx}: {k} chunks");
    }
    assert_eq!(
        c.groups.keys().collect::<Vec<_>>(),
        ours.groups.keys().collect::<Vec<_>>(),
        "{ctx}: groups"
    );
    for (g, child) in &c.groups {
        assert_same(child, &ours.groups[g], &format!("{ctx}/{g}"));
    }
}

#[test]
fn reads_netcdf_c_files_like_netcdf_c() {
    for name in ["xr_nc4.nc", "raw_nc4.nc"] {
        let c = NetcdfC.read(&fixture(name)).unwrap();
        let ours = Netcdf4.read(&fixture(name)).unwrap();
        assert!(ours.skipped.is_empty(), "{name}: {:?}", ours.skipped);
        assert_same(&c, &ours, name);
    }
}

/// Everything netCDF-4 can hold that the CF layer produces.
fn sample() -> RawDataset {
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
            len: 4,
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
    ds.dims.insert(
        "unused".into(),
        RawDim {
            len: 7,
            unlimited: false,
        },
    );
    ds.attrs
        .insert("title".into(), AttrValue::from("netCDF-4 sample"));
    ds.attrs
        .insert("versions".into(), AttrValue::I32(vec![1, 2, 3]));
    ds.attrs.insert(
        "names".into(),
        AttrValue::Strs(vec!["a".into(), "bc".into()]),
    );
    let mut x = var(
        &["x"],
        TypedArray::F64(array![0.5, 1.5, 2.5, 3.5].into_dyn()),
    );
    x.attrs.insert("units".into(), AttrValue::from("m"));
    ds.vars.insert("x".into(), x);
    let mut t = var(&["time"], TypedArray::I64(array![0i64, 6, 12].into_dyn()));
    t.attrs
        .insert("units".into(), AttrValue::from("hours since 2000-01-01"));
    ds.vars.insert("time".into(), t);
    let mut temp = var(
        &["time", "x"],
        TypedArray::F32(ArrayD::from_shape_fn(IxDyn(&[3, 4]), |i| {
            (i[0] * 10 + i[1]) as f32
        })),
    );
    temp.attrs
        .insert("_FillValue".into(), AttrValue::F32(vec![-999.0]));
    temp.attrs
        .insert("long_name".into(), AttrValue::from("temperature"));
    temp.storage.chunks = Some(vec![1, 4]);
    temp.storage.deflate = Some(4);
    temp.storage.shuffle = true;
    ds.vars.insert("temp".into(), temp);
    ds.vars.insert(
        "counts".into(),
        var(
            &["x"],
            TypedArray::U16(array![1u16, 2, 65535, 0].into_dyn()),
        ),
    );
    ds.vars.insert(
        "big".into(),
        var(
            &["x"],
            TypedArray::U64(array![u64::MAX, 0, 1, 2].into_dyn()),
        ),
    );
    ds.vars.insert(
        "label".into(),
        var(
            &["x"],
            TypedArray::Str(
                array![
                    "α".to_owned(),
                    "".to_owned(),
                    "c".to_owned(),
                    "dd".to_owned()
                ]
                .into_dyn(),
            ),
        ),
    );
    let mut c = var(
        &["x", "nchar"],
        TypedArray::U8(
            ArrayD::from_shape_vec(IxDyn(&[4, 5]), b"abc\0\0hello\0\0\0\0\0z\0\0\0\0".to_vec())
                .unwrap(),
        ),
    );
    c.is_char = true;
    ds.vars.insert("chars".into(), c);
    ds.vars.insert(
        "scalar".into(),
        var(&[], TypedArray::F64(ArrayD::from_elem(IxDyn(&[]), 42.0))),
    );
    // A variable named like a dimension without being its coordinate variable.
    ds.dims.insert(
        "y".into(),
        RawDim {
            len: 2,
            unlimited: false,
        },
    );
    ds.vars.insert(
        "y".into(),
        var(&["x", "y"], TypedArray::I32(ArrayD::zeros(IxDyn(&[4, 2])))),
    );
    let mut g = RawDataset::default();
    g.dims.insert(
        "z".into(),
        RawDim {
            len: 2,
            unlimited: false,
        },
    );
    g.attrs.insert("level".into(), AttrValue::I8(vec![1]));
    g.vars.insert(
        "gv".into(),
        var(
            &["x", "z"],
            TypedArray::I16(ArrayD::from_shape_fn(IxDyn(&[4, 2]), |i| {
                (i[0] + i[1]) as i16
            })),
        ),
    );
    let mut deeper = RawDataset::default();
    deeper.vars.insert(
        "leaf".into(),
        var(&["z"], TypedArray::F32(array![1.0f32, 2.0].into_dyn())),
    );
    g.groups.insert("deeper".into(), deeper);
    ds.groups.insert("grp".into(), g);
    ds
}

#[test]
fn netcdf_c_reads_our_files_like_its_own() {
    let ds = sample();
    let theirs = tmp("c.nc");
    NetcdfC
        .write(&ds, &theirs, &WriteOptions::format(Format::Netcdf4))
        .unwrap();
    let reference = NetcdfC.read(&theirs).unwrap();
    // We read netCDF-C's file the way netCDF-C does…
    assert_same(
        &reference,
        &Netcdf4.read(&theirs).unwrap(),
        "ours reading C's file",
    );
    // …and netCDF-C (and we) read our file the way it reads its own.
    let ours = tmp("ours.nc");
    Netcdf4
        .write(&ds, &ours, &WriteOptions::format(Format::Netcdf4))
        .unwrap();
    assert_same(
        &reference,
        &NetcdfC.read(&ours).unwrap(),
        "C reading our file",
    );
    assert_same(
        &reference,
        &Netcdf4.read(&ours).unwrap(),
        "ours reading our file",
    );
}

#[test]
fn classic_model() {
    let mut ds = sample();
    ds.groups.clear();
    ds.attrs.shift_remove("names"); // string lists need the full model
    ds.vars
        .retain(|_, v| v.is_char || Format::Netcdf4Classic.supports(v.data.dtype()));
    let theirs = tmp("c-classic.nc");
    NetcdfC
        .write(&ds, &theirs, &WriteOptions::format(Format::Netcdf4Classic))
        .unwrap();
    let ours = tmp("ours-classic.nc");
    Netcdf4
        .write(&ds, &ours, &WriteOptions::format(Format::Netcdf4Classic))
        .unwrap();
    assert_same(
        &NetcdfC.read(&theirs).unwrap(),
        &NetcdfC.read(&ours).unwrap(),
        "classic",
    );
    assert!(
        Netcdf4
            .write(
                &sample(),
                &tmp("bad.nc"),
                &WriteOptions::format(Format::Netcdf4Classic)
            )
            .is_err()
    );
}
