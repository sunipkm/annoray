//! The pure-Rust HDF5 crate against netCDF-C (the HDF5 C library): netCDF-C must read the files
//! it writes, and it must read netCDF-C's files as netCDF-C does.
#![cfg(feature = "netcdf-c")]
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use std::path::PathBuf;

use annoray::AttrValue;
use annoray::hdf5::{Array, AttrData, WDataset, WGroup, WNode};
use annoray::io::netcdf_c::NetcdfC;
use annoray::io::netcdf4::Netcdf4;
use annoray::io::{Backend, RawDataset, WriteOptions};
use ndarray::{ArrayD, IxDyn, array};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures/nc")
        .join(name)
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("annoray-hdf5-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn sample() -> WGroup {
    let mut chunked = WDataset::new(Array::F64(ArrayD::from_shape_fn(IxDyn(&[7, 5]), |ix| {
        (ix[0] * 10 + ix[1]) as f64 / 3.0
    })));
    chunked.chunks = Some(vec![3, 2]);
    chunked.deflate = Some(5);
    chunked.shuffle = true;
    chunked
        .attrs
        .push(("units".into(), AttrData::Text("m s-1".into())));
    let mut unlimited = WDataset::new(Array::I32(array![1, 2, 3, 4, 5].into_dyn()));
    unlimited.unlimited = vec![true];
    unlimited.chunks = Some(vec![2]);
    let mut filled = WDataset::new(Array::I16(array![-1i16, 2, 3].into_dyn()));
    filled.fill = Some((-999i16).to_le_bytes().to_vec());
    let mut chars = WDataset::new(Array::U8(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), b"abxyz\0".to_vec()).unwrap(),
    ));
    chars.char_data = true;
    let mut referencing = WDataset::new(Array::F32(array![1.5f32].into_dyn()));
    referencing.attrs.push((
        "refs".into(),
        AttrData::References(vec!["/contig".into(), "/group/inner".into()]),
    ));
    referencing.attrs.push((
        "ref_lists".into(),
        AttrData::ReferenceLists(vec![vec!["/contig".into()]]),
    ));
    referencing.attrs.push((
        "pairs".into(),
        AttrData::ReferenceIndexPairs(vec![("/contig".into(), 0), ("/chunked".into(), 1)]),
    ));
    let mut many = WGroup::default();
    for i in 0..40 {
        many.members.push((
            format!("d{i:02}"),
            WNode::Dataset(WDataset::new(Array::I64(ArrayD::from_elem(IxDyn(&[]), i)))),
        ));
    }
    let mut inner = WGroup::default();
    inner.attrs.push((
        "level".into(),
        AttrData::Array(Array::I32(array![1].into_dyn())),
    ));
    inner.members.push((
        "inner".into(),
        WNode::Dataset(WDataset::new(Array::U64(array![u64::MAX, 7].into_dyn()))),
    ));
    WGroup {
        attrs: vec![
            ("title".into(), AttrData::Text("written by annoray".into())),
            (
                "ints".into(),
                AttrData::Array(Array::I32(array![1, -2, 3].into_dyn())),
            ),
            (
                "doubles".into(),
                AttrData::Array(Array::F64(array![0.5].into_dyn())),
            ),
            (
                "vlen".into(),
                AttrData::Array(Array::Str(
                    array!["α".to_owned(), String::new(), "βγ".to_owned()].into_dyn(),
                )),
            ),
            ("vstr".into(), AttrData::VarString("scalar vlen".into())),
            ("empty".into(), AttrData::Text("".into())),
        ],
        members: vec![
            (
                "contig".into(),
                WNode::Dataset(WDataset::new(Array::U8(array![1u8, 2, 255].into_dyn()))),
            ),
            ("chunked".into(), WNode::Dataset(chunked)),
            ("unlimited".into(), WNode::Dataset(unlimited)),
            ("filled".into(), WNode::Dataset(filled)),
            (
                "scalar".into(),
                WNode::Dataset(WDataset::new(Array::F32(ArrayD::from_elem(
                    IxDyn(&[]),
                    2.5,
                )))),
            ),
            (
                "empty_ds".into(),
                WNode::Dataset(WDataset::new(Array::F64(ArrayD::zeros(IxDyn(&[0, 4]))))),
            ),
            (
                "strings".into(),
                WNode::Dataset(WDataset::new(Array::Str(
                    array!["héllo".to_owned(), String::new(), "x".repeat(300)].into_dyn(),
                ))),
            ),
            ("chars".into(), WNode::Dataset(chars)),
            ("referencing".into(), WNode::Dataset(referencing)),
            ("group".into(), WNode::Group(inner)),
            ("many".into(), WNode::Group(many)),
        ],
    }
}

/// Write the sample to a file of its own (tests run in parallel).
fn written(name: &str) -> PathBuf {
    let path = tmp(name);
    std::fs::write(&path, annoray_hdf5::write_file(&sample()).unwrap()).unwrap();
    path
}

fn same_values(ours: &RawDataset, c: &RawDataset, ctx: &str) {
    for (name, v) in &c.vars {
        let Some(w) = ours.vars.get(name) else {
            panic!("{ctx}: {name} missing; skipped = {:?}", ours.skipped)
        };
        let a = format!("{:?}", v.data);
        let b = format!("{:?}", w.data);
        // NC_CHAR arrays are plain bytes to HDF5.
        if !v.is_char {
            assert_eq!(a, b, "{ctx}: {name}");
        }
        for (k, attr) in &v.attrs {
            if k.as_str() == "_FillValue" {
                continue; // netCDF-C reports the fill value property; HDF5 stores it as a property too
            }
            assert_eq!(
                Some(format!("{attr:?}")),
                w.attrs.get(k).map(|x| format!("{x:?}")),
                "{ctx}: {name}@{k}"
            );
        }
    }
    for (k, attr) in &c.attrs {
        assert_eq!(
            Some(format!("{attr:?}")),
            ours.attrs.get(k).map(|x| format!("{x:?}")),
            "{ctx}: global @{k}"
        );
    }
    for (g, child) in &c.groups {
        same_values(&ours.groups[g], child, &format!("{ctx}/{g}"));
    }
}

#[test]
fn netcdf_c_reads_our_files() {
    // netCDF-C opens plain HDF5 through the HDF5 C library.
    let raw = NetcdfC.read(&written("c1.h5")).unwrap();
    let ours = Netcdf4.read(&written("c2.h5")).unwrap();
    for name in [
        "contig",
        "chunked",
        "unlimited",
        "filled",
        "scalar",
        "strings",
    ] {
        assert_eq!(
            format!("{:?}", raw.vars[name].data),
            format!("{:?}", ours.vars[name].data),
            "{name}"
        );
    }
    assert_eq!(
        raw.attrs.get("title"),
        Some(&AttrValue::from("written by annoray"))
    );
    assert_eq!(
        raw.attrs.get("vlen"),
        Some(&AttrValue::Strs(vec!["α".into(), "".into(), "βγ".into()]))
    );
    assert_eq!(
        raw.groups["group"].attrs.get("level"),
        Some(&AttrValue::I32(vec![1]))
    );
}

#[test]
fn netcdf4_files_match_netcdf_c() {
    for name in ["xr_nc4.nc", "raw_nc4.nc"] {
        let c = NetcdfC.read(&fixture(name)).unwrap();
        let ours = Netcdf4.read(&fixture(name)).unwrap();
        same_values(&ours, &c, name);
    }
}

#[test]
fn round_trips_netcdf_c_datasets() {
    let c = NetcdfC.read(&fixture("raw_nc4.nc")).unwrap();
    let path = tmp("raw.nc");
    Netcdf4.write(&c, &path, &WriteOptions::default()).unwrap();
    let back = Netcdf4.read(&path).unwrap();
    same_values(&back, &c, "round trip");
}
