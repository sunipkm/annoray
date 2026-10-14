//! netCDF I/O through the C backend, checked against files written (and values decoded) by
//! Python xarray 2026.4.0 / netCDF4 1.7.4 (`fixtures/gen_netcdf.py`).
#![cfg(feature = "netcdf-c")]
#![allow(
    clippy::unwrap_used,
    clippy::arithmetic_side_effects,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::path::PathBuf;

use chrono::NaiveDateTime;
use xarray_rs::io::{Format, OpenOptions, ToNetcdf, WriteOptions, open_dataset, open_dataset_with};
use xarray_rs::units::time::Calendar;
use xarray_rs::{AttrValue, CfDate, DType, Dataset, TypedArray, Variable, VariableBase};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures/nc")
        .join(name)
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xarray-rs-io-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn f64s<S: xarray_rs::Repr>(v: &VariableBase<S>) -> Vec<f64> {
    match v.typed() {
        TypedArray::F32(a) => a.iter().map(|x| f64::from(*x)).collect(),
        TypedArray::F64(a) => a.iter().copied().collect(),
        other => panic!("not float: {other:?}"),
    }
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-9)
}

const TEMP: [f64; 24] = [
    270.0,
    271.0,
    272.0,
    273.0,
    274.0,
    275.0,
    276.0,
    277.0,
    278.0,
    f64::NAN,
    280.0,
    281.0,
    282.0,
    283.0,
    284.0,
    285.0,
    286.0,
    287.0,
    288.0,
    289.0,
    290.0,
    291.0,
    292.0,
    293.0,
];
const PRECIP: [f64; 24] = [
    f64::NAN,
    0.13,
    0.26,
    0.39,
    0.52,
    0.65,
    0.78,
    0.91,
    1.04,
    1.17,
    1.3,
    1.43,
    1.57,
    1.7,
    1.83,
    1.96,
    2.09,
    2.22,
    2.35,
    2.48,
    2.61,
    2.74,
    2.87,
    3.0,
];

fn date(s: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M").unwrap()
}

/// What xarray decodes from both golden files (flag only exists in the netCDF-4 one).
fn check_golden(ds: &Dataset, has_flag: bool) {
    let mut coords: Vec<&str> = ds.coords().keys().map(|k| k.as_str()).collect();
    coords.sort_unstable();
    assert_eq!(coords, ["height", "lat", "lon", "time"]);

    let temp = &ds.data_vars()["temp"];
    assert_eq!(temp.dtype(), DType::F32);
    assert_eq!(temp.dims(), ["time", "lat", "lon"]);
    assert_eq!(temp.unit().unwrap().label(), Some("K"));
    assert_eq!(
        temp.attrs().get("long_name"),
        Some(&AttrValue::from("air temperature"))
    );
    assert!(close(&f64s(temp), &TEMP));

    let precip = &ds.data_vars()["precip"];
    assert_eq!(precip.dtype(), DType::F64);
    assert!(close(
        &f64s(precip)
            .iter()
            .map(|v| (v * 100.0).round() / 100.0)
            .collect::<Vec<_>>(),
        &PRECIP
    ));
    assert_eq!(precip.unit().unwrap(), &"mm/day".parse().unwrap());
    assert_eq!(precip.encoding().dtype, Some(DType::I16));
    assert_eq!(
        precip.encoding().scale_factor,
        Some(AttrValue::F64(vec![0.01]))
    );

    let station = ds.data_vars()["station"].data::<String>().unwrap();
    assert_eq!(
        station.iter().cloned().collect::<Vec<_>>(),
        ["a", "bb", "ccc"]
    );
    if has_flag {
        assert_eq!(
            ds.data_vars()["flag"]
                .data::<u8>()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            [1, 200, 255]
        );
    }
    let land = ds.data_vars()["is_land"].data::<bool>().unwrap();
    assert_eq!(
        land.iter().copied().collect::<Vec<_>>(),
        [true, false, true]
    );

    let mt = ds.data_vars()["model_time"].data::<CfDate>().unwrap();
    let expected: Vec<CfDate> = [1, 15, 28]
        .iter()
        .map(|d| CfDate::from_ymd(Calendar::NoLeap, 2001, 2, *d).unwrap())
        .collect();
    assert_eq!(mt.iter().copied().collect::<Vec<_>>(), expected);

    let time = ds.coords()["time"].data::<NaiveDateTime>().unwrap();
    let expected: Vec<NaiveDateTime> = [
        "2000-01-01 00:00",
        "2000-01-01 06:00",
        "2000-01-01 12:00",
        "2000-01-01 18:00",
    ]
    .iter()
    .map(|s| date(s))
    .collect();
    assert_eq!(time.iter().copied().collect::<Vec<_>>(), expected);
    assert!(
        ds.coords()["time"]
            .encoding()
            .time_units
            .as_deref()
            .unwrap()
            .starts_with("hours since 2000-01-01")
    );

    let lat = &ds.coords()["lat"];
    assert_eq!(
        (lat.dtype(), f64s(lat)),
        (DType::F32, vec![-10.0, 0.0, 10.0])
    );
    assert_eq!(lat.unit().unwrap().label(), Some("degrees_north"));
    let height = &ds.coords()["height"];
    assert_eq!(
        (height.ndim(), f64s(height), height.unit().unwrap().label()),
        (0, vec![2.0], Some("m"))
    );
    assert_eq!(
        ds.attrs().get("title"),
        Some(&AttrValue::from("golden file"))
    );
}

#[test]
fn reads_xarray_netcdf4() {
    let opened = open_dataset_with(fixture("xr_nc4.nc"), &OpenOptions::default()).unwrap();
    assert!(opened.warnings.is_empty(), "{:?}", opened.warnings);
    let ds = opened.dataset;
    check_golden(&ds, true);
    assert_eq!(ds.encoding().unlimited_dims, ["time"]);
    assert_eq!(
        ds.data_vars()["temp"].encoding().chunks,
        Some(vec![1, 3, 2])
    );
}

#[test]
fn reads_xarray_netcdf3() {
    let ds = open_dataset(fixture("xr_nc3.nc")).unwrap();
    check_golden(&ds, false);
    assert!(ds.data_vars()["station"].encoding().char_dim.is_some());
}

#[test]
fn reads_raw_netcdf4_with_groups() {
    let ds = open_dataset(fixture("raw_nc4.nc")).unwrap();
    let u = ds.data_vars()["ubyte_as_signed"].data::<u8>().unwrap();
    assert_eq!(u.iter().copied().collect::<Vec<_>>(), [255, 0, 127]);
    let names = ds.data_vars()["names"].data::<String>().unwrap();
    assert_eq!(
        names.iter().cloned().collect::<Vec<_>>(),
        ["one", "three", "x"]
    );
    assert!(close(
        &f64s(&ds.data_vars()["with_missing"]),
        &[1.0, f64::NAN, 3.0]
    ));
    assert_eq!(ds.attrs().get("ints"), Some(&AttrValue::I32(vec![1, 2, 3])));

    let grp = open_dataset_with(fixture("raw_nc4.nc"), &OpenOptions::default().group("grp"))
        .unwrap()
        .dataset;
    let g = &grp.data_vars()["gvar"];
    assert_eq!(g.unit().unwrap().label(), Some("km"));
    assert_eq!(f64s(g), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let deeper = open_dataset_with(
        fixture("raw_nc4.nc"),
        &OpenOptions::default().group("grp/deeper"),
    )
    .unwrap()
    .dataset;
    assert_eq!(
        deeper.attrs().get("title"),
        Some(&AttrValue::from("nested"))
    );
    assert!(
        open_dataset_with(fixture("raw_nc4.nc"), &OpenOptions::default().group("nope")).is_err()
    );
}

/// Values, dtypes, dims, units and attributes survive a write and re-read.
fn assert_same(a: &Dataset, b: &Dataset) {
    let same_var = |name: &str, x: &Variable, y: &Variable| {
        assert_eq!(x.dims(), y.dims(), "{name}");
        assert_eq!(x.dtype(), y.dtype(), "{name}");
        assert_eq!(x.unit(), y.unit(), "{name}");
        assert_eq!(x.attrs(), y.attrs(), "{name}");
        if matches!(x.dtype(), DType::F32 | DType::F64) {
            assert!(close(&f64s(x), &f64s(y)), "{name}");
        } else {
            assert!(xarray_rs::variables_equal(x, y), "{name}: {x:?} vs {y:?}");
        }
    };
    assert_eq!(
        a.data_vars().keys().collect::<Vec<_>>(),
        b.data_vars().keys().collect::<Vec<_>>()
    );
    assert_eq!(
        a.coords().keys().collect::<Vec<_>>(),
        b.coords().keys().collect::<Vec<_>>()
    );
    for (k, v) in a.data_vars() {
        same_var(k, v, &b.data_vars()[k]);
    }
    for (k, v) in a.coords() {
        same_var(k, v, &b.coords()[k]);
    }
    assert_eq!(a.attrs(), b.attrs());
}

#[test]
fn round_trips_in_every_format() {
    let original = open_dataset(fixture("xr_nc4.nc")).unwrap();
    for (i, format) in [
        Format::Netcdf4,
        Format::Netcdf4Classic,
        Format::Netcdf3Classic,
        Format::Netcdf3Offset64,
        Format::Netcdf3Data64,
    ]
    .into_iter()
    .enumerate()
    {
        let path = tmp(&format!("rt{i}.nc"));
        original
            .to_netcdf_with(&path, &WriteOptions::format(format))
            .unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let back = open_dataset(&path).unwrap();
        assert_same(&original, &back);
    }
}

#[test]
fn writes_fresh_datasets_with_xarray_defaults() {
    use xarray_rs::ndarray::array;
    let t0 = date("2020-03-01 00:00");
    let mut ds = Dataset::new()
        .with_coord(
            "time",
            Variable::new(
                ["time"],
                array![t0, t0 + chrono::TimeDelta::days(1)].into_dyn(),
            )
            .unwrap(),
        )
        .unwrap()
        .with_var(
            "v",
            Variable::new(["time"], array![1.5, f64::NAN].into_dyn())
                .unwrap()
                .with_unit("m s-1".parse().unwrap()),
        )
        .unwrap()
        .with_var(
            "count",
            Variable::new(["time"], array![3i64, 4].into_dyn()).unwrap(),
        )
        .unwrap();
    ds.encoding_mut().unlimited_dims = vec!["time".into()];
    let path = tmp("fresh.nc");
    ds.to_netcdf(&path).unwrap();
    let raw = xarray_rs::io::default_backend().read(&path).unwrap();
    // Times: coarsest exact unit, first date as reference, as xarray does.
    assert_eq!(
        raw.vars["time"].attrs.get("units"),
        Some(&AttrValue::from("days since 2020-03-01 00:00:00"))
    );
    assert_eq!(
        raw.vars["time"].attrs.get("calendar"),
        Some(&AttrValue::from("proleptic_gregorian"))
    );
    assert_eq!(raw.vars["time"].data.dtype(), DType::I64);
    // Floats get a NaN fill value; units keep their spelling.
    assert!(
        matches!(raw.vars["v"].attrs.get("_FillValue"), Some(AttrValue::F64(v)) if v[0].is_nan())
    );
    assert_eq!(
        raw.vars["v"].attrs.get("units"),
        Some(&AttrValue::from("m s-1"))
    );
    assert!(raw.dims["time"].unlimited);
    let back = open_dataset(&path).unwrap();
    assert_same(&ds, &back);

    // int64 that doesn't fit in 32 bits cannot go to classic netCDF-3.
    let mut big = Dataset::new();
    big.add_var(
        "x",
        Variable::new(["n"], array![i64::MAX].into_dyn()).unwrap(),
    )
    .unwrap();
    assert!(
        big.to_netcdf_with(tmp("big.nc"), &WriteOptions::format(Format::Netcdf3Classic))
            .is_err()
    );
    assert!(
        big.to_netcdf_with(tmp("big5.nc"), &WriteOptions::format(Format::Netcdf3Data64))
            .is_ok()
    );
}

/// Python xarray must read what we write. Needs python3 with xarray and netCDF4:
/// `cargo test -p xarray-rs --features netcdf-c -- --ignored python`.
#[test]
#[ignore = "needs python3 with xarray and netCDF4"]
fn python_xarray_reads_our_files() {
    let original = open_dataset(fixture("xr_nc4.nc")).unwrap();
    for (i, format) in [Format::Netcdf4, Format::Netcdf3Offset64]
        .into_iter()
        .enumerate()
    {
        let path = tmp(&format!("py{i}.nc"));
        original
            .to_netcdf_with(&path, &WriteOptions::format(format))
            .unwrap();
        let script =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/check_rust_output.py");
        let out = std::process::Command::new("python3")
            .arg("-I")
            .arg(&script)
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "OK");
    }
}
