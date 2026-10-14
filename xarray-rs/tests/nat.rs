//! Missing times (NaT): reading xarray's encodings (`fixtures/nc/nat.nc`, from
//! `fixtures/gen_nat.py`), computing with them, and writing them back.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::path::PathBuf;

use chrono::{NaiveDateTime, TimeDelta};
use xarray_rs::io::{Format, OpenOptions, ToNetcdf, WriteOptions, open_dataset_with};
use xarray_rs::{Calendar, CfDate, DataArray, Dataset, Join, NaT, align};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/nat.nc")
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xarray-rs-nat-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn dt(s: &str) -> NaiveDateTime {
    s.parse().unwrap()
}

fn dates(ds: &Dataset, name: &str) -> Vec<NaiveDateTime> {
    ds.data_vars()[name]
        .data::<NaiveDateTime>()
        .unwrap()
        .iter()
        .copied()
        .collect()
}

fn check(ds: &Dataset) {
    check_without_ns(ds);
    let nat = NaiveDateTime::NAT;
    // Exact to the nanosecond (no rounding through f64), and the _FillValue is NaT.
    assert_eq!(
        dates(ds, "t_ns"),
        [
            dt("2023-11-14T22:13:20.123456789"),
            nat,
            dt("1970-01-01T00:00:00.000000001")
        ]
    );
}

fn check_without_ns(ds: &Dataset) {
    let nat = NaiveDateTime::NAT;
    let day = |d: &str| dt(&format!("{d}T00:00:00"));
    assert_eq!(dates(ds, "t"), [day("2020-01-01"), nat, day("2020-01-03")]);
    assert_eq!(
        dates(ds, "t_float"),
        [day("2020-01-01"), nat, day("2020-01-03")]
    );
    let noleap: Vec<CfDate> = ds.data_vars()["t_noleap"]
        .data::<CfDate>()
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(
        noleap,
        [
            CfDate::from_ymd(Calendar::NoLeap, 2000, 3, 1).unwrap(),
            CfDate::NAT,
            CfDate::from_ymd(Calendar::NoLeap, 2001, 1, 1).unwrap(),
        ]
    );
    let d: Vec<TimeDelta> = ds.data_vars()["d"]
        .data::<TimeDelta>()
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(
        d,
        [
            TimeDelta::minutes(90),
            TimeDelta::NAT,
            TimeDelta::minutes(-5)
        ]
    );
}

#[test]
fn reads_xarray_encodings() {
    for lazy in [true, false] {
        let ds = open_dataset_with(fixture(), &OpenOptions::default().lazy(lazy)).unwrap();
        assert!(ds.warnings.is_empty(), "{:?}", ds.warnings);
        check(&ds.dataset);
    }
}

#[test]
fn computes_with_nat() {
    let ds = open_dataset_with(fixture(), &OpenOptions::default())
        .unwrap()
        .dataset
        .loaded()
        .unwrap();
    let t = ds.get("t").unwrap().to_owned();
    assert!(format!("{t}").contains("NaT"));

    // Reductions skip NaT.
    let mean = t.mean(..).unwrap();
    assert_eq!(
        mean.data::<NaiveDateTime>().unwrap().first(),
        Some(&dt("2020-01-02T00:00:00"))
    );
    let min = t.min(..).unwrap();
    assert_eq!(
        min.data::<NaiveDateTime>().unwrap().first(),
        Some(&dt("2020-01-01T00:00:00"))
    );
    assert_eq!(
        t.count(..).unwrap().data::<i64>().unwrap().first(),
        Some(&2)
    );

    // Arithmetic propagates NaT.
    let first = t.isel([("x", 0)]).unwrap().to_owned();
    let diff = (&t - &first).unwrap();
    let diff: Vec<TimeDelta> = diff.data::<TimeDelta>().unwrap().iter().copied().collect();
    assert_eq!(
        diff,
        [TimeDelta::zero(), TimeDelta::NAT, TimeDelta::days(2)]
    );

    // Alignment fills missing labels with NaT.
    let other = DataArray::from_array(["x"], xarray_rs::ndarray::array![0.0, 0.0].into_dyn())
        .unwrap()
        .with_dim_coord("x", xarray_rs::ndarray::array![3i64, 4])
        .unwrap();
    let out = align(&[&t, &other], Join::Outer).unwrap();
    let filled: Vec<NaiveDateTime> = out[0]
        .data::<NaiveDateTime>()
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(filled.len(), 4);
    assert!(filled[3].is_nat());
}

#[test]
fn round_trips() {
    let ds = open_dataset_with(fixture(), &OpenOptions::default())
        .unwrap()
        .dataset;
    for (i, format) in [Format::Netcdf4, Format::Netcdf3Offset64]
        .into_iter()
        .enumerate()
    {
        let path = tmp(&format!("rt{i}.nc"));
        ds.to_netcdf_with(&path, &WriteOptions::format(format))
            .unwrap();
        let back = open_dataset_with(&path, &OpenOptions::default()).unwrap();
        assert!(back.warnings.is_empty(), "{:?}", back.warnings);
        let mut back = back.dataset;
        if format == Format::Netcdf3Offset64 {
            // No 64-bit integers: with NaT present, times are stored as float64 (as xarray
            // does), so nanoseconds are rounded. Compare those to the microsecond.
            let ns = dates(&back, "t_ns");
            let want = dates(&ds, "t_ns");
            for (a, b) in ns.iter().zip(&want) {
                assert!(a.is_nat() == b.is_nat(), "{ns:?}");
                assert!(a.is_nat() || (*a - *b).num_microseconds().unwrap().abs() <= 1);
            }
            back.remove("t_ns").unwrap();
            check_without_ns(&back);
            continue;
        }
        check(&back);
    }
}

#[test]
#[ignore = "needs python3 with xarray and netCDF4"]
fn python_xarray_reads_our_nat() {
    let ds = open_dataset_with(fixture(), &OpenOptions::default())
        .unwrap()
        .dataset;
    let path = tmp("py.nc");
    ds.to_netcdf_with(&path, &WriteOptions::format(Format::Netcdf4))
        .unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/check_nat_output.py");
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
