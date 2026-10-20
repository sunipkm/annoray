//! Real-world dates outside the range of an `Instant` (1678–2262): CF dates where xarray gives
//! cftime objects, chosen for lazy data from the first and last stored values.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use annoray::io::{OpenOptions, ToNetcdf, open_dataset_with};
use annoray::units::Calendar;
use annoray::{CfDate, DType, Dataset, Variable};
use ndarray::Array1;

fn dates(years: &[i64]) -> Variable {
    let d: Vec<CfDate> = years
        .iter()
        .map(|y| CfDate::from_ymd(Calendar::ProlepticGregorian, *y, 1, 1).unwrap())
        .collect();
    Variable::new(["x"], Array1::from(d).into_dyn()).unwrap()
}

/// A file with date variable `t` (not a coordinate, so read lazily) holding `years`.
fn file(name: &str, years: &[i64]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("annoray-dates-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let mut ds = Dataset::new();
    ds.add_var("t", dates(years)).unwrap();
    ds.to_netcdf(&path).unwrap();
    path
}

fn open(path: &std::path::Path, lazy: bool) -> Dataset {
    open_dataset_with(path, &OpenOptions::default().lazy(lazy))
        .unwrap()
        .dataset
}

#[test]
fn lazy_dates_with_an_end_out_of_range_are_cf_dates() {
    for (name, years) in [("first.nc", [1500, 2000]), ("last.nc", [2000, 2500])] {
        let path = file(name, &years);
        let ds = open(&path, true);
        let t = &ds.data_vars()["t"];
        assert!(t.is_lazy());
        assert_eq!(t.dtype(), DType::CfDate, "{name}");
        let read = t.to_owned().into_typed().unwrap();
        assert_eq!(read.dtype(), DType::CfDate);
    }
}

#[test]
fn lazy_dates_in_range_at_the_ends_are_instants() {
    let path = file("ends.nc", &[2000, 2001]);
    let t = open(&path, true).data_vars()["t"].clone();
    assert_eq!(t.dtype(), DType::DateTime);
    assert_eq!(t.into_typed().unwrap().dtype(), DType::DateTime);
}

#[test]
fn a_middle_date_out_of_range_is_an_error_when_read() {
    // As in xarray: only the ends decide the type.
    let path = file("middle.nc", &[2000, 1500, 2001]);
    let t = open(&path, true).data_vars()["t"].clone();
    assert_eq!(t.dtype(), DType::DateTime);
    let err = t.into_typed().unwrap_err().to_string();
    assert!(err.contains("1678–2262"), "{err}");
    // Read eagerly, every value is seen: CF dates.
    let eager = open(&path, false);
    assert_eq!(eager.data_vars()["t"].dtype(), DType::CfDate);
}
