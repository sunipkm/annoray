//! netCDF-3 through the default (pure-Rust) path, with no C libraries.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use xarray_rs::io::{Format, ToNetcdf, WriteOptions, open_dataset};
use xarray_rs::{AttrValue, DType};

#[test]
fn opens_and_round_trips_netcdf3_without_c() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc3.nc");
    let ds = open_dataset(&p).unwrap();
    assert_eq!(ds.data_vars()["temp"].dtype(), DType::F32);
    assert_eq!(
        ds.data_vars()["station"]
            .data::<String>()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        ["a", "bb", "ccc"]
    );
    assert_eq!(
        ds.attrs().get("title"),
        Some(&AttrValue::from("golden file"))
    );

    let dir = std::env::temp_dir().join(format!("xarray-rs-pure-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (i, f) in [
        Format::Netcdf3Classic,
        Format::Netcdf3Offset64,
        Format::Netcdf3Data64,
    ]
    .into_iter()
    .enumerate()
    {
        let out = dir.join(format!("rt{i}.nc"));
        ds.to_netcdf_with(&out, &WriteOptions::format(f)).unwrap();
        let back = open_dataset(&out).unwrap();
        assert_eq!(
            back.data_vars().keys().collect::<Vec<_>>(),
            ds.data_vars().keys().collect::<Vec<_>>()
        );
        assert_eq!(
            back.coords()["time"]
                .data::<chrono::NaiveDateTime>()
                .unwrap(),
            ds.coords()["time"].data::<chrono::NaiveDateTime>().unwrap()
        );
    }
}

#[test]
#[cfg(not(feature = "netcdf-c"))]
fn netcdf4_without_c_is_a_clear_error() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let err = open_dataset(&p).unwrap_err().to_string();
    assert!(err.contains("netcdf-c"), "{err}");
}
