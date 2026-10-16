//! Zarr stores: xarray's (v2 and v3) read as their netCDF twins do; annoray's round-trip,
//! lazily and in chunks.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use std::path::PathBuf;

use annoray::io::zarr::{ZarrFormat, ZarrOptions};
use annoray::io::{ToZarr, open_dataset};
use annoray::{DType, Dataset, Variable, variables_equal};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/zarr")
}

/// Same dimensions, type, unit and values (NaN equal to NaN).
fn same_var(a: &Variable, b: &Variable, what: &str) {
    assert_eq!(a.dims(), b.dims(), "{what}");
    assert_eq!(a.dtype(), b.dtype(), "{what}");
    assert_eq!(a.unit(), b.unit(), "{what}");
    let numeric = !matches!(
        a.dtype(),
        DType::Str | DType::DateTime | DType::CfDate | DType::TimeDelta
    );
    if numeric {
        let (x, y) = (a.to_f64().unwrap(), b.to_f64().unwrap());
        for (p, q) in x.iter().zip(y.iter()) {
            assert!((p.is_nan() && q.is_nan()) || p == q, "{what}: {p} vs {q}");
        }
    } else {
        assert!(variables_equal(a, b), "{what}");
    }
}

fn same_ds(a: &Dataset, b: &Dataset, what: &str) {
    let names = |d: &Dataset| -> Vec<String> {
        let mut v: Vec<String> = d.data_vars().keys().map(ToString::to_string).collect();
        v.sort();
        v
    };
    assert_eq!(names(a), names(b), "{what}: data variables");
    for (k, v) in a.data_vars() {
        same_var(v, &b.data_vars()[k], &format!("{what}: {k}"));
        assert_eq!(
            v.attrs().get("long_name"),
            b.data_vars()[k].attrs().get("long_name")
        );
    }
    for (k, c) in a.coords() {
        same_var(c, &b.coords()[k], &format!("{what}: coordinate {k}"));
    }
    assert_eq!(a.attrs().get("title"), b.attrs().get("title"), "{what}");
}

#[test]
fn xarray_stores_read_like_their_netcdf_twins() {
    for v in [2, 3] {
        let mut z = open_dataset(fixtures().join(format!("xarray_v{v}.zarr"))).unwrap();
        assert!(z.data_vars()["temp"].is_lazy(), "v{v}: lazy");
        z.load().unwrap();
        let mut nc = open_dataset(fixtures().join(format!("xarray_v{v}.nc"))).unwrap();
        nc.load().unwrap();
        same_ds(&z, &nc, &format!("v{v}"));
        // CF decoding happened: times, units, masked integer, unpacked floats.
        assert_eq!(z.coords()["time"].dtype(), DType::DateTime);
        assert_eq!(z.data_vars()["count"].dtype(), DType::F32);
        assert!(z.data_vars()["count"].to_f64().unwrap()[1].is_nan());
        assert_eq!(z.data_vars()["temp"].unit().unwrap().to_string(), "degC");
    }
}

#[test]
fn stores_round_trip_in_both_formats() {
    let src = {
        let mut d = open_dataset(fixtures().join("xarray_v2.nc")).unwrap();
        d.load().unwrap();
        d
    };
    for format in [ZarrFormat::V2, ZarrFormat::V3] {
        let dir =
            std::env::temp_dir().join(format!("annoray-zarr-rt-{}-{format:?}", std::process::id()));
        src.to_zarr_with(&dir, &ZarrOptions::format(format))
            .unwrap();
        let mut back = open_dataset(&dir).unwrap();
        back.load().unwrap();
        same_ds(&back, &src, &format!("{format:?}"));
        // Writing over an existing store replaces it; a non-store directory is refused.
        src.to_zarr_with(&dir, &ZarrOptions::format(format))
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }
    let not_zarr = std::env::temp_dir().join(format!("annoray-not-zarr-{}", std::process::id()));
    std::fs::create_dir_all(&not_zarr).unwrap();
    std::fs::write(not_zarr.join("keep.txt"), "x").unwrap();
    assert!(src.to_zarr(&not_zarr).is_err());
    assert!(not_zarr.join("keep.txt").exists());
    std::fs::remove_dir_all(&not_zarr).ok();
}

#[test]
fn lazy_data_is_written_chunk_by_chunk() {
    let lazy = open_dataset(fixtures().join("xarray_v3.zarr")).unwrap();
    let dir = std::env::temp_dir().join(format!("annoray-zarr-lazy-{}", std::process::id()));
    lazy.to_zarr(&dir).unwrap();
    assert!(
        lazy.data_vars()["temp"].is_lazy(),
        "the source was not loaded"
    );
    let mut back = open_dataset(&dir).unwrap();
    back.load().unwrap();
    let mut want = lazy.clone();
    want.load().unwrap();
    same_ds(&back, &want, "lazy");
    std::fs::remove_dir_all(&dir).ok();
}
