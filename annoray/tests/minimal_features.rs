//! A build without the `netcdf4` and `zarr` features: netCDF-3 reads and writes as ever, and
//! the formats left out are errors that name the feature. Run with
//! `cargo test -p annoray --no-default-features --test minimal_features`.
#![cfg(not(any(feature = "netcdf4", feature = "zarr")))]
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::io::{Format, ToNetcdf, WriteOptions, open_dataset, open_dataset_from_bytes};
use annoray::ndarray::{ArrayD, IxDyn};
use annoray::{Dataset, Variable};

#[test]
fn netcdf3_alone() {
    let ds = Dataset::new()
        .with_var(
            "t",
            Variable::new(["x"], ArrayD::from_shape_fn(IxDyn(&[4]), |i| i[0] as f64))
                .unwrap()
                .with_unit("K".parse().unwrap()),
        )
        .unwrap();
    let dir = std::env::temp_dir().join(format!("annoray-minimal-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("a.nc");
    ds.to_netcdf_with(&path, &WriteOptions::format(Format::Netcdf3Classic))
        .unwrap();
    let back = open_dataset(&path).unwrap();
    assert_eq!(back.data_vars()["t"].data::<f64>().unwrap()[[3]], 3.0);
    assert_eq!(back.data_vars()["t"].unit().unwrap().to_string(), "K");

    // netCDF-4 is the default format: writing it names the missing feature.
    let e = ds.to_netcdf(dir.join("b.nc")).unwrap_err().to_string();
    assert!(e.contains("`netcdf4` feature"), "{e}");
    // So does an HDF5 file, from a path or from memory, and a Zarr store.
    let hdf5 = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let e = open_dataset(&hdf5).unwrap_err().to_string();
    assert!(e.contains("`netcdf4` feature"), "{e}");
    let bytes = std::fs::read(&hdf5).unwrap();
    let e = open_dataset_from_bytes(bytes, &Default::default())
        .unwrap_err()
        .to_string();
    assert!(e.contains("`netcdf4` feature"), "{e}");
    let store = dir.join("s.zarr");
    std::fs::create_dir_all(&store).unwrap();
    std::fs::write(store.join("zarr.json"), "{}").unwrap();
    let e = open_dataset(&store).unwrap_err().to_string();
    assert!(e.contains("`zarr` feature"), "{e}");
}
