//! Choosing a compression codec per variable survives writing and reading netCDF-4.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use xarray_rs::io::ToNetcdf;
use xarray_rs::ndarray::{ArrayD, IxDyn};
use xarray_rs::{Compression, Dataset, Variable};

#[cfg(all(feature = "zstd", feature = "lz4"))]
#[test]
fn codecs_round_trip_through_netcdf4() {
    use xarray_rs::io::open_dataset;
    use xarray_rs::{BloscCodec, BloscShuffle};
    let codecs = [
        Compression::Zstd { level: 5 },
        Compression::Lz4,
        Compression::Blosc {
            codec: BloscCodec::Zstd,
            level: 4,
            shuffle: BloscShuffle::Byte,
        },
    ];
    let mut ds = Dataset::new();
    for (i, c) in codecs.iter().enumerate() {
        let values =
            ArrayD::from_shape_fn(IxDyn(&[50, 40]), |ix| (ix[0] * 40 + ix[1]) as f64 / 3.0);
        let mut v = Variable::new(["y", "x"], values).unwrap();
        v.encoding_mut().chunks = Some(vec![20, 20]);
        v.encoding_mut().compression = Some(*c);
        ds.add_var(format!("v{i}"), v).unwrap();
    }
    let dir = std::env::temp_dir().join(format!("xarray-rs-comp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("c.nc");
    ds.to_netcdf(&path).unwrap();
    let back = open_dataset(&path).unwrap();
    for (i, c) in codecs.iter().enumerate() {
        let name = format!("v{i}");
        let v = &back.data_vars()[name.as_str()];
        assert_eq!(v.encoding().compression, Some(*c), "{name}");
        assert_eq!(
            v.data::<f64>().unwrap(),
            ds.data_vars()[name.as_str()].data::<f64>().unwrap()
        );
    }
}

#[cfg(not(feature = "zstd"))]
#[test]
fn zstd_needs_its_feature() {
    let mut v = Variable::new(["x"], ArrayD::from_elem(IxDyn(&[4]), 1.0)).unwrap();
    v.encoding_mut().chunks = Some(vec![2]);
    v.encoding_mut().compression = Some(Compression::Zstd { level: 1 });
    let ds = Dataset::new().with_var("v", v).unwrap();
    let path = std::env::temp_dir().join(format!("xarray-rs-nozstd-{}.nc", std::process::id()));
    let err = ds.to_netcdf(&path).unwrap_err().to_string();
    assert!(err.contains("feature"), "{err}");
}
