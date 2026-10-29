//! Choosing a compression codec per variable survives writing and reading netCDF-4.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use annoray::io::ToNetcdf;
use annoray::ndarray::{ArrayD, IxDyn};
use annoray::{Compression, Dataset, Variable};

#[cfg(all(feature = "zstd", feature = "lz4"))]
#[test]
fn codecs_round_trip_through_netcdf4() {
    use annoray::io::open_dataset;
    use annoray::{BloscCodec, BloscShuffle};
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
    let dir = std::env::temp_dir().join(format!("annoray-comp-{}", std::process::id()));
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
    let path = std::env::temp_dir().join(format!("annoray-nozstd-{}.nc", std::process::id()));
    let err = ds.to_netcdf(&path).unwrap_err().to_string();
    assert!(err.contains("feature"), "{err}");
}

/// A compression default given when writing: every large variable without its own
/// compression gets it; the dataset is not changed.
#[test]
fn compression_default_when_writing() {
    use annoray::io::{Format, WriteOptions, open_dataset};
    use annoray::{Compress, DataTree};

    let big = |k: f64| {
        Variable::new(
            ["y", "x"],
            ArrayD::from_shape_fn(IxDyn(&[40, 30]), move |ix| (ix[0] * 30 + ix[1]) as f64 * k),
        )
        .unwrap()
    };
    let mut own = big(2.0);
    own.encoding_mut().deflate = Some(9);
    own.encoding_mut().chunks = Some(vec![10, 10]);
    let text = Variable::new(
        ["n"],
        ArrayD::from_shape_fn(IxDyn(&[300]), |ix| format!("s{}", ix[0])),
    )
    .unwrap();
    let ds = Dataset::new()
        .with_var("big", big(1.0))
        .unwrap()
        .with_var("own", own)
        .unwrap()
        .with_var("text", text)
        .unwrap()
        .with_var(
            "small",
            Variable::new(["x"], ArrayD::from_elem(IxDyn(&[30]), 1.5)).unwrap(),
        )
        .unwrap()
        .with_var(
            "scalar",
            Variable::new(Vec::<&str>::new(), ArrayD::from_elem(IxDyn(&[]), 7.0)).unwrap(),
        )
        .unwrap();
    let dir = std::env::temp_dir().join(format!("annoray-default-comp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let opts = WriteOptions::format(Format::Netcdf4).compress(Compress::deflate(4));
    let path = dir.join("d.nc");
    ds.to_netcdf_with(&path, &opts).unwrap();
    let back = open_dataset(&path).unwrap();
    let enc = |k: &str| back.data_vars()[k].encoding().clone();
    assert_eq!((enc("big").deflate, enc("big").shuffle), (Some(4), true));
    assert!(enc("big").chunks.is_some()); // netCDF-C's default chunk shape
    assert_eq!(enc("own").deflate, Some(9)); // its own encoding wins
    assert_eq!(enc("own").chunks, Some(vec![10, 10]));
    for k in ["text", "small", "scalar"] {
        assert_eq!(enc(k).deflate, None, "{k}");
    }
    for k in ["big", "own", "small"] {
        assert_eq!(
            back.data_vars()[k].data::<f64>().unwrap(),
            ds.data_vars()[k].data::<f64>().unwrap()
        );
    }
    // The dataset written is unchanged.
    assert_eq!(ds.data_vars()["big"].encoding().deflate, None);

    // Lazy values (rewriting a file without loading it), a lower threshold, and no shuffle.
    let opts = WriteOptions::format(Format::Netcdf4)
        .compress(Compress::deflate(1).shuffle(false).min_elements(30));
    let plain = dir.join("plain.nc");
    ds.to_netcdf(&plain).unwrap();
    let lazy = open_dataset(&plain).unwrap();
    assert!(lazy.data_vars()["big"].is_lazy());
    let again = dir.join("again.nc");
    lazy.to_netcdf_with(&again, &opts).unwrap();
    let back = open_dataset(&again).unwrap();
    let enc = |k: &str| back.data_vars()[k].encoding().clone();
    assert_eq!((enc("big").deflate, enc("big").shuffle), (Some(1), false));
    assert_eq!(enc("small").deflate, Some(1)); // 30 values: at the threshold

    // Trees: every group.
    let tree = DataTree::new(ds.clone())
        .with_child("g", DataTree::new(ds.clone()))
        .unwrap();
    let path = dir.join("tree.nc");
    tree.to_netcdf_with(
        &path,
        &WriteOptions::format(Format::Netcdf4).compress(Compress::deflate(2)),
    )
    .unwrap();
    let back = annoray::io::open_datatree(&path).unwrap();
    for t in [&back, &back.children()["g"]] {
        assert_eq!(t.dataset().data_vars()["big"].encoding().deflate, Some(2));
    }

    // netCDF-3 has no compression: the option is ignored.
    let path = dir.join("d3.nc");
    let mut classic = ds.clone();
    classic.remove("text").unwrap();
    let opts = WriteOptions::format(Format::Netcdf3Classic).compress(Compress::deflate(4));
    classic.to_netcdf_with(&path, &opts).unwrap();
    let back = open_dataset(&path).unwrap();
    assert_eq!(back.data_vars()["big"].encoding().deflate, None);
}
