//! Read limits: a file's metadata is checked before any values are read, and every read
//! after that.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::arithmetic_side_effects
)]

use std::path::PathBuf;

use annoray::io::{
    Format, OpenOptions, ReadLimits, ToNetcdf, WriteOptions, open_dataset_from_bytes,
    open_dataset_with, open_datatree_with,
};
use annoray::ndarray::{ArrayD, IxDyn};
use annoray::{CoreError, DataTree, Dataset, Variable};

/// `t` (100 × 100 doubles, 80 kB, in 10 × 10 chunks of 800 bytes; five attributes and the
/// `_FillValue` written for floats), a 3-D `cube`, and a coordinate; netCDF-4, or netCDF-3
/// (unchunked).
fn file(format: Format) -> PathBuf {
    let mut t = Variable::new(
        ["y", "x"],
        ArrayD::from_shape_fn(IxDyn(&[100, 100]), |ix| (ix[0] * 100 + ix[1]) as f64),
    )
    .unwrap()
    .with_attrs((0..5).map(|i| (format!("a{i}"), "v")));
    t.encoding_mut().chunks = Some(vec![10, 10]);
    let cube = Variable::new(["p", "q", "r"], ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1i32)).unwrap();
    let ds = Dataset::new()
        .with_var("t", t)
        .unwrap()
        .with_var("cube", cube)
        .unwrap()
        .with_coord(
            "x",
            Variable::new(
                ["x"],
                ArrayD::from_shape_fn(IxDyn(&[100]), |ix| ix[0] as f64),
            )
            .unwrap(),
        )
        .unwrap();
    let dir = std::env::temp_dir().join(format!("annoray-limits-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{format:?}.nc"));
    ds.to_netcdf_with(&path, &WriteOptions::format(format))
        .unwrap();
    path
}

fn limit_of(e: &annoray::Error) -> (String, u64, u64) {
    match e.core() {
        Some(CoreError::Limit { what, found, limit }) => (what.clone(), *found, *limit),
        _ => panic!("not a limit: {e:?}"),
    }
}

fn open(path: &PathBuf, limits: ReadLimits, lazy: bool) -> Result<Dataset, annoray::Error> {
    let opts = OpenOptions::default().lazy(lazy).limits(limits);
    open_dataset_with(path, &opts).map(|o| o.dataset)
}

#[test]
fn metadata_limits_refuse_the_file_before_reading() {
    for format in [Format::Netcdf4, Format::Netcdf3Classic] {
        let path = file(format);
        for lazy in [true, false] {
            let cases = [
                (
                    ReadLimits::default().max_rank(2),
                    "dimensions of /cube",
                    3,
                    2,
                ),
                (
                    ReadLimits::default().max_elements(9_999),
                    "values of /t",
                    10_000,
                    9_999,
                ),
                (
                    ReadLimits::default().max_attributes(5),
                    "attributes of /t",
                    6,
                    5,
                ),
                (
                    ReadLimits::default().max_variables(2),
                    "variables in the file",
                    3,
                    2,
                ),
            ];
            for (limits, what, found, limit) in cases {
                let e = open(&path, limits, lazy).unwrap_err();
                assert_eq!(
                    limit_of(&e),
                    (what.to_owned(), found, limit),
                    "{format:?} {lazy}"
                );
            }
            // The total: 80 kB of `t`, 96 bytes of `cube`, 800 of `x`.
            let e = open(&path, ReadLimits::default().max_total_bytes(80_000), lazy).unwrap_err();
            let (what, found, _) = limit_of(&e);
            assert_eq!(what, "bytes of values in the file");
            assert!(found > 80_000);
            assert!(open(&path, ReadLimits::default().max_total_bytes(81_000), lazy).is_ok());
            assert!(open(&path, ReadLimits::untrusted(), lazy).is_ok());
        }
    }
}

#[test]
fn reads_are_held_to_the_read_limit() {
    let limits = ReadLimits::default().max_read_bytes(8_000);
    for format in [Format::Netcdf4, Format::Netcdf3Classic] {
        let path = file(format);
        // Lazily: opening reads only the coordinate (800 bytes); a slice within the limit
        // reads, the whole variable does not.
        let ds = open(&path, limits, true).unwrap();
        let t = &ds.data_vars()["t"];
        let mut part = t.isel([("y", 0..10)]).unwrap().to_owned();
        part.load().unwrap();
        assert_eq!(part.data::<f64>().unwrap()[[3, 4]], 304.0);
        let mut whole = t.clone();
        let e = whole.load().unwrap_err();
        assert!(
            matches!(
                e,
                CoreError::Limit {
                    found: 80_000,
                    limit: 8_000,
                    ..
                }
            ),
            "{e}"
        );
        // Eagerly: the variable is read when opening.
        let e = open(&path, limits, false).unwrap_err();
        assert_eq!(
            limit_of(&e),
            ("bytes of one read".to_owned(), 80_000, 8_000)
        );
    }
    // A chunk larger than the limit is refused when opening: any read would decode it whole.
    let path = file(Format::Netcdf4);
    let e = open(&path, ReadLimits::default().max_read_bytes(799), true).unwrap_err();
    assert_eq!(
        limit_of(&e),
        ("bytes of one chunk of /t".to_owned(), 800, 799)
    );

    // Zarr stores alike (the chunk shape from the array's metadata).
    let ds = open(&path, ReadLimits::default(), true).unwrap();
    let store = path.with_file_name("t.zarr");
    let _ = std::fs::remove_dir_all(&store);
    annoray::io::ToZarr::to_zarr(&ds, &store).unwrap();
    let e = open(&store, ReadLimits::default().max_read_bytes(799), true).unwrap_err();
    assert_eq!(
        limit_of(&e),
        ("bytes of one chunk of /t".to_owned(), 800, 799)
    );
    let ds = open(&store, limits, true).unwrap();
    assert!(ds.data_vars()["t"].clone().load().is_err());
}

#[test]
fn files_in_memory_and_trees_are_held_too() {
    let path = file(Format::Netcdf4);
    let bytes = std::fs::read(&path).unwrap();
    let opts = OpenOptions::default().limits(ReadLimits::default().max_rank(2));
    let e = open_dataset_from_bytes(bytes.clone(), &opts).unwrap_err();
    assert_eq!(limit_of(&e).0, "dimensions of /cube");
    assert!(open_dataset_from_bytes(bytes, &OpenOptions::default()).is_ok());

    // Group paths name the variable; the variables of every group count.
    let ds = open_dataset_with(&path, &OpenOptions::default())
        .unwrap()
        .dataset;
    let tree = DataTree::new(Dataset::new())
        .with_child("g", DataTree::new(ds.clone()))
        .unwrap()
        .with_child("h", DataTree::new(ds))
        .unwrap();
    let tpath = path.with_file_name("tree.nc");
    tree.to_netcdf(&tpath).unwrap();
    let opts = OpenOptions::default().limits(ReadLimits::default().max_elements(9_999));
    let e = open_datatree_with(&tpath, &opts).unwrap_err();
    assert_eq!(limit_of(&e).0, "values of /g/t");
    let opts = OpenOptions::default().limits(ReadLimits::default().max_variables(5));
    let e = open_datatree_with(&tpath, &opts).unwrap_err();
    assert_eq!(limit_of(&e), ("variables in the file".to_owned(), 6, 5));
}

/// A fuzzer's file: a variable the netCDF-4 reader reads while opening (before the metadata
/// checks) declares 2 GiB. The read limit holds that read too: the variable is skipped with a
/// warning, as the reader skips variables it cannot read, and nothing is allocated for it.
#[test]
fn reads_while_opening_are_held_too() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/fuzz/eager_string_oom.h5");
    let opts = OpenOptions::default().limits(ReadLimits::untrusted().max_read_bytes(16 << 20));
    let opened = open_dataset_with(&path, &opts).unwrap();
    assert!(!opened.dataset.data_vars().contains_key("time"));
    assert!(!opened.dataset.coords().contains_key("time"));
    assert!(
        opened
            .warnings
            .iter()
            .any(|w| w.contains("too large to load")),
        "{:?}",
        opened.warnings
    );
}
