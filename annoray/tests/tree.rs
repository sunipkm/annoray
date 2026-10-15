//! DataTree: reading xarray's tree file (`fixtures/nc/tree.nc`, from `fixtures/gen_tree.py`),
//! inheritance, validation, and writing.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::path::PathBuf;

use annoray::io::{Format, OpenOptions, ToNetcdf, WriteOptions, open_datatree, open_datatree_with};
use annoray::ndarray::array;
use annoray::{CoreError, DataTree, Dataset, Variable};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/tree.nc")
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("annoray-tree-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn check(tree: &DataTree) {
    assert_eq!(tree.paths(), ["/", "/obs", "/obs/hourly", "/empty"]);
    // /obs does not store x itself; it inherits it.
    let obs = tree.get("/obs").unwrap();
    assert!(!obs.dataset().contains("x"));
    let inherited = tree.inherited("obs").unwrap();
    assert_eq!(
        inherited.coords()["x"]
            .data::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [10.0, 20.0, 30.0]
    );
    assert_eq!(
        inherited.data_vars()["temp"]
            .unit()
            .map(ToString::to_string)
            .as_deref(),
        Some("K")
    );
    let hourly = tree.inherited("/obs/hourly").unwrap();
    // Only dimension coordinates are inherited (as in xarray): `x`, not `/obs`'s `station`.
    assert!(hourly.coords().contains_key("x"));
    assert!(!hourly.coords().contains_key("station"));
    assert_eq!(
        hourly.coords().keys().next().map(|k| k.as_str()),
        Some("x"),
        "inherited first"
    );
    assert_eq!(hourly.data_vars()["v"].shape(), [2, 3]);
    assert!(tree.get("empty").unwrap().dataset().data_vars().is_empty());
}

#[test]
fn reads_xarray_trees() {
    for lazy in [true, false] {
        let opened = open_datatree_with(fixture(), &OpenOptions::default().lazy(lazy)).unwrap();
        assert!(opened.warnings.is_empty(), "{:?}", opened.warnings);
        check(&opened.tree);
    }
    // A subtree.
    let sub = open_datatree_with(fixture(), &OpenOptions::default().group("obs")).unwrap();
    assert_eq!(sub.tree.paths(), ["/", "/hourly"]);
    assert!(format!("{}", sub.tree).contains("Group: /hourly"));
}

#[test]
fn round_trips() {
    let tree = open_datatree(fixture()).unwrap();
    let path = tmp("rt.nc");
    tree.to_netcdf(&path).unwrap();
    check(&open_datatree(&path).unwrap());
    // Groups need netCDF-4.
    assert!(
        tree.to_netcdf_with(
            tmp("rt3.nc"),
            &WriteOptions::format(Format::Netcdf3Offset64)
        )
        .is_err()
    );
}

#[test]
fn validation_and_mapping() {
    let root = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![1.0, 2.0].into_dyn()).unwrap(),
        )
        .unwrap();
    let wrong_len = Dataset::new()
        .with_var(
            "v",
            Variable::new(["x"], array![1.0, 2.0, 3.0].into_dyn()).unwrap(),
        )
        .unwrap();
    let mut tree = DataTree::new(root.clone());
    assert!(matches!(
        tree.add_child("c", DataTree::new(wrong_len)),
        Err(CoreError::CannotCombine { .. })
    ));
    let wrong_coord = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![1.0, 9.0].into_dyn()).unwrap(),
        )
        .unwrap();
    assert!(tree.add_child("c", DataTree::new(wrong_coord)).is_err());
    let ok = Dataset::new()
        .with_var(
            "v",
            Variable::new(["x"], array![3.0, 4.0].into_dyn()).unwrap(),
        )
        .unwrap();
    tree.add_child("c", DataTree::new(ok)).unwrap();
    assert!(tree.add_child("c", DataTree::default()).is_err());
    assert!(tree.add_child("a/b", DataTree::default()).is_err());

    let doubled = tree
        .map_over_datasets(&mut |ds| {
            let mut out = ds.clone();
            if let Some(v) = ds.data_vars().get("v") {
                out.remove("v")?;
                out.add_var(
                    "v",
                    Variable::new(["x"], v.data::<f64>()?.mapv(|x| x * 2.0))?,
                )?;
            }
            Ok(out)
        })
        .unwrap();
    assert_eq!(
        doubled.get("c").unwrap().dataset().data_vars()["v"]
            .data::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [6.0, 8.0]
    );
}

#[test]
#[ignore = "needs python3 with xarray"]
fn python_xarray_reads_our_tree() {
    let path = tmp("py.nc");
    open_datatree(fixture()).unwrap().to_netcdf(&path).unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/gen_tree.py");
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
