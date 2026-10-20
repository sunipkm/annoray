//! open_mfdataset (`fixtures/nc/mf`, from `fixtures/gen_mf.py`): the parts, in any order, put
//! back together as the whole, lazily.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::path::PathBuf;

use annoray::Instant;
use annoray::io::{MfOptions, glob, open_dataset, open_mfdataset};
use annoray::{Dataset, Variable, VariableBase};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/mf")
}

fn floats<S: annoray::Repr>(v: &VariableBase<S>) -> Vec<f64> {
    v.data::<f64>().unwrap().iter().copied().collect()
}

fn same(a: &Dataset, b: &Dataset) {
    assert_eq!(a.sizes(), b.sizes());
    for (k, v) in b.data_vars() {
        assert_eq!(floats(&a.data_vars()[k.as_str()]), floats(v), "{k}");
        assert_eq!(a.data_vars()[k.as_str()].unit(), v.unit(), "{k}");
    }
    let t = |d: &Dataset| -> Vec<Instant> {
        d.coords()["time"]
            .data::<Instant>()
            .unwrap()
            .iter()
            .copied()
            .collect()
    };
    assert_eq!(t(a), t(b));
}

#[test]
fn parts_combine_by_coords_in_any_order() {
    let whole = open_dataset(dir().join("whole.nc"))
        .unwrap()
        .loaded()
        .unwrap();
    let mut parts = glob(dir().join("part_*.nc").to_str().unwrap()).unwrap();
    assert_eq!(parts.len(), 3);
    parts.reverse();
    let opened = open_mfdataset(&parts, &MfOptions::default()).unwrap();
    assert!(opened.warnings.is_empty(), "{:?}", opened.warnings);
    let mf = opened.dataset;
    // Still lazy: nothing has been read yet.
    assert!(mf.data_vars()["temp"].is_lazy());
    // A slice within one file, and one across files.
    let one = mf.data_vars()["temp"].isel([("time", 1..3)]).unwrap();
    assert_eq!(floats(&one), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let across = mf.data_vars()["temp"].isel([("time", 3..7)]).unwrap();
    assert_eq!(floats(&across), (9..21).map(f64::from).collect::<Vec<_>>());
    same(&mf.loaded().unwrap(), &whole);
}

#[test]
fn nested_concat_along_a_new_dimension() {
    let parts = glob(dir().join("part_*.nc").to_str().unwrap()).unwrap();
    // Each part along a new `run` dimension: they disagree on time, so the exact join fails.
    assert!(open_mfdataset(&parts, &MfOptions::default().concat_dim("run")).is_err());
    let same_part = vec![parts[0].clone(), parts[0].clone()];
    let mf = open_mfdataset(&same_part, &MfOptions::default().concat_dim("run"))
        .unwrap()
        .dataset;
    assert_eq!(mf.data_vars()["temp"].dims(), ["run", "time", "x"]);
    assert!(mf.data_vars()["temp"].is_lazy());
    let v: Variable = mf.data_vars()["temp"]
        .isel([("run", 1)])
        .unwrap()
        .to_owned();
    assert_eq!(floats(&v), (0..12).map(f64::from).collect::<Vec<_>>());
    // Along a new dimension every data variable is stacked, `elev` included.
    assert_eq!(mf.data_vars()["elev"].dims(), ["run", "x"]);
}

#[test]
fn errors() {
    assert!(open_mfdataset(Vec::<PathBuf>::new(), &MfOptions::default()).is_err());
    assert!(glob("[").is_err());
    let none = glob(dir().join("nothing_*.nc").to_str().unwrap()).unwrap();
    assert!(none.is_empty());
}
