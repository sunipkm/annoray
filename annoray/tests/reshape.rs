//! squeeze, expand_dims, swap_dims, drop_sel, sortby, diff, shift, roll, cumsum, cumprod
//! (expected values as Python xarray gives them, except the documented deviations).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::Unit;
use annoray::{CoreError, DataArray, Dataset, Variable};
use ndarray::{ArrayD, IxDyn, array};

const NAN: f64 = f64::NAN;

fn vals(d: &DataArray) -> Vec<f64> {
    d.to_f64().unwrap().iter().copied().collect()
}

fn same(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g.is_nan() && w.is_nan()) || g == w, "{got:?} vs {want:?}");
    }
}

fn coord(d: &DataArray, name: &str) -> Vec<f64> {
    d.coords()[name].to_f64().unwrap().iter().copied().collect()
}

/// `[1, NaN, 2, 3]` along x with coordinate `[3, 1, NaN, 2]`.
fn sample() -> DataArray {
    DataArray::from_array(["x"], array![1.0, NAN, 2.0, 3.0].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![3.0, 1.0, NAN, 2.0])
        .unwrap()
}

#[test]
fn cumulative_sums_and_products_skip_nan() {
    same(&vals(&sample().cumsum("x").unwrap()), &[1.0, 1.0, 3.0, 6.0]);
    same(
        &vals(&sample().cumprod("x").unwrap()),
        &[1.0, 1.0, 2.0, 6.0],
    );
    let ints = DataArray::from_array(["x"], array![1i32, 2, 3].into_dyn()).unwrap();
    let c = ints.cumsum("x").unwrap();
    assert_eq!(
        c.data::<i64>().unwrap().iter().copied().collect::<Vec<_>>(),
        [1, 3, 6]
    );
    // Units: a cumulative sum keeps them; °C and cumulative products with units refuse.
    let m = sample().with_unit("m".parse().unwrap());
    assert_eq!(m.cumsum("x").unwrap().unit(), m.unit());
    assert!(m.cumprod("x").is_err());
    assert!(
        sample()
            .with_unit("degC".parse().unwrap())
            .cumsum("x")
            .is_err()
    );
}

#[test]
fn sortby_puts_nan_last_both_ways() {
    let a = sample();
    same(
        &coord(&a.sortby("x", "x", true).unwrap(), "x"),
        &[1.0, 2.0, 3.0, NAN],
    );
    let d = a.sortby("x", "x", false).unwrap();
    same(&coord(&d, "x"), &[3.0, 2.0, 1.0, NAN]);
    same(&vals(&d), &[1.0, 3.0, NAN, 2.0]);
}

#[test]
fn diff_labels_and_units() {
    let a = sample();
    let d = a.diff("x", 1, true).unwrap();
    same(&vals(&d), &[NAN, NAN, 1.0]);
    same(&coord(&d, "x"), &[1.0, NAN, 2.0]);
    same(
        &coord(&a.diff("x", 1, false).unwrap(), "x"),
        &[3.0, 1.0, NAN],
    );
    let sq = DataArray::from_array(["t"], array![0.0, 1.0, 4.0, 9.0].into_dyn()).unwrap();
    same(&vals(&sq.diff("t", 2, true).unwrap()), &[2.0, 2.0]);
    // °C − °C is a temperature step.
    let c = DataArray::from_array(["t"], array![10.0, 12.5].into_dyn())
        .unwrap()
        .with_unit("degC".parse().unwrap());
    let step = c.diff("t", 1, true).unwrap();
    assert!(
        step.unit()
            .unwrap()
            .is_convertible(&"K".parse::<Unit>().unwrap())
    );
    same(&vals(&step), &[2.5]);
}

#[test]
fn shift_and_roll() {
    same(
        &vals(&sample().shift("x", 1).unwrap()),
        &[NAN, 1.0, NAN, 2.0],
    );
    let ints = DataArray::from_array(["x"], array![1i32, 2, 3].into_dyn()).unwrap();
    same(&vals(&ints.shift("x", -1).unwrap()), &[2.0, 3.0, NAN]);
    // Coordinates do not shift.
    same(
        &coord(&sample().shift("x", 1).unwrap(), "x"),
        &[3.0, 1.0, NAN, 2.0],
    );
    let r = sample().roll("x", 1, false).unwrap();
    same(&vals(&r), &[3.0, 1.0, NAN, 2.0]);
    same(&coord(&r, "x"), &[3.0, 1.0, NAN, 2.0]);
    same(
        &coord(&sample().roll("x", 1, true).unwrap(), "x"),
        &[2.0, 3.0, 1.0, NAN],
    );
    same(
        &vals(&sample().roll("x", -5, false).unwrap()),
        &[NAN, 2.0, 3.0, 1.0],
    );
}

#[test]
fn squeeze_and_expand_dims() {
    let b = DataArray::from_array(["x", "y"], ArrayD::from_elem(IxDyn(&[1, 3]), 1.0))
        .unwrap()
        .with_dim_coord("x", array![5.0])
        .unwrap();
    let s = b.squeeze(&[]).unwrap();
    assert_eq!(s.dims(), ["y"]);
    assert_eq!(s.coords()["x"].ndim(), 0);
    assert!(matches!(
        b.squeeze(&["y"]),
        Err(CoreError::InvalidReduction(_))
    ));
    // A scalar coordinate becomes the coordinate of the new dimension.
    let e = s.expand_dims("x", 0).unwrap();
    assert_eq!(e.dims(), ["x", "y"]);
    same(&coord(&e, "x"), &[5.0]);
    assert_eq!(s.expand_dims("z", -1).unwrap().dims(), ["y", "z"]);
    assert!(matches!(
        s.expand_dims("y", 0),
        Err(CoreError::DuplicateDim(_))
    ));
}

#[test]
fn swap_dims_and_drop_sel() {
    let d = DataArray::from_array(["x"], array![1.0, 2.0].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![0.0, 1.0])
        .unwrap()
        .with_coord(
            "lbl",
            Variable::new(["x"], array!["a".to_owned(), "b".to_owned()].into_dyn()).unwrap(),
        )
        .unwrap();
    let sw = d.swap_dims("x", "lbl").unwrap();
    assert_eq!(sw.dims(), ["lbl"]);
    assert_eq!(sw.coords()["x"].dims(), ["lbl"]);
    same(
        &vals(&annoray::sel!(sw[lbl = "b"]).unwrap().to_owned()),
        &[2.0],
    );
    assert!(d.swap_dims("x", "nope").is_err());
    same(&coord(&d.drop_sel("x", [1.0]).unwrap(), "x"), &[0.0]);
    assert!(d.drop_sel("x", [7.0]).is_err());
}

#[test]
fn datasets() {
    let ds = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![2.0, 0.0, 1.0].into_dyn()).unwrap(),
        )
        .unwrap();
    let mut ds = ds;
    ds.add_var(
        "a",
        Variable::new(["x"], array![20.0, 0.0, 10.0].into_dyn()).unwrap(),
    )
    .unwrap();
    ds.add_var("b", Variable::new(["y"], array![1.0].into_dyn()).unwrap())
        .unwrap();
    let sorted = ds.sortby("x", "x", true).unwrap();
    same(
        &vals(&sorted.get("a").unwrap().to_owned()),
        &[0.0, 10.0, 20.0],
    );
    let d = sorted.diff("x", 1, true).unwrap();
    same(&vals(&d.get("a").unwrap().to_owned()), &[10.0, 10.0]);
    assert_eq!(d.get("b").unwrap().dims(), ["y"]);
    let c = sorted.cumsum("x").unwrap();
    same(&vals(&c.get("a").unwrap().to_owned()), &[0.0, 10.0, 30.0]);
    assert_eq!(ds.squeeze(&[]).unwrap().get("b").unwrap().dims().len(), 0);
}

#[test]
fn lazy_data_stays_lazy() {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let ds = annoray::io::open_dataset(&path).unwrap();
    let (name, da) = ds
        .data_vars()
        .iter()
        .find(|(_, v)| {
            v.is_lazy()
                && v.ndim() >= 1
                && v.shape()[0] >= 2
                && v.dtype().to_string().starts_with("float")
        })
        .map(|(k, _)| (k.clone(), ds.get(k).unwrap().to_owned()))
        .unwrap();
    let dim = da.dims()[0].to_string();
    // A separate open: loading a clone would fill the lazy cache the clones share.
    let mut all = annoray::io::open_dataset(&path).unwrap();
    all.load().unwrap();
    let loaded = all.get(&name).unwrap().to_owned();
    for (what, lazy, eager) in [
        (
            "roll",
            da.roll(&dim, 1, false).unwrap(),
            loaded.roll(&dim, 1, false).unwrap(),
        ),
        (
            "shift",
            da.shift(&dim, 1).unwrap(),
            loaded.shift(&dim, 1).unwrap(),
        ),
        (
            "diff",
            da.diff(&dim, 1, true).unwrap(),
            loaded.diff(&dim, 1, true).unwrap(),
        ),
        (
            "expand",
            da.expand_dims("new", 0).unwrap(),
            loaded.expand_dims("new", 0).unwrap(),
        ),
    ] {
        assert!(lazy.variable().is_lazy(), "{name}: {what} loaded the data");
        same(&vals(&lazy), &vals(&eager));
    }
}
