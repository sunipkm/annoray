//! Vectorized (pointwise) indexing, as xarray's `isel`/`sel` with `DataArray` indexers.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss
)]

use annoray::{CoreError, DataArray, Dataset, Label, Method, Variable};
use ndarray::{Array2, Array3, array};

fn vals(d: &DataArray) -> Vec<f64> {
    d.to_f64().unwrap().iter().copied().collect()
}

fn coord(d: &DataArray, name: &str) -> Vec<f64> {
    d.coords()[name].to_f64().unwrap().iter().copied().collect()
}

/// `10 x + y` on x = 0, 10, 20 and y = 0..4.
fn grid() -> DataArray {
    DataArray::from_array(
        ["x", "y"],
        Array2::from_shape_fn((3, 4), |(i, j)| (10 * i + j) as f64).into_dyn(),
    )
    .unwrap()
    .with_dim_coord("x", array![0.0, 10.0, 20.0])
    .unwrap()
    .with_dim_coord("y", array![0.0, 1.0, 2.0, 3.0])
    .unwrap()
}

fn ints(dim: &str, v: Vec<i64>) -> DataArray {
    DataArray::from_array([dim], ndarray::Array1::from(v).into_dyn()).unwrap()
}

#[test]
fn points_pick_one_element_each() {
    let ix = ints("points", vec![0, 2, -2]);
    let iy = ints("points", vec![1, 3, 0]);
    let p = grid().isel_vectorized(&[("x", &ix), ("y", &iy)]).unwrap();
    assert_eq!(p.dims(), ["points"]);
    assert_eq!(vals(&p), [1.0, 23.0, 10.0]);
    assert_eq!(coord(&p, "x"), [0.0, 20.0, 10.0]);
    assert_eq!(p.coords()["x"].dims(), ["points"]);
    assert_eq!(coord(&p, "y"), [1.0, 3.0, 0.0]);
}

#[test]
fn different_indexer_dims_broadcast() {
    let ix = ints("a", vec![0, 2]);
    let iy = ints("b", vec![1, 2, 3]);
    let p = grid().isel_vectorized(&[("x", &ix), ("y", &iy)]).unwrap();
    assert_eq!(p.dims(), ["a", "b"]);
    assert_eq!(vals(&p), [1.0, 2.0, 3.0, 21.0, 22.0, 23.0]);
    // A 2-D indexer.
    let i2 = DataArray::from_array(["u", "v"], array![[0i64, 1], [2, 0]].into_dyn()).unwrap();
    let q = grid().isel_vectorized(&[("x", &i2)]).unwrap();
    assert_eq!(q.dims(), ["u", "v", "y"]);
    assert_eq!(q.variable().shape(), [2, 2, 4]);
    assert_eq!(vals(&q)[4..8], [10.0, 11.0, 12.0, 13.0]);
}

#[test]
fn shared_free_dimension_is_matched_elementwise() {
    // Indexer for x along y: picks (i[j], j).
    let i = ints("y", vec![2, 1, 0, 2]);
    let d = grid().isel_vectorized(&[("x", &i)]).unwrap();
    assert_eq!(d.dims(), ["y"]);
    assert_eq!(vals(&d), [20.0, 11.0, 2.0, 23.0]);
}

#[test]
fn result_dims_follow_the_array_order() {
    let a = DataArray::from_array(
        ["z", "x", "y"],
        Array3::from_shape_fn((2, 3, 4), |(k, i, j)| (100 * k + 10 * i + j) as f64).into_dyn(),
    )
    .unwrap();
    let ix = ints("points", vec![1, 2]);
    let iy = ints("points", vec![0, 3]);
    let p = a.isel_vectorized(&[("x", &ix), ("y", &iy)]).unwrap();
    assert_eq!(p.dims(), ["z", "points"]);
    assert_eq!(vals(&p), [10.0, 23.0, 110.0, 123.0]);
}

#[test]
fn masks_bounds_and_errors() {
    let m = DataArray::from_array(["x"], array![true, false, true].into_dyn()).unwrap();
    let r = grid().isel_vectorized(&[("x", &m)]).unwrap();
    assert_eq!(r.dims(), ["x", "y"]);
    assert_eq!(coord(&r, "x"), [0.0, 20.0]);
    let oob = ints("p", vec![3]);
    assert!(matches!(
        grid().isel_vectorized(&[("x", &oob)]),
        Err(CoreError::Index(_))
    ));
    assert!(grid().isel_vectorized(&[("nope", &oob)]).is_err());
    let f = DataArray::from_array(["p"], array![0.5].into_dyn()).unwrap();
    assert!(grid().isel_vectorized(&[("x", &f)]).is_err());
    let (a, b) = (ints("p", vec![0, 1]), ints("p", vec![0, 1, 2]));
    assert!(grid().isel_vectorized(&[("x", &a), ("y", &b)]).is_err());
}

#[test]
fn indexer_coordinates_are_attached() {
    let ix = ints("points", vec![0, 2])
        .with_dim_coord("points", array!["a".to_owned(), "b".to_owned()])
        .unwrap();
    let p = grid().isel_vectorized(&[("x", &ix)]).unwrap();
    assert_eq!(p.coords()["points"].dims(), ["points"]);
    // A conflicting coordinate of the same name is refused.
    let clash = ints("points", vec![0])
        .with_coord(
            "y",
            Variable::new(["points"], array![9.0].into_dyn()).unwrap(),
        )
        .unwrap();
    assert!(grid().isel_vectorized(&[("x", &clash)]).is_err());
}

#[test]
fn sel_by_label_arrays() {
    let lx = DataArray::from_array(["points"], array![20.0, 0.0].into_dyn()).unwrap();
    let ly = DataArray::from_array(["points"], array![2.9, 0.2].into_dyn()).unwrap();
    let p = grid()
        .sel_vectorized(&[("x", &lx), ("y", &ly)], Method::Nearest, None)
        .unwrap();
    assert_eq!(vals(&p), [23.0, 0.0]);
    assert!(
        grid()
            .sel_vectorized(&[("y", &ly)], Method::Exact, None)
            .is_err()
    );
    // Units: labels in metres on a km axis.
    let km = grid()
        .with_coord(
            "x",
            Variable::new(["x"], array![0.0, 10.0, 20.0].into_dyn())
                .unwrap()
                .with_unit("km".parse().unwrap()),
        )
        .unwrap();
    let lm = DataArray::from_array(["p"], array![10000.0].into_dyn())
        .unwrap()
        .with_unit("m".parse().unwrap());
    let s = km
        .sel_vectorized(&[("x", &lm)], Method::Exact, None)
        .unwrap();
    assert_eq!(s.dims(), ["p", "y"]);
    assert_eq!(vals(&s), [10.0, 11.0, 12.0, 13.0]);
    let tol = Label::Float(0.5);
    let near = DataArray::from_array(["p"], array![10.4].into_dyn()).unwrap();
    assert!(
        grid()
            .sel_vectorized(&[("x", &near)], Method::Nearest, Some(&tol))
            .is_ok()
    );
}

#[test]
fn datasets_and_lazy_data() {
    let g = grid();
    let mut ds = Dataset::new();
    for (k, c) in g.coords() {
        ds.add_coord(k.clone(), c.clone()).unwrap();
    }
    ds.add_var("v", g.variable().clone()).unwrap();
    ds.add_var(
        "w",
        Variable::new(["y"], array![5.0, 6.0, 7.0, 8.0].into_dyn()).unwrap(),
    )
    .unwrap();
    let ix = ints("points", vec![2, 0]);
    let iy = ints("points", vec![3, 1]);
    let p = ds.isel_vectorized(&[("x", &ix), ("y", &iy)]).unwrap();
    assert_eq!(vals(&p.get("v").unwrap().to_owned()), [23.0, 1.0]);
    assert_eq!(vals(&p.get("w").unwrap().to_owned()), [8.0, 6.0]);

    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let lazy = annoray::io::open_dataset(&path)
        .unwrap()
        .get("temp")
        .unwrap()
        .to_owned();
    let mut all = annoray::io::open_dataset(&path).unwrap();
    all.load().unwrap();
    let eager = all.get("temp").unwrap().to_owned();
    let (d1, d2) = (lazy.dims()[1].to_string(), lazy.dims()[2].to_string());
    let a = ints("points", vec![0, 1, 0]);
    let b = ints("points", vec![1, 0, 1]);
    let l = lazy
        .isel_vectorized(&[(d1.as_str(), &a), (d2.as_str(), &b)])
        .unwrap();
    let e = eager
        .isel_vectorized(&[(d1.as_str(), &a), (d2.as_str(), &b)])
        .unwrap();
    assert_eq!(vals(&l), vals(&e));
    assert!(lazy.variable().is_lazy());
}
