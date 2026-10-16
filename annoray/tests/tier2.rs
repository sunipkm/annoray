//! `equals`/`identical`/`broadcast_equals`, `broadcast`, `cov`/`corr`.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::Unit;
use annoray::{DataArray, Dataset, Name, broadcast, corr, cov};
use ndarray::array;

fn u(s: &str) -> Unit {
    s.parse().unwrap()
}

fn arr(v: ndarray::ArrayD<f64>) -> DataArray {
    DataArray::from_array(["x"], v)
        .unwrap()
        .with_dim_coord("x", array![0.0, 1.0, 2.0])
        .unwrap()
}

#[test]
fn equals_converts_units_identical_does_not() {
    let km = arr(array![1.0, f64::NAN, 2.5].into_dyn()).with_unit(u("km"));
    let m = arr(array![1000.0, f64::NAN, 2500.0].into_dyn()).with_unit(u("m"));
    assert!(km.equals(&m).unwrap(), "1 km equals 1000 m; NaN equals NaN");
    assert!(!km.identical(&m).unwrap());
    assert!(km.identical(&km.clone()).unwrap());
    let named = km.clone().with_name("n");
    assert!(km.equals(&named).unwrap() && !km.identical(&named).unwrap());
    let attrd = km.clone().with_attrs([("k", "v")]);
    assert!(km.equals(&attrd).unwrap() && !km.identical(&attrd).unwrap());
    let s = arr(array![1.0, f64::NAN, 2.5].into_dyn()).with_unit(u("s"));
    assert!(!km.equals(&s).unwrap());
    let other_x = km
        .clone()
        .with_dim_coord("x", array![5.0, 6.0, 7.0])
        .unwrap();
    assert!(!km.equals(&other_x).unwrap());
    let text = DataArray::from_array(["x"], array!["a", "b", "c"].mapv(String::from).into_dyn())
        .unwrap()
        .with_dim_coord("x", array![0.0, 1.0, 2.0])
        .unwrap();
    assert!(!km.equals(&text).unwrap());
    // Broadcast: a scalar against an array of that value.
    let three = arr(array![3.0, 3.0, 3.0].into_dyn());
    let scalar = DataArray::scalar(3.0).unwrap();
    assert!(three.broadcast_equals(&scalar).unwrap());
    assert!(!three.equals(&scalar).unwrap());
}

#[test]
fn dataset_equality_ignores_order_identical_checks_attrs() {
    let mut a = Dataset::new();
    a.add_var(
        "p",
        arr(array![1.0, 2.0, 3.0].into_dyn()).variable().clone(),
    )
    .unwrap();
    a.add_var(
        "q",
        arr(array![4.0, 5.0, 6.0].into_dyn()).variable().clone(),
    )
    .unwrap();
    let mut b = Dataset::new();
    b.add_var("q", a.data_vars()["q"].clone()).unwrap();
    b.add_var("p", a.data_vars()["p"].clone()).unwrap();
    assert!(a.equals(&b).unwrap() && a.identical(&b).unwrap());
    // NaN attributes equal each other (a _FillValue read twice).
    a.attrs_mut().insert("fill".into(), f64::NAN.into());
    b.attrs_mut().insert("fill".into(), f64::NAN.into());
    assert!(a.identical(&b).unwrap());
    b.attrs_mut().insert("extra".into(), "1".into());
    assert!(a.equals(&b).unwrap() && !a.identical(&b).unwrap());
}

#[test]
fn broadcast_prepends_missing_dims() {
    let x = arr(array![1.0, 2.0, 3.0].into_dyn());
    let y = DataArray::from_array(["y"], array![10.0, 20.0].into_dyn())
        .unwrap()
        .with_dim_coord("y", array![0.5, 1.5])
        .unwrap();
    let out = broadcast(&[&x, &y], &[]).unwrap();
    assert_eq!(out[0].dims(), ["x", "y"]);
    assert_eq!(out[1].dims(), ["x", "y"]);
    assert!(out[0].coords().contains_key("y") && out[1].coords().contains_key("x"));
    assert_eq!(
        out[0].to_f64().unwrap().iter().copied().collect::<Vec<_>>(),
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
    let like = x.broadcast_like(&y, &[]).unwrap();
    assert_eq!(like.dims(), ["x", "y"]);
    let excluded = x.broadcast_like(&y, &["y"]).unwrap();
    assert_eq!(excluded.dims(), ["x"]);
}

#[test]
fn covariance_and_correlation() {
    let a = arr(array![1.0, 2.0, 4.0].into_dyn()).with_unit(u("m"));
    let b = arr(array![2.0, 4.0, f64::NAN].into_dyn()).with_unit(u("s"));
    // Valid pairs: (1, 2), (2, 4): means 1.5 and 3; products 0.5·1 + 0.5·1 = 1; ddof 1 → 1.
    let c = cov(&a, &b, None, 1).unwrap();
    assert!((c.to_f64().unwrap().first().copied().unwrap() - 1.0).abs() < 1e-12);
    assert!(c.unit().unwrap().is_convertible(&u("m s")));
    let r = corr(&a, &b, None).unwrap();
    assert!((r.to_f64().unwrap().first().copied().unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(r.unit(), None);
    let dims = [Name::from("x")];
    assert_eq!(cov(&a, &a, Some(&dims), 0).unwrap().ndim(), 0);
}
