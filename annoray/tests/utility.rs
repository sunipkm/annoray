//! `astype`, `full_like`, three-way `where`, dataset renames and `expand_dims`, and dropping
//! coordinates.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::Unit;
use annoray::{CoreError, DType, DataArray, Dataset, Variable, where_};
use ndarray::array;

fn u(s: &str) -> Unit {
    s.parse().unwrap()
}

fn f64s(a: &DataArray) -> Vec<f64> {
    a.to_f64().unwrap().iter().copied().collect()
}

fn da(values: ndarray::ArrayD<f64>, unit: Option<&str>) -> DataArray {
    let mut a = DataArray::from_array(["x"], values)
        .unwrap()
        .with_dim_coord("x", array![0.0, 1.0, 2.0])
        .unwrap()
        .with_name("a")
        .with_attrs([("k", "v")]);
    if let Some(s) = unit {
        a = a.with_unit(u(s));
    }
    a
}

#[test]
fn astype_converts_exactly_or_fails() {
    let a = da(array![-1.7, 0.0, 2.9].into_dyn(), Some("m"));
    let i = a.astype(DType::I32).unwrap();
    assert_eq!(i.dtype(), DType::I32);
    assert_eq!(
        i.data::<i32>().unwrap().iter().copied().collect::<Vec<_>>(),
        [-1, 0, 2]
    );
    assert_eq!(i.unit(), Some(&u("m")));
    assert_eq!(i.attrs()["k"], "v".into());
    // NaN has no integer; numpy would give INT_MIN.
    let nan = da(array![1.0, f64::NAN, 2.0].into_dyn(), None);
    assert!(matches!(
        nan.astype(DType::I64),
        Err(CoreError::Cast { .. })
    ));
    // 300 does not fit u8 (numpy: 44).
    let big = DataArray::from_array(["x"], array![300_i64].into_dyn()).unwrap();
    assert!(matches!(big.astype(DType::U8), Err(CoreError::Cast { .. })));
    assert!(matches!(
        da(array![1e300, 0.0, 0.0].into_dyn(), None).astype(DType::F32),
        Err(CoreError::Cast { .. })
    ));
    // Booleans: nonzero, NaN true; no unit.
    let b = nan.astype(DType::Bool).unwrap();
    assert_eq!(
        b.data::<bool>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [true, true, true]
    );
    let b = a.astype(DType::Bool).unwrap();
    assert_eq!(b.unit(), None);
    let text = DataArray::from_array(["x"], array!["a".to_owned()].into_dyn()).unwrap();
    assert!(matches!(
        text.astype(DType::F64),
        Err(CoreError::UnsupportedDType { .. })
    ));
}

#[test]
fn full_like_keeps_shape_and_unit() {
    let a = da(array![1.0, 2.0, 3.0].into_dyn(), Some("K"));
    let f = a.full_like(7.0, None).unwrap();
    assert_eq!(f64s(&f), [7.0, 7.0, 7.0]);
    assert_eq!(f.unit(), Some(&u("K")));
    assert_eq!(f.name().map(ToString::to_string).as_deref(), Some("a"));
    assert!(f.coords().contains_key("x"));
    // An integer fill must be whole (numpy truncates 0.5 to 0).
    assert!(matches!(
        a.full_like(0.5, Some(DType::I32)),
        Err(CoreError::Cast { .. })
    ));
    let i = a.full_like(3.0, Some(DType::I16)).unwrap();
    assert_eq!(i.dtype(), DType::I16);
    let flags = a.full_like(1.0, Some(DType::Bool)).unwrap();
    assert_eq!(flags.unit(), None);
    assert!(a.full_like(2.0, Some(DType::Bool)).is_err());
}

#[test]
fn where_picks_converts_and_broadcasts() {
    let x = da(array![1.0, 2.0, 3.0].into_dyn(), Some("m"));
    let cond = x
        .gt(&DataArray::scalar(1.5).unwrap().with_unit(u("m")))
        .unwrap();
    // Kilometres become metres.
    let y = DataArray::scalar(1.0).unwrap().with_unit(u("km"));
    let r = where_(&cond, &x, &y).unwrap();
    assert_eq!(f64s(&r), [1000.0, 2.0, 3.0]);
    assert_eq!(r.unit(), Some(&u("m")));
    assert_eq!(r.attrs()["k"], "v".into());
    // A bare number is refused, except 0 and NaN.
    let five = DataArray::scalar(5.0).unwrap();
    assert!(where_(&cond, &x, &five).is_err());
    let nan = DataArray::scalar(f64::NAN).unwrap();
    assert!(f64s(&where_(&cond, &x, &nan).unwrap())[0].is_nan());
    let zero = DataArray::scalar(0.0).unwrap();
    assert_eq!(f64s(&x.where_else(&cond, &zero).unwrap()), [0.0, 2.0, 3.0]);
    // Broadcasting: a condition along y over data along x.
    let c2 = DataArray::from_array(["y"], array![true, false].into_dyn()).unwrap();
    let r = where_(&c2, &x, &zero).unwrap();
    assert_eq!(r.dims(), ["y", "x"]);
    assert_eq!(f64s(&r), [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    let r = x.where_else(&c2, &zero).unwrap();
    assert_eq!(r.dims(), ["x", "y"]);
    // Same-type branches keep their type (text here).
    let t = DataArray::from_array(["x"], array!["a", "b", "c"].mapv(String::from).into_dyn())
        .unwrap()
        .with_dim_coord("x", array![0.0, 1.0, 2.0])
        .unwrap();
    let dash = DataArray::scalar(String::from("-")).unwrap();
    let r = where_(&cond, &t, &dash).unwrap();
    assert_eq!(
        r.data::<String>()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        ["-", "b", "c"]
    );
    // Shared dimensions must be labelled alike (an exact join).
    let shifted = da(array![1.0, 2.0, 3.0].into_dyn(), Some("m"))
        .with_dim_coord("x", array![1.0, 2.0, 3.0])
        .unwrap();
    assert!(where_(&cond, &x, &shifted).is_err());
}

fn sample() -> Dataset {
    let mut ds = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![10.0, 20.0].into_dyn()).unwrap(),
        )
        .unwrap();
    ds.add_var(
        "a",
        Variable::new(
            ["x", "y"],
            array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
        )
        .unwrap(),
    )
    .unwrap();
    ds.add_var(
        "b",
        Variable::new(["x"], array![1.0, 2.0].into_dyn()).unwrap(),
    )
    .unwrap();
    ds.attrs_mut().insert("title".into(), "t".into());
    ds
}

fn keys<T>(m: &indexmap::IndexMap<annoray::Name, T>) -> Vec<String> {
    m.keys().map(ToString::to_string).collect()
}

#[test]
fn dataset_renames() {
    let ds = sample();
    let r = ds.rename(&[("x", "xx"), ("a", "aa")]).unwrap();
    assert_eq!(keys(r.data_vars()), ["aa", "b"]);
    assert_eq!(keys(r.coords()), ["xx"]);
    assert_eq!(r.get("aa").unwrap().dims(), ["xx", "y"]);
    assert_eq!(r.attrs()["title"], "t".into());
    // Only the variable: the coordinate x is now along dimension x under another name.
    let r = ds.rename_vars(&[("x", "xx")]).unwrap();
    assert_eq!(r.coords()["xx"].dims(), ["x"]);
    assert_eq!(r.get("a").unwrap().dims(), ["x", "y"]);
    // Only the dimension.
    let r = ds.rename_dims(&[("y", "z")]).unwrap();
    assert_eq!(r.get("a").unwrap().dims(), ["x", "z"]);
    // Swapped at once.
    let r = ds.rename_vars(&[("a", "b"), ("b", "a")]).unwrap();
    assert_eq!(r.get("a").unwrap().dims(), ["x"]);
    // Taken names.
    assert!(matches!(
        ds.rename_vars(&[("a", "b")]),
        Err(CoreError::NameConflict(_))
    ));
    assert!(ds.rename_dims(&[("y", "x")]).is_err());
    assert!(matches!(
        ds.rename(&[("nope", "z")]),
        Err(CoreError::NoSuchVariable(_))
    ));
}

#[test]
fn dataset_expand_dims_and_dropping_coordinates() {
    let ds = sample();
    let e = ds.expand_dims("t", 0).unwrap();
    assert_eq!(e.get("a").unwrap().dims(), ["t", "x", "y"]);
    assert_eq!(e.get("b").unwrap().dims(), ["t", "x"]);
    assert_eq!(e.coords()["x"].dims(), ["x"]);
    let e = ds.expand_dims("t", -1).unwrap();
    assert_eq!(e.get("a").unwrap().dims(), ["x", "y", "t"]);
    assert!(ds.expand_dims("x", 0).is_err());

    let a = da(array![1.0, 2.0, 3.0].into_dyn(), None);
    let d = a.drop_coords(&["x"]).unwrap();
    assert!(d.coords().is_empty());
    assert_eq!(d.dims(), ["x"]);
    assert!(matches!(
        a.drop_coords(&["nope"]),
        Err(CoreError::NoSuchVariable(_))
    ));
}

#[test]
fn lazy_data_stays_lazy() {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let ds = annoray::io::open_dataset(&path).unwrap();
    let (k, v) = ds
        .data_vars()
        .iter()
        .find(|(_, v)| v.is_lazy() && v.dtype() == DType::F64)
        .unwrap();
    let cast = v.astype(DType::F32).unwrap();
    assert!(cast.is_lazy(), "{k}");
    let filled = v.full_like(0.0, None).unwrap();
    assert!(!filled.is_lazy());
    assert_eq!(filled.shape(), v.shape());
    let r = ds.rename_vars(&[(k.as_str(), "renamed")]).unwrap();
    assert!(r.data_vars()["renamed"].is_lazy());
}

#[test]
fn in_place_assignment_checks_first() {
    let mut ds = sample();
    let before = ds.clone();
    // Wrong length for x: refused, nothing changed.
    let bad = DataArray::from_array(["x"], array![1.0, 2.0, 3.0].into_dyn()).unwrap();
    assert!(ds.set_array("c", bad).is_err());
    assert_eq!(keys(ds.data_vars()), keys(before.data_vars()));
    // Replacing returns the displaced variable.
    let twice = DataArray::from_array(["x"], array![5.0, 6.0].into_dyn()).unwrap();
    let displaced = ds.set_array("b", twice).unwrap();
    assert_eq!(displaced.len(), 1);
    assert_eq!(
        displaced[0].to_f64().unwrap().as_slice().unwrap(),
        [1.0, 2.0]
    );
    // A coordinate's name replaces the coordinate.
    let x = DataArray::from_array(["x"], array![1.0, 2.0].into_dyn()).unwrap();
    assert_eq!(ds.set_array("x", x).unwrap().len(), 1);
    assert!(ds.data_vars().get("x").is_none());
    ds.attrs_of_mut("a").unwrap().insert("k".into(), "v".into());
    assert_eq!(ds.data_vars()["a"].attrs()["k"], "v".into());
    assert!(
        ds.set_coord_var(
            "a",
            Variable::new(["x"], array![0.0, 0.0].into_dyn()).unwrap()
        )
        .is_err()
    );

    let mut a = da(array![1.0, 2.0, 3.0].into_dyn(), None);
    assert!(
        a.set_coord_var("lab", Variable::new(["z"], array![0.0].into_dyn()).unwrap())
            .is_err()
    );
    let old = a
        .set_coord_var(
            "x",
            Variable::new(["x"], array![7.0, 8.0, 9.0].into_dyn()).unwrap(),
        )
        .unwrap();
    assert!(old.is_some());
    a.set_name(None);
    assert!(a.name().is_none());
    a.remove_coord("x").unwrap();
    assert!(a.coords().is_empty());
}

#[test]
fn values_slice_borrows_or_copies_in_order() {
    use std::borrow::Cow;
    let v = annoray::Variable::new(
        ["y", "x"],
        annoray::ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
    )
    .unwrap();
    let s = v.values_slice::<f64>().unwrap();
    assert!(matches!(s, Cow::Borrowed(_)));
    assert_eq!(s.as_ptr(), v.data::<f64>().unwrap().as_ptr());
    // A reversed view is not stored in order: copied, in its own order.
    let r = annoray::isel!(v[:, ::-1]).unwrap();
    let s = r.values_slice::<f64>().unwrap();
    assert!(matches!(s, Cow::Owned(_)));
    assert_eq!(*s, [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    assert!(v.values_slice::<i32>().is_err());
}
