//! stack and unstack (expected values as xarray gives them).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss
)]

use annoray::{CoreError, DType, DataArray, Dataset, Indexer, PySlice, Variable};
use ndarray::{Array2, Array3, array};

const NAN: f64 = f64::NAN;

fn vals(d: &DataArray) -> Vec<f64> {
    d.to_f64().unwrap().iter().copied().collect()
}

fn coord(d: &DataArray, name: &str) -> Vec<f64> {
    d.coords()[name].to_f64().unwrap().iter().copied().collect()
}

fn same(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g.is_nan() && w.is_nan()) || g == w, "{got:?} vs {want:?}");
    }
}

/// `10 x + y` over x = 30, 10, 20 (unsorted) and y = 0..2, plus t along x.
fn grid() -> DataArray {
    DataArray::from_array(
        ["x", "y"],
        Array2::from_shape_fn((3, 2), |(i, j)| (10 * i + j) as f64).into_dyn(),
    )
    .unwrap()
    .with_unit("m".parse().unwrap())
    .with_dim_coord("x", array![30.0, 10.0, 20.0])
    .unwrap()
    .with_dim_coord("y", array!["a".to_owned(), "b".to_owned()])
    .unwrap()
    .with_coord(
        "t",
        Variable::new(["x"], array![1.0, 2.0, 3.0].into_dyn()).unwrap(),
    )
    .unwrap()
}

#[test]
fn stack_flattens_row_major_at_the_end() {
    let s = grid().stack("z", &["x", "y"]).unwrap();
    assert_eq!(s.dims(), ["z"]);
    same(&vals(&s), &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
    same(&coord(&s, "x"), &[30.0, 30.0, 10.0, 10.0, 20.0, 20.0]);
    assert_eq!(s.coords()["y"].dims(), ["z"]);
    same(&coord(&s, "t"), &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    assert_eq!(s.unit(), grid().unit());
    assert_eq!(s.stacked_dims(), ["z"]);
    // Other dimensions come first.
    let a = DataArray::from_array(
        ["x", "w", "y"],
        Array3::from_shape_fn((2, 3, 2), |(i, k, j)| (100 * k + 10 * i + j) as f64).into_dyn(),
    )
    .unwrap();
    let s = a.stack("z", &["y", "x"]).unwrap();
    assert_eq!(s.dims(), ["w", "z"]);
    same(&vals(&s)[..4], &[0.0, 10.0, 1.0, 11.0]);
    // Without coordinates, the levels are 0..n.
    same(&coord(&s, "x"), &[0.0, 1.0, 0.0, 1.0]);
    assert!(matches!(
        a.stack("w", &["x"]),
        Err(CoreError::DuplicateDim(_))
    ));
    assert!(a.stack("z", &["nope"]).is_err());
}

#[test]
fn unstack_round_trips_in_original_order() {
    let g = grid();
    let back = g
        .stack("z", &["x", "y"])
        .unwrap()
        .unstack("z", None)
        .unwrap();
    assert_eq!(back.dims(), ["x", "y"]);
    same(&vals(&back), &vals(&g));
    same(&coord(&back, "x"), &[30.0, 10.0, 20.0]);
    assert_eq!(back.coords()["x"].dims(), ["x"]);
    assert_eq!(back.coords()["t"].dims(), ["x", "y"]);
    assert!(back.stacked_dims().is_empty());
    // Reversed level order.
    let r = g
        .stack("z", &["y", "x"])
        .unwrap()
        .unstack("z", None)
        .unwrap();
    assert_eq!(r.dims(), ["y", "x"]);
}

#[test]
fn absent_combinations_are_filled() {
    let s = grid().stack("z", &["x", "y"]).unwrap();
    let part = s
        .isel([("z", Indexer::Slice(PySlice::new(Some(1), Some(5), None)))])
        .unwrap()
        .to_owned();
    let u = part.unstack("z", None).unwrap();
    // Levels in order of appearance: x = 30, 10, 20; y = b, a.
    same(&coord(&u, "x"), &[30.0, 10.0, 20.0]);
    same(&vals(&u), &[1.0, NAN, 11.0, 10.0, NAN, 20.0]);
    // Integers become f64, unless the fill fits.
    let ints = DataArray::from_array(["x", "y"], array![[1i32, 2], [3, 4]].into_dyn()).unwrap();
    let si = ints.stack("z", &["x", "y"]).unwrap();
    let pi = si
        .isel([("z", Indexer::Slice(PySlice::new(None, Some(3), None)))])
        .unwrap()
        .to_owned();
    let f = pi.unstack("z", None).unwrap();
    assert_eq!(f.variable().dtype(), DType::F64);
    same(&vals(&f), &[1.0, 2.0, 3.0, NAN]);
    let z = pi.unstack("z", Some(0.0)).unwrap();
    assert_eq!(z.variable().dtype(), DType::I32);
    same(&vals(&z), &[1.0, 2.0, 3.0, 0.0]);
    assert_eq!(
        si.unstack("z", None).unwrap().variable().dtype(),
        DType::I32
    );
    let strs = DataArray::from_array(
        ["x", "y"],
        array![
            ["a".to_owned(), "b".to_owned()],
            ["c".to_owned(), "d".to_owned()]
        ]
        .into_dyn(),
    )
    .unwrap()
    .stack("z", &["x", "y"])
    .unwrap();
    let ps = strs
        .isel([("z", Indexer::Slice(PySlice::new(None, Some(3), None)))])
        .unwrap()
        .to_owned();
    assert!(matches!(
        ps.unstack("z", None),
        Err(CoreError::UnsupportedDType { .. })
    ));
}

#[test]
fn unmarked_levels_and_errors() {
    // Coordinates along the dimension, without markers (as read from a file).
    let flat = DataArray::from_array(["p"], array![1.0, 2.0, 3.0, 4.0].into_dyn())
        .unwrap()
        .with_coord(
            "a",
            Variable::new(["p"], array![0i64, 0, 1, 1].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_coord(
            "b",
            Variable::new(["p"], array![5i64, 6, 5, 6].into_dyn()).unwrap(),
        )
        .unwrap();
    let u = flat.unstack("p", None).unwrap();
    assert_eq!(u.dims(), ["a", "b"]);
    same(&vals(&u), &[1.0, 2.0, 3.0, 4.0]);
    let dup = DataArray::from_array(["p"], array![1.0, 2.0].into_dyn())
        .unwrap()
        .with_coord(
            "a",
            Variable::new(["p"], array![0i64, 0].into_dyn()).unwrap(),
        )
        .unwrap();
    assert!(dup.unstack("p", None).is_err());
    let bare = DataArray::from_array(["p"], array![1.0].into_dyn()).unwrap();
    assert!(bare.unstack("p", None).is_err());
}

#[test]
fn datasets() {
    let g = grid();
    let mut ds = Dataset::new();
    ds.add_array("v", g.clone()).unwrap();
    ds.add_var(
        "w",
        Variable::new(["y"], array![7.0, 8.0].into_dyn()).unwrap(),
    )
    .unwrap();
    ds.add_var(
        "c",
        Variable::new(Vec::<&str>::new(), ndarray::arr0(1.0).into_dyn()).unwrap(),
    )
    .unwrap();
    let s = ds.stack("z", &["x", "y"]).unwrap();
    assert_eq!(s.get("w").unwrap().dims(), ["z"]);
    same(
        &vals(&s.get("w").unwrap().to_owned()),
        &[7.0, 8.0, 7.0, 8.0, 7.0, 8.0],
    );
    assert_eq!(s.get("c").unwrap().dims().len(), 0);
    let u = s.unstack("z", None).unwrap();
    same(&vals(&u.get("v").unwrap().to_owned()), &vals(&g));
    // A broadcast variable comes back along both levels.
    assert_eq!(u.get("w").unwrap().dims(), ["x", "y"]);
}

#[test]
fn unstack_with_other_dimensions() {
    let a = DataArray::from_array(
        ["w", "x", "y"],
        Array3::from_shape_fn((2, 3, 4), |(k, i, j)| (100 * k + 10 * i + j) as f64).into_dyn(),
    )
    .unwrap();
    for dims in [["x", "y"], ["y", "w"], ["w", "x"]] {
        let s = a.stack("z", &dims).unwrap();
        let back = s.unstack("z", None).unwrap();
        let order: Vec<String> = back.dims().iter().map(ToString::to_string).collect();
        let refs: Vec<&str> = order.iter().map(String::as_str).collect();
        same(&vals(&back), &vals(&a.transpose(&refs).unwrap()));
        // With an absent combination.
        let part = s
            .isel([("z", Indexer::Slice(PySlice::new(Some(1), None, None)))])
            .unwrap()
            .to_owned();
        let u = part.unstack("z", None).unwrap();
        assert_eq!(
            vals(&u).iter().filter(|v| v.is_nan()).count(),
            vals(&u).len() / s.variable().shape()[s.dims().len() - 1]
        );
    }
}

#[test]
fn stacked_datasets_round_trip_through_files() {
    use annoray::io::{Format, OpenOptions, ToNetcdf, WriteOptions, open_dataset_with};
    let dir = std::env::temp_dir().join(format!("annoray-stack-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut ds = Dataset::new();
    ds.add_array("v", grid()).unwrap();
    let s = ds.stack("z", &["x", "y"]).unwrap();
    // Drop one combination: the file must still say where each point goes.
    let s = s
        .isel([("z", Indexer::Slice(PySlice::new(Some(1), None, None)))])
        .unwrap()
        .to_owned();
    for (i, format) in [Format::Netcdf4, Format::Netcdf3Classic]
        .into_iter()
        .enumerate()
    {
        let path = dir.join(format!("stacked{i}.nc"));
        s.to_netcdf_with(&path, &WriteOptions::format(format))
            .unwrap();
        // On disk: levels as dimensions, z as a gathered index.
        let raw = open_dataset_with(&path, &OpenOptions::default().decode(raw_coords()))
            .unwrap()
            .dataset;
        let z = raw.get("z").unwrap();
        assert_eq!(
            z.variable().attrs().get("compress"),
            Some(&annoray::AttrValue::Str("x y".into()))
        );
        // Levels in order of appearance (x: 30, 10, 20; y: b, a), positions row-major.
        same(&vals(&z.to_owned()), &[0.0, 3.0, 2.0, 5.0, 4.0]);
        assert_eq!(raw.get("x").unwrap().dims(), ["x"]);
        // Read back: stacked again, with `t` (stacked along) not taken for a level.
        let back = open_dataset_with(&path, &OpenOptions::default()).unwrap();
        assert!(back.warnings.is_empty(), "{:?}", back.warnings);
        let back = back.dataset;
        assert_eq!(back.stacked_dims(), ["z"]);
        assert!(!back.coords().contains_key("z"));
        let want = s.unstack("z", None).unwrap();
        let got = back.unstack("z", None).unwrap();
        same(
            &vals(&got.get("v").unwrap().to_owned()),
            &vals(&want.get("v").unwrap().to_owned()),
        );
        assert_eq!(got.get("v").unwrap().dims(), ["x", "y"]);
        assert_eq!(got.get("v").unwrap().unit(), want.get("v").unwrap().unit());
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn raw_coords() -> annoray::cf::DecodeOptions {
    let mut o = annoray::cf::DecodeOptions::default();
    o.decode_coords = false;
    o
}

#[test]
fn string_dimension_coordinates_stay_coordinates_in_netcdf3() {
    use annoray::io::{Format, OpenOptions, ToNetcdf, WriteOptions, open_dataset_with};
    let dir = std::env::temp_dir().join(format!("annoray-strcoord-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut ds = Dataset::new();
    ds.add_array("v", grid()).unwrap();
    let path = dir.join("s.nc");
    ds.to_netcdf_with(&path, &WriteOptions::format(Format::Netcdf3Classic))
        .unwrap();
    let back = open_dataset_with(&path, &OpenOptions::default())
        .unwrap()
        .dataset;
    assert!(back.coords().contains_key("y"));
    assert!(!back.data_vars().contains_key("y"));
    std::fs::remove_dir_all(&dir).ok();
}
