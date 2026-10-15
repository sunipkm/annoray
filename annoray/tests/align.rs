//! Alignment against Python xarray (`fixtures/align.txt`, from `fixtures/gen_align.py`).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use annoray::{
    CoreError, DataArray, Dataset, Join, TypedArray, Variable, View, align, align_datasets,
};
use chrono::NaiveDateTime;
use ndarray::{Array1, ArrayD, IxDyn, array};

fn da<T: annoray::Element, C: annoray::Element>(
    dim: &str,
    values: Vec<T>,
    coord: Array1<C>,
) -> DataArray {
    DataArray::from_array(
        [dim],
        ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap(),
    )
    .unwrap()
    .with_dim_coord(dim, coord)
    .unwrap()
}

fn dates(days: &[u32]) -> Array1<NaiveDateTime> {
    days.iter()
        .map(|d| format!("2020-01-{d:02}T00:00:00").parse().unwrap())
        .collect()
}

fn s(v: &[&str]) -> Array1<String> {
    v.iter().map(|x| (*x).to_owned()).collect()
}

fn case(name: &str) -> Vec<DataArray> {
    match name {
        "num" => vec![
            da(
                "x",
                vec![10.0, 20.0, 30.0, 40.0],
                array![1.0, 2.0, 3.0, 4.0],
            ),
            da("x", vec![1i32, 2, 3], array![3.0, 4.0, 5.0]),
        ],
        "unsorted" => vec![
            da("x", vec![1.0, 2.0, 3.0], array![3.0, 1.0, 2.0]),
            da("x", vec![4.0, 5.0, 6.0], array![2.0, 3.0, 9.0]),
        ],
        "desc" => vec![
            da("x", vec![1.0, 2.0, 3.0], array![30.0, 20.0, 10.0]),
            da("x", vec![4.0, 5.0], array![20.0, 10.0]),
        ],
        "str" => vec![
            da("s", vec![1.0, 2.0, 3.0], s(&["c", "a", "b"])),
            da("s", vec![4.0, 5.0], s(&["b", "d"])),
        ],
        "time" => vec![
            da("t", vec![1.0, 2.0, 3.0], dates(&[1, 2, 3])),
            da("t", vec![4.0, 5.0], dates(&[3, 4])),
        ],
        "equal" => vec![
            da("x", vec![1.0, 2.0], array![1.0, 2.0]),
            da("x", vec![3.0, 4.0], array![1.0, 2.0]),
        ],
        "three" => vec![
            da("x", vec![1.0, 2.0, 3.0], array![1.0, 2.0, 3.0]),
            da("x", vec![4.0, 5.0, 6.0], array![2.0, 3.0, 4.0]),
            da("x", vec![7.0, 8.0], array![3.0, 2.0]),
        ],
        "nocoord" => vec![
            DataArray::from_array(
                ["x", "y"],
                array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn(),
            )
            .unwrap()
            .with_dim_coord("x", array![1.0, 2.0, 3.0])
            .unwrap(),
            DataArray::from_array(["y"], array![7.0, 8.0].into_dyn()).unwrap(),
            da("x", vec![9.0, 10.0], array![3.0, 1.0]),
        ],
        _ => panic!("unknown case {name}"),
    }
}

fn join(name: &str) -> Join {
    match name {
        "inner" => Join::Inner,
        "outer" => Join::Outer,
        "left" => Join::Left,
        "right" => Join::Right,
        "exact" => Join::Exact,
        _ => panic!("unknown join {name}"),
    }
}

fn strings(t: &TypedArray<View<'_>>) -> Vec<String> {
    match t {
        TypedArray::F64(a) => a.iter().map(|v| format!("{v:?}")).collect(),
        TypedArray::I32(a) => a.iter().map(i32::to_string).collect(),
        TypedArray::Str(a) => a.iter().cloned().collect(),
        TypedArray::DateTime(a) => a
            .iter()
            .map(|v| v.format("%Y-%m-%dT%H:%M:%S").to_string())
            .collect(),
        _ => panic!("unexpected type"),
    }
}

fn numpy_dtype(t: &TypedArray<View<'_>>) -> &'static str {
    match t {
        TypedArray::F64(_) => "float64",
        TypedArray::I32(_) => "int32",
        _ => panic!("unexpected type"),
    }
}

fn same(got: &[String], want: &str) {
    let want: Vec<&str> = want.split(' ').collect();
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(&want) {
        let ok = match (g.parse::<f64>(), w.parse::<f64>()) {
            (Ok(a), Ok(b)) => (a.is_nan() && b.is_nan()) || a == b,
            _ => g == w,
        };
        assert!(ok, "{got:?} vs {want:?}");
    }
}

#[test]
fn matches_xarray() {
    let table = include_str!("../../fixtures/align.txt");
    let mut checked = 0;
    for line in table.lines() {
        let f: Vec<&str> = line.split('|').collect();
        if f[0] == "arith" {
            let c = case("num");
            let r = (&c[0] + &c[1]).unwrap();
            same(&strings(&r.coord("x").unwrap().typed().unwrap()), f[3]);
            same(&strings(&r.variable().typed().unwrap()), f[6]);
            checked += 1;
            continue;
        }
        let objs = case(f[0]);
        let refs: Vec<&DataArray> = objs.iter().collect();
        let got = align(&refs, join(f[1]));
        if f[2] == "error" {
            assert!(
                matches!(got, Err(CoreError::NotAligned(_))),
                "{line}: {got:?}"
            );
            checked += 1;
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{line}: {e}"));
        let o = &got[f[2].parse::<usize>().unwrap()];
        let dim = o.dims()[0].clone();
        if f[3] == "-" {
            assert!(o.coords().get(&dim).is_none(), "{line}");
        } else {
            same(&strings(&o.coord(&dim).unwrap().typed().unwrap()), f[3]);
        }
        let values = o.variable().typed().unwrap();
        assert_eq!(numpy_dtype(&values), f[4], "{line}");
        let shape = o
            .shape()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(shape, f[5], "{line}");
        same(&strings(&values), f[6]);
        checked += 1;
    }
    assert!(checked > 70, "only {checked} cases");
}

#[test]
fn errors_and_laziness() {
    // Coordinates of different types cannot be aligned.
    let a = da("x", vec![1.0, 2.0], array![1.0, 2.0]);
    let b = da("x", vec![1.0, 2.0], s(&["a", "b"]));
    assert!(matches!(
        align(&[&a, &b], Join::Inner),
        Err(CoreError::CannotAlign { .. })
    ));
    // Missing labels in string data have no fill value.
    let c = da("x", vec!["p".to_owned(), "q".into()], array![1.0, 2.0]);
    let d = da("x", vec![1.0], array![3.0]);
    assert!(matches!(
        align(&[&c, &d], Join::Outer),
        Err(CoreError::UnsupportedDType { .. })
    ));
    // Duplicate labels cannot be reindexed.
    let e = da("x", vec![1.0, 2.0], array![1.0, 1.0]);
    assert!(align(&[&e, &a], Join::Outer).is_err());
    // A dimension without a coordinate must match the aligned length.
    let f = DataArray::from_array(["x"], array![1.0, 2.0, 3.0].into_dyn()).unwrap();
    assert!(matches!(
        align(&[&a, &f], Join::Inner),
        Err(CoreError::DimLengthMismatch { .. })
    ));
}

#[test]
fn datasets() {
    let ds1 = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![1.0, 2.0, 3.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var(
            "a",
            Variable::new(["x"], array![1i64, 2, 3].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var(
            "b",
            Variable::new(["y"], array![true, false].into_dyn()).unwrap(),
        )
        .unwrap();
    let ds2 = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![3.0, 2.0].into_dyn()).unwrap(),
        )
        .unwrap();
    let out = align_datasets(&[&ds1, &ds2], Join::Inner).unwrap();
    let a = out[0].data_vars()["a"].data::<i64>().unwrap();
    assert_eq!(a.iter().copied().collect::<Vec<_>>(), [2, 3]);
    // Variables without the dimension are untouched, booleans included.
    assert_eq!(out[0].data_vars()["b"].dtype(), annoray::DType::Bool);
    let x = out[1].coords()["x"].data::<f64>().unwrap();
    assert_eq!(x.iter().copied().collect::<Vec<_>>(), [2.0, 3.0]);
}
