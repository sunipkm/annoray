//! isnull, fillna, where_, dropna, reindex (expected values from Python xarray).
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use chrono::NaiveDateTime;
use ndarray::{Array1, array};
use xr_core::{DataArray, Dataset, How, NaT, Variable};

const NAN: f64 = f64::NAN;

fn vals(da: &DataArray) -> Vec<f64> {
    da.data::<f64>().unwrap().iter().copied().collect()
}

fn same(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g.is_nan() && w.is_nan()) || g == w, "{got:?} vs {want:?}");
    }
}

fn g() -> DataArray {
    DataArray::from_array(["y", "x"], array![[1i32, 2, 3], [4, 5, 6]].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![10i64, 20, 30])
        .unwrap()
}

fn cond(x: Array1<i64>, c: Array1<bool>) -> DataArray {
    DataArray::from_array(["x"], c.into_dyn())
        .unwrap()
        .with_dim_coord("x", x)
        .unwrap()
}

#[test]
fn where_and_fill() {
    let c = cond(array![10, 20, 30], array![true, false, true]);
    same(
        &vals(&g().where_(&c).unwrap()),
        &[1.0, NAN, 3.0, 4.0, NAN, 6.0],
    );
    same(
        &vals(&g().where_or(&c, -1.0).unwrap()),
        &[1.0, -1.0, 3.0, 4.0, -1.0, 6.0],
    );
    // The condition is aligned first (inner join).
    let c2 = cond(array![20, 30], array![true, false]);
    let w = g().where_(&c2).unwrap();
    same(&vals(&w), &[2.0, NAN, 5.0, NAN]);
    assert_eq!(
        w.coord("x")
            .unwrap()
            .data::<i64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [20, 30]
    );
    // Times become NaT.
    let t: Array1<NaiveDateTime> = vec!["2000-01-01T00:00:00".parse().unwrap(); 3].into();
    let td = DataArray::from_array(["x"], t.into_dyn())
        .unwrap()
        .with_dim_coord("x", array![10i64, 20, 30])
        .unwrap();
    let tw = td.where_(&c).unwrap();
    assert!(tw.data::<NaiveDateTime>().unwrap()[[1]].is_nat());
    assert_eq!(
        tw.isnull()
            .unwrap()
            .data::<bool>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [false, true, false]
    );
    // Conditions with extra dimensions, and strings, are errors.
    let extra = DataArray::from_array(["z"], array![true].into_dyn()).unwrap();
    assert!(g().where_(&extra).is_err());

    let f = DataArray::from_array(
        ["y", "x"],
        array![[1.0, NAN, 3.0], [NAN, NAN, NAN]].into_dyn(),
    )
    .unwrap();
    same(
        &vals(&f.fillna(0.0).unwrap()),
        &[1.0, 0.0, 3.0, 0.0, 0.0, 0.0],
    );
    assert_eq!(
        f.notnull()
            .unwrap()
            .data::<bool>()
            .unwrap()
            .iter()
            .filter(|b| **b)
            .count(),
        2
    );
}

#[test]
fn dropna_and_reindex() {
    let f = DataArray::from_array(
        ["y", "x"],
        array![[1.0, NAN, 3.0], [NAN, NAN, NAN]].into_dyn(),
    )
    .unwrap();
    assert_eq!(f.dropna("y", How::Any).unwrap().shape(), [0, 3]);
    same(&vals(&f.dropna("y", How::All).unwrap()), &[1.0, NAN, 3.0]);
    assert_eq!(f.dropna("x", How::Any).unwrap().shape(), [2, 0]);
    same(
        &vals(&f.dropna("x", How::All).unwrap()),
        &[1.0, 3.0, NAN, NAN],
    );

    let labels = Variable::new(["x"], array![30i64, 15, 10].into_dyn()).unwrap();
    let r = g().reindex("x", &labels).unwrap();
    same(&vals(&r), &[3.0, NAN, 1.0, 6.0, NAN, 4.0]);
    let like = cond(array![30, 10], array![true, true]);
    // No label is missing, so the integers stay integers.
    let r = g().reindex_like(&like).unwrap();
    assert_eq!(
        r.data::<i32>().unwrap().iter().copied().collect::<Vec<_>>(),
        [3, 1, 6, 4]
    );

    let ds = Dataset::new()
        .with_var(
            "a",
            Variable::new(["x"], array![1.0, NAN, 3.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var(
            "b",
            Variable::new(["x"], array![NAN, 2.0, 3.0].into_dyn()).unwrap(),
        )
        .unwrap();
    assert_eq!(ds.dropna("x", How::Any).unwrap().sizes()["x"], 1);
    assert_eq!(ds.dropna("x", How::All).unwrap().sizes()["x"], 3);
    same(
        &ds.fillna(9.0).unwrap().data_vars()["b"]
            .data::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        &[9.0, 2.0, 3.0],
    );
}
