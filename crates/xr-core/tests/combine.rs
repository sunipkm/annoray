//! concat and merge against Python xarray (`fixtures/combine.txt`, from
//! `fixtures/gen_combine.py`), under both xarray's new and legacy defaults.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use ndarray::{ArrayD, IxDyn, array};
use xr_core::xr_units::Unit;
use xr_core::{
    ConcatOptions, CoreError, DataArray, Dataset, MergeOptions, Name, TypedArray, Variable, View,
    concat, concat_datasets, merge,
};

const NAN: f64 = f64::NAN;

fn v<T: xr_core::Element>(dims: &[&str], shape: &[usize], values: Vec<T>) -> Variable {
    Variable::new(
        dims.iter().copied(),
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap(),
    )
    .unwrap()
}

fn ds(time: Option<Vec<f64>>, x: [f64; 3], temp: Variable, konst: [f64; 3]) -> Dataset {
    let mut d = Dataset::new()
        .with_coord("x", v(&["x"], &[3], x.to_vec()))
        .unwrap();
    if let Some(t) = time {
        d.add_coord("time", v(&["time"], &[t.len()], t)).unwrap();
    }
    d.with_var("temp", temp)
        .unwrap()
        .with_var("const", v(&["x"], &[3], konst.to_vec()))
        .unwrap()
}

const X: [f64; 3] = [10.0, 20.0, 30.0];
const K: [f64; 3] = [1.0, 2.0, 3.0];

fn a() -> Dataset {
    ds(
        Some(vec![1.0, 2.0]),
        X,
        v(&["time", "x"], &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        K,
    )
}

fn b(x: [f64; 3], konst: [f64; 3]) -> Dataset {
    ds(
        Some(vec![3.0]),
        x,
        v(&["time", "x"], &[1, 3], vec![7.0, 8.0, 9.0]),
        konst,
    )
}

fn concat_case(name: &str) -> (Vec<Dataset>, &'static str) {
    match name {
        "time" => (vec![a(), b(X, K)], "time"),
        "const_differs" => (vec![a(), b(X, [9.0; 3])], "time"),
        "misaligned" => (vec![a(), b([20.0, 30.0, 40.0], K)], "time"),
        "dtype" => (
            vec![
                a(),
                ds(
                    Some(vec![3.0]),
                    X,
                    v(&["time", "x"], &[1, 3], vec![7i32, 8, 9]),
                    K,
                ),
            ],
            "time",
        ),
        "new_dim" => (
            vec![
                ds(None, X, v(&["x"], &[3], vec![1.0, 2.0, 3.0]), K),
                ds(None, X, v(&["x"], &[3], vec![4.0, 5.0, 6.0]), K),
            ],
            "run",
        ),
        _ => panic!("unknown case {name}"),
    }
}

fn mds(vars: &[(&str, Vec<f64>)], x: Vec<f64>) -> Dataset {
    let n = x.len();
    let mut d = Dataset::new().with_coord("x", v(&["x"], &[n], x)).unwrap();
    for (k, vals) in vars {
        d.add_var(*k, v(&["x"], &[n], vals.clone())).unwrap();
    }
    d
}

fn merge_case(name: &str) -> Vec<Dataset> {
    let x = vec![1.0, 2.0, 3.0];
    let m1 = mds(&[("a", vec![1.0, NAN, 3.0])], x.clone());
    match name {
        "fill" => vec![
            m1,
            mds(&[("a", vec![NAN, 2.0, 3.0]), ("b", vec![4.0, 5.0, 6.0])], x),
        ],
        "conflict" => vec![m1, mds(&[("a", vec![9.0, 2.0, 3.0])], x)],
        "misaligned" => vec![m1, mds(&[("c", vec![7.0, 8.0])], vec![2.0, 4.0])],
        _ => panic!("unknown case {name}"),
    }
}

fn values(t: &TypedArray<View<'_>>) -> String {
    match t {
        TypedArray::F64(a) => a
            .iter()
            .map(|v| {
                if v.is_nan() {
                    "nan".into()
                } else {
                    format!("{v:?}")
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => panic!("unexpected type"),
    }
}

/// The dataset in the generator's format.
fn describe(d: &Dataset) -> Vec<String> {
    let mut out = Vec::new();
    for (kind, map) in [("coord", d.coords()), ("var", d.data_vars())] {
        let mut names: Vec<&Name> = map.keys().collect();
        names.sort();
        for n in names {
            let var = &map[n];
            let dims: Vec<&str> = var.dims().iter().map(Name::as_str).collect();
            out.push(format!(
                "  {n}|{kind}|{}|float64|{}",
                dims.join(","),
                values(&var.typed().unwrap())
            ));
        }
    }
    out
}

#[test]
fn matches_xarray() {
    let table = include_str!("../../../fixtures/combine.txt");
    let mut lines = table.lines().peekable();
    let mut checked = 0;
    while let Some(head) = lines.next() {
        let f: Vec<&str> = head.split('|').collect();
        let (case, mode) = (f[0], f[1]);
        let mut expected = Vec::new();
        while let Some(l) = lines.peek().filter(|l| l.starts_with("  ")) {
            expected.push((*l).to_owned());
            lines.next();
        }
        let legacy = mode == "legacy";
        let got = if let Some(name) = case.strip_prefix("concat_") {
            let (objs, dim) = concat_case(name);
            let refs: Vec<&Dataset> = objs.iter().collect();
            let opts = if legacy {
                ConcatOptions::legacy()
            } else {
                ConcatOptions::default()
            };
            concat_datasets(&refs, dim, &opts)
        } else {
            let objs = merge_case(case.strip_prefix("merge_").unwrap());
            let refs: Vec<&Dataset> = objs.iter().collect();
            let opts = if legacy {
                MergeOptions::legacy()
            } else {
                MergeOptions::default()
            };
            merge(&refs, &opts)
        };
        checked += 1;
        if f.get(2) == Some(&"error") {
            assert!(got.is_err(), "{head}: expected an error, got {got:?}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{head}: {e}"));
        assert_eq!(describe(&got), expected, "{head}");
    }
    assert_eq!(checked, 16);
}

#[test]
fn data_arrays_units_and_errors() {
    let da = |vals: Vec<f64>, x: Vec<f64>| {
        DataArray::from_array(
            ["x"],
            ArrayD::from_shape_vec(IxDyn(&[vals.len()]), vals).unwrap(),
        )
        .unwrap()
        .with_name("h")
        .with_dim_coord("x", ndarray::Array1::from(x))
        .unwrap()
    };
    let m = Unit::parse("m").unwrap();
    let km = Unit::parse("km").unwrap();
    let p = da(vec![1.0, 2.0], vec![1.0, 2.0]).with_unit(m.clone());
    let q = da(vec![0.003], vec![3.0]).with_unit(km);
    let r = concat(&[&p, &q], "x", &ConcatOptions::default()).unwrap();
    // Converted to the first array's unit.
    assert_eq!(r.unit(), Some(&m));
    assert_eq!(
        r.data::<f64>().unwrap().iter().copied().collect::<Vec<_>>(),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(
        r.coord("x")
            .unwrap()
            .data::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(r.name().map(Name::as_str), Some("h"));

    // A unit in one array and none in another.
    let bare = da(vec![3.0], vec![3.0]);
    assert!(matches!(
        concat(&[&p, &bare], "x", &ConcatOptions::default()),
        Err(CoreError::CannotCombine { .. })
    ));
    // Incompatible units.
    let s = da(vec![3.0], vec![3.0]).with_unit(Unit::parse("s").unwrap());
    assert!(concat(&[&p, &s], "x", &ConcatOptions::default()).is_err());
    // Strings and numbers do not mix.
    let t = DataArray::from_array(["x"], array!["a".to_owned()].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![3.0])
        .unwrap();
    assert!(
        concat(
            &[&p.with_unit(Unit::one()), &t],
            "x",
            &ConcatOptions::default()
        )
        .is_err()
    );
}
