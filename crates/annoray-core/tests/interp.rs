//! interp against Python xarray (`fixtures/interp.txt`, from `fixtures/gen_interp.py`).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use annoray_core::annoray_units::Unit;
use annoray_core::{DataArray, Dataset, InterpMethod, TypedArray, Variable};
use chrono::NaiveDateTime;
use ndarray::{Array1, ArrayD, IxDyn, array};

const NAN: f64 = f64::NAN;
const TARGETS: [f64; 13] = [
    -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 5.0, 7.25, 25.0, NAN,
];

fn da(values: ArrayD<f64>, x: Array1<f64>) -> DataArray {
    DataArray::from_array(["x"], values)
        .unwrap()
        .with_dim_coord("x", x)
        .unwrap()
}

fn input(name: &str) -> DataArray {
    match name {
        "nan" => da(
            array![0.0, 10.0, NAN, 30.0, 40.0].into_dyn(),
            array![0.0, 1.0, 2.0, 3.0, 4.0],
        ),
        "uneven" => da(
            array![1.0, 4.0, 9.0, 16.0, 25.0].into_dyn(),
            array![1.0, 2.0, 3.5, 7.0, 7.5],
        ),
        "unsorted" => da(array![1.0, 2.0, 4.0].into_dyn(), array![2.0, 0.0, 1.0]),
        "desc" => da(array![5.0, 3.0, 1.0].into_dyn(), array![30.0, 20.0, 10.0]),
        "int" => DataArray::from_array(["x"], array![1i32, 3, 9].into_dyn())
            .unwrap()
            .with_dim_coord("x", array![0.0, 1.0, 2.0])
            .unwrap(),
        "grid" | "grid2" => DataArray::from_array(
            ["y", "x"],
            ArrayD::from_shape_vec(IxDyn(&[3, 4]), (0..12).map(f64::from).collect()).unwrap(),
        )
        .unwrap()
        .with_dim_coord("y", array![0.0, 1.0, 2.0])
        .unwrap()
        .with_dim_coord("x", array![0.0, 10.0, 20.0, 30.0])
        .unwrap(),
        _ => panic!("unknown {name}"),
    }
}

fn target(v: &[f64]) -> Variable {
    Variable::new(["x"], Array1::from(v.to_vec()).into_dyn()).unwrap()
}

fn values(d: &DataArray) -> Vec<f64> {
    match d.variable().typed().unwrap() {
        TypedArray::F64(a) => a.iter().copied().collect(),
        _ => panic!("expected f64"),
    }
}

fn same(got: &[f64], want: &str, line: &str) {
    let want: Vec<f64> = want.split(' ').map(|w| w.parse().unwrap()).collect();
    assert_eq!(got.len(), want.len(), "{line}");
    for (g, w) in got.iter().zip(&want) {
        let ok = (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * w.abs().max(1.0);
        assert!(ok, "{line}\n got {got:?}");
    }
}

#[test]
fn matches_xarray() {
    let table = include_str!("../../../fixtures/interp.txt");
    let dt = |s: &str| -> NaiveDateTime { s.parse().unwrap() };
    for line in table.lines() {
        let f: Vec<&str> = line.split('|').collect();
        let method = if f[1] == "nearest" {
            InterpMethod::Nearest
        } else {
            InterpMethod::Linear
        };
        let out = match f[0] {
            "grid2" => {
                let y = Variable::new(["y"], array![0.5, 1.75].into_dyn()).unwrap();
                input("grid")
                    .interp("x", &target(&[5.0, 25.0]), method)
                    .unwrap()
                    .interp("y", &y, method)
                    .unwrap()
            }
            "time" => {
                let t: Array1<NaiveDateTime> = array![
                    dt("2000-01-01T00:00:00"),
                    dt("2000-01-02T00:00:00"),
                    dt("2000-01-03T00:00:00")
                ];
                let at: Array1<NaiveDateTime> = array![
                    dt("2000-01-01T06:00:00"),
                    dt("2000-01-02T18:00:00"),
                    dt("2000-01-04T00:00:00")
                ];
                DataArray::from_array(["x"], array![0.0, 24.0, 48.0].into_dyn())
                    .unwrap()
                    .with_dim_coord("x", t)
                    .unwrap()
                    .interp("x", &Variable::new(["x"], at.into_dyn()).unwrap(), method)
                    .unwrap()
            }
            name => input(name).interp("x", &target(&TARGETS), method).unwrap(),
        };
        let shape: Vec<String> = out.shape().iter().map(usize::to_string).collect();
        assert_eq!(shape.join(","), f[3], "{line}");
        let got = values(&out);
        // xarray's `nearest` returns the last value for a NaN target; NaN is ours.
        let want = if f[1] == "nearest" && f[0] != "grid2" {
            let mut w: Vec<&str> = f[4].split(' ').collect();
            let n = TARGETS.len();
            for row in w.chunks_mut(n) {
                if let Some(last) = row.last_mut() {
                    *last = "nan";
                }
            }
            w.join(" ")
        } else {
            f[4].to_owned()
        };
        same(&got, &want, line);
    }
}

#[test]
fn coords_units_datasets() {
    let km = Unit::parse("km").unwrap();
    let m = Unit::parse("m").unwrap();
    let base = DataArray::from_array(["x"], array![0.0, 10.0, 20.0].into_dyn())
        .unwrap()
        .with_coord(
            "x",
            Variable::new(["x"], array![0.0, 1.0, 2.0].into_dyn())
                .unwrap()
                .with_unit(km),
        )
        .unwrap()
        .with_coord(
            "aux",
            Variable::new(["x"], array![5.0, 6.0, 7.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_coord(
            "lab",
            Variable::new(
                ["x"],
                array!["a".to_owned(), "b".into(), "c".into()].into_dyn(),
            )
            .unwrap(),
        )
        .unwrap();
    // Targets in metres are converted to the coordinate's kilometres.
    let at = Variable::new(["x"], array![500.0, 1500.0].into_dyn())
        .unwrap()
        .with_unit(m);
    let r = base.interp("x", &at, InterpMethod::Linear).unwrap();
    same(&values(&r), "5.0 15.0", "units");
    assert_eq!(
        r.coord("aux")
            .unwrap()
            .data::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [5.5, 6.5]
    );
    assert!(r.coord("lab").is_err(), "string coordinates are dropped");

    let ds = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![0.0, 1.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var(
            "v",
            Variable::new(["x"], array![0.0, 2.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var("c", Variable::new(["y"], array![7.0].into_dyn()).unwrap())
        .unwrap();
    let out = ds
        .interp("x", &target(&[0.25]), InterpMethod::Linear)
        .unwrap();
    assert_eq!(out.data_vars()["v"].data::<f64>().unwrap()[[0]], 0.5);
    assert_eq!(out.data_vars()["c"].dims(), ["y"]);

    // interp_like and errors.
    let like = DataArray::from_array(["x"], array![0.0].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![1.5])
        .unwrap();
    same(
        &values(
            &input("nan")
                .interp_like(&like, InterpMethod::Linear)
                .unwrap(),
        ),
        "nan",
        "like",
    );
    assert!(
        input("nan")
            .interp("y", &target(&[1.0]), InterpMethod::Linear)
            .is_err()
    );
}
