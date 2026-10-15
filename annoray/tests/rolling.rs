//! rolling against Python xarray (`fixtures/rolling.txt`, from `fixtures/gen_rolling.py`).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use annoray::units::Unit;
use annoray::{CoreError, DataArray, Dataset, Reduction, RollingOptions, TypedArray, Variable};
use ndarray::{ArrayD, IxDyn, array};

const NAN: f64 = f64::NAN;

fn input(name: &str) -> DataArray {
    match name {
        "f" => DataArray::from_array(
            ["x"],
            array![1.0, 2.0, NAN, 4.0, 5.0, 6.0, NAN, NAN, 9.0].into_dyn(),
        )
        .unwrap(),
        "i" => DataArray::from_array(["x"], array![3i32, 1, 4, 1, 5, 9, 2].into_dyn()).unwrap(),
        "g" => DataArray::from_array(
            ["y", "x"],
            ArrayD::from_shape_vec(IxDyn(&[3, 4]), (0..12).map(f64::from).collect()).unwrap(),
        )
        .unwrap(),
        _ => panic!("unknown input {name}"),
    }
}

fn op(name: &str) -> Reduction {
    match name {
        "sum" => Reduction::Sum,
        "mean" => Reduction::Mean,
        "median" => Reduction::Median,
        "std" => Reduction::Std,
        "var" => Reduction::Var,
        "min" => Reduction::Min,
        "max" => Reduction::Max,
        "count" => Reduction::Count,
        "prod" => Reduction::Prod,
        "argmin" => Reduction::ArgMin,
        "argmax" => Reduction::ArgMax,
        _ => panic!("unknown op {name}"),
    }
}

fn floats(da: &DataArray) -> Vec<f64> {
    match da.variable().typed().unwrap() {
        TypedArray::F64(a) => a.iter().copied().collect(),
        TypedArray::F32(a) => a.iter().map(|v| f64::from(*v)).collect(),
        _ => panic!("expected floats"),
    }
}

fn close(got: &[f64], want: &str) -> bool {
    let want: Vec<f64> = want.split(' ').map(|w| w.parse().unwrap()).collect();
    got.len() == want.len()
        && got
            .iter()
            .zip(&want)
            .all(|(g, w)| (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * w.abs().max(1.0))
}

#[test]
fn matches_xarray() {
    let table = include_str!("../../fixtures/rolling.txt");
    let mut checked = 0;
    for line in table.lines() {
        let f: Vec<&str> = line.split('|').collect();
        let da = input(f[0]);
        let window: usize = f[2].parse().unwrap();
        let mut opts = RollingOptions::default().center(f[3] == "True");
        if f[4] != "None" {
            opts = opts.min_periods(f[4].parse().unwrap());
        }
        let r = da.rolling("x", window, &opts).unwrap();
        let out = if f[1] == "construct" {
            r.construct("w").unwrap()
        } else {
            r.reduce(op(f[1])).unwrap_or_else(|e| panic!("{line}: {e}"))
        };
        let shape: Vec<String> = out.shape().iter().map(usize::to_string).collect();
        assert_eq!(shape.join(","), f[5], "{line}");
        let got = floats(&out);
        assert!(close(&got, f[6]), "{line}\n got {got:?}");
        checked += 1;
    }
    assert_eq!(checked, 207);
}

#[test]
fn units_coords_datasets_errors() {
    let m = Unit::parse("m").unwrap();
    let da = input("f")
        .with_unit(m.clone())
        .with_dim_coord(
            "x",
            ndarray::Array1::from((0..9).map(f64::from).collect::<Vec<_>>()),
        )
        .unwrap();
    let r = da.rolling("x", 3, &RollingOptions::default()).unwrap();
    assert_eq!(r.mean().unwrap().unit(), Some(&m));
    assert_eq!(r.var().unwrap().unit(), Some(&Unit::parse("m2").unwrap()));
    assert_eq!(r.count().unwrap().unit(), None);
    assert_eq!(r.sum().unwrap().coords().len(), 1);

    let ds = Dataset::new()
        .with_var("a", input("f").variable().to_owned())
        .unwrap()
        .with_var(
            "s",
            Variable::new(["x"], ArrayD::from_elem(IxDyn(&[9]), "z".to_owned())).unwrap(),
        )
        .unwrap();
    let out = ds
        .rolling("x", 2, &RollingOptions::default().min_periods(1))
        .unwrap()
        .sum()
        .unwrap();
    assert_eq!(out.data_vars()["a"].data::<f64>().unwrap()[[1]], 3.0);
    assert_eq!(out.data_vars()["s"].dtype(), annoray::DType::Str);

    assert!(matches!(
        da.rolling("x", 0, &RollingOptions::default()),
        Err(CoreError::InvalidReduction(_))
    ));
    assert!(
        da.rolling("x", 3, &RollingOptions::default().min_periods(4))
            .is_err()
    );
    assert!(da.rolling("nope", 3, &RollingOptions::default()).is_err());
    assert!(r.reduce(Reduction::Any).is_err());
}
