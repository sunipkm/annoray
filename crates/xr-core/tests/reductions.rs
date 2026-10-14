//! Reductions against Python xarray (`fixtures/reductions.txt`, from `fixtures/gen_reductions.py`).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use chrono::{NaiveDateTime, TimeDelta};
use ndarray::{ArrayD, IxDyn};
use xr_core::xr_units::Unit;
use xr_core::{
    AttrValue, CoreError, DType, DataArray, Dataset, Name, ReduceOptions, Reduction, TypedArray,
    Variable,
};

/// Cases where xarray-rs deliberately differs: medians of dates are supported; `any`/`all`
/// of dates and durations are not.
const DIFFERENCES: &[(&str, &str)] = &[
    ("time", "median"),
    ("time", "any"),
    ("time", "all"),
    ("delta", "any"),
    ("delta", "all"),
];

const NAN: f64 = f64::NAN;
const F: [f64; 12] = [1.0, NAN, 3.0, 4.0, NAN, NAN, NAN, NAN, 2.5, -1.0, 0.0, 7.0];

fn grid<T>(v: Vec<T>) -> ArrayD<T> {
    ArrayD::from_shape_vec(IxDyn(&[3, 4]), v).unwrap()
}

fn input(case: &str) -> Variable {
    let xy = ["x", "y"];
    match case {
        "f64" => Variable::new(xy, grid(F.to_vec())).unwrap(),
        #[allow(clippy::cast_possible_truncation)]
        "f32" => Variable::new(xy, grid(F.iter().map(|v| *v as f32).collect())).unwrap(),
        "i32" => {
            Variable::new(xy, grid(vec![1i32, -2, 3, 4, 5, 6, -7, 8, 9, 10, 11, -12])).unwrap()
        }
        "u8" => Variable::new(xy, grid(vec![1u8, 2, 3, 4, 250, 6, 7, 8, 0, 0, 0, 1])).unwrap(),
        "bool" => {
            let b = [1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1];
            Variable::new(xy, grid(b.iter().map(|v| *v == 1).collect())).unwrap()
        }
        "time" => {
            let t = [
                "2020-01-01T00:00:00",
                "2021-06-15T12:00:00",
                "1999-12-31T23:59:59.5",
                "2020-01-02T00:00:00",
            ];
            let t: Vec<NaiveDateTime> = t.iter().map(|s| s.parse().unwrap()).collect();
            Variable::new(["t"], ArrayD::from_shape_vec(IxDyn(&[4]), t).unwrap()).unwrap()
        }
        "delta" => {
            let d = [1_000_000_000i64, -5, 3_600_000_000_000, 7].map(TimeDelta::nanoseconds);
            Variable::new(
                ["t"],
                ArrayD::from_shape_vec(IxDyn(&[4]), d.to_vec()).unwrap(),
            )
            .unwrap()
        }
        _ => panic!("unknown case {case}"),
    }
}

fn op(name: &str) -> Reduction {
    use Reduction::*;
    match name {
        "sum" => Sum,
        "prod" => Prod,
        "mean" => Mean,
        "median" => Median,
        "std" => Std,
        "var" => Var,
        "min" => Min,
        "max" => Max,
        "count" => Count,
        "argmin" => ArgMin,
        "argmax" => ArgMax,
        "any" => Any,
        "all" => All,
        _ => panic!("unknown op {name}"),
    }
}

fn numpy_dtype(d: DType) -> &'static str {
    match d {
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::U64 => "uint64",
        DType::Bool => "bool",
        DType::DateTime => "datetime64[ns]",
        DType::TimeDelta => "timedelta64[ns]",
        other => panic!("unexpected dtype {other}"),
    }
}

/// Values formatted as the generator does (floats are compared numerically).
fn values(t: &TypedArray<xr_core::View<'_>>) -> Vec<String> {
    match t {
        TypedArray::F32(a) => a.iter().map(|v| f64::from(*v).to_string()).collect(),
        TypedArray::F64(a) => a.iter().map(f64::to_string).collect(),
        TypedArray::I32(a) => a.iter().map(i32::to_string).collect(),
        TypedArray::I64(a) => a.iter().map(i64::to_string).collect(),
        TypedArray::U8(a) => a.iter().map(u8::to_string).collect(),
        TypedArray::U64(a) => a.iter().map(u64::to_string).collect(),
        TypedArray::Bool(a) => a.iter().map(bool::to_string).collect(),
        TypedArray::DateTime(a) => a
            .iter()
            .map(|v| v.format("%Y-%m-%dT%H:%M:%S%.9f").to_string())
            .collect(),
        TypedArray::TimeDelta(a) => a
            .iter()
            .map(|v| v.num_nanoseconds().unwrap().to_string())
            .collect(),
        _ => panic!("unexpected result type"),
    }
}

fn close(got: &str, want: &str, tol: f64) -> bool {
    match (got.parse::<f64>(), want.parse::<f64>()) {
        (Ok(g), Ok(w)) if g.is_nan() || w.is_nan() => g.is_nan() && w.is_nan(),
        (Ok(g), Ok(w)) => (g - w).abs() <= tol * w.abs().max(1.0),
        _ => got == want,
    }
}

#[test]
fn matches_xarray() {
    let table = include_str!("../../../fixtures/reductions.txt");
    let mut checked = 0;
    for line in table.lines() {
        let f: Vec<&str> = line.splitn(7, '|').collect();
        let [case, name, dims, skipna, ddof, min_count, expected] = f[..] else {
            panic!("bad line {line}");
        };
        let mut opts = ReduceOptions::default();
        if !skipna.is_empty() {
            opts = opts.skipna(skipna == "true");
        }
        if !ddof.is_empty() {
            opts = opts.ddof(ddof.parse().unwrap());
        }
        if !min_count.is_empty() {
            opts = opts.min_count(min_count.parse().unwrap());
        }
        let v = input(case);
        let got = if dims == "all" {
            v.reduce(op(name), .., &opts)
        } else {
            let names: Vec<Name> = dims.split(',').map(Name::from).collect();
            v.reduce(op(name), names, &opts)
        };
        if DIFFERENCES.contains(&(case, name)) {
            continue;
        }
        checked += 1;
        if expected == "error" {
            assert!(got.is_err(), "{line}: expected an error, got {got:?}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{line}: {e}"));
        let mut parts = expected.splitn(3, '|');
        let (dtype, shape, want) = (
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
        );
        assert_eq!(numpy_dtype(got.dtype()), dtype, "{line}");
        let shape_str = got
            .shape()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(shape_str, shape, "{line}");
        let tol = if got.dtype() == DType::F32 {
            1e-6
        } else {
            1e-12
        };
        let got_values = values(&got.typed().unwrap());
        let want: Vec<&str> = want.split(' ').collect();
        assert_eq!(got_values.len(), want.len(), "{line}");
        for (g, w) in got_values.iter().zip(&want) {
            assert!(close(g, w, tol), "{line}: got {got_values:?}");
        }
    }
    assert!(checked > 350, "only {checked} cases checked");
}

#[test]
fn coordinates_units_and_attrs() {
    let m = Unit::parse("m").unwrap();
    let da = DataArray::new(input("f64").with_unit(m.clone()))
        .with_name("h")
        .with_attrs([("long_name", "height")])
        .with_dim_coord("x", ndarray::array![10.0, 20.0, 30.0])
        .unwrap();
    let s = da.sum("x").unwrap();
    assert_eq!(s.dims(), ["y"]);
    assert!(s.coords().is_empty(), "the reduced coordinate is dropped");
    assert_eq!(s.unit(), Some(&m));
    assert_eq!(s.name().map(Name::as_str), Some("h"));
    assert!(s.attrs().is_empty());
    let kept = da
        .reduce(
            Reduction::Mean,
            "y",
            &ReduceOptions::default().keep_attrs(true),
        )
        .unwrap();
    assert_eq!(kept.coords().len(), 1);
    assert_eq!(
        kept.attrs().get("long_name"),
        Some(&AttrValue::from("height"))
    );

    let var = da.var(..).unwrap();
    assert_eq!(var.unit(), Some(&Unit::parse("m2").unwrap()));
    assert_eq!(da.count(..).unwrap().unit(), None);

    let c = Unit::parse("degC").unwrap();
    let t = DataArray::new(input("f64").with_unit(c.clone()));
    assert!(t.sum(..).is_err(), "°C cannot be summed");
    assert_eq!(t.mean(..).unwrap().unit(), Some(&c));
    assert_eq!(t.std(..).unwrap().unit(), Some(&c.delta()));

    assert!(matches!(da.sum("z"), Err(CoreError::NoSuchDim(_))));
    assert!(matches!(da.argmin(..), Err(CoreError::InvalidReduction(_))));
}

#[test]
fn dataset_reductions() {
    let ds = Dataset::new()
        .with_var("a", input("f64"))
        .unwrap()
        .with_var("n", input("i32"))
        .unwrap()
        .with_var(
            "s",
            Variable::new(
                ["y"],
                ArrayD::from_shape_vec(
                    IxDyn(&[4]),
                    vec!["p".to_owned(), "q".into(), "r".into(), "s".into()],
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
        .with_var(
            "only_x",
            Variable::new(
                ["x"],
                ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let m = ds.mean("y").unwrap();
    // Non-numeric variables are dropped; variables without the dimension are kept.
    assert_eq!(
        m.data_vars().keys().map(Name::as_str).collect::<Vec<_>>(),
        ["a", "n", "only_x"]
    );
    assert_eq!(m.data_vars()["a"].dims(), ["x"]);
    assert_eq!(m.data_vars()["only_x"].dims(), ["x"]);
    let mx = ds.max(..).unwrap();
    assert_eq!(
        mx.data_vars()["s"]
            .data::<String>()
            .unwrap()
            .iter()
            .next()
            .map(String::as_str),
        Some("s")
    );
    assert!(ds.sum("nope").is_err());
}
