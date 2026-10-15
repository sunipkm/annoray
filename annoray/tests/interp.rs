//! interp against Python xarray (`fixtures/interp.txt`, from `fixtures/gen_interp.py`).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use annoray::units::Unit;
use annoray::{DataArray, Dataset, Extrapolate, InterpMethod, InterpOptions, TypedArray, Variable};
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
    let table = include_str!("../../fixtures/interp.txt");
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

// ----- logarithmic methods -----

fn close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        let ok = (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * w.abs().max(1.0);
        assert!(ok, "got {got:?}, want {want:?}");
    }
}

#[test]
fn log_log_is_exact_for_power_laws() {
    // y = x^2 on a coarse grid: log-log interpolation reproduces it between points.
    let a = da(
        array![1.0, 100.0, 10000.0].into_dyn(),
        array![1.0, 10.0, 100.0],
    );
    let out = a
        .interp(
            "x",
            &target(&[3.0, 50.0, 10.0, 0.5, 200.0]),
            InterpMethod::LogLog,
        )
        .unwrap();
    close(&values(&out), &[9.0, 2500.0, 100.0, NAN, NAN]);
    // Linear interpolation is far off.
    let lin = a
        .interp("x", &target(&[3.0]), InterpMethod::Linear)
        .unwrap();
    assert!((values(&lin)[0] - 23.0).abs() < 1e-9);
}

#[test]
fn log_value_is_exact_for_exponentials() {
    // A scale height of 10: n = exp(-z / 10).
    let z = array![0.0, 10.0, 30.0];
    let a = da(z.mapv(|z: f64| (-z / 10.0).exp()).into_dyn(), z);
    let out = a
        .interp("x", &target(&[5.0, 20.0]), InterpMethod::LogValue)
        .unwrap();
    close(&values(&out), &[(-0.5f64).exp(), (-2.0f64).exp()]);
}

#[test]
fn log_coord_is_linear_in_the_logarithm() {
    let a = da(array![0.0, 2.0].into_dyn(), array![1.0, 100.0]);
    let out = a
        .interp("x", &target(&[10.0]), InterpMethod::LogCoord)
        .unwrap();
    close(&values(&out), &[1.0]);
}

#[test]
fn log_values_need_positive_neighbours() {
    let a = da(array![0.0, 4.0, -1.0].into_dyn(), array![1.0, 2.0, 3.0]);
    let out = a
        .interp("x", &target(&[1.0, 1.5, 2.0, 2.5]), InterpMethod::LogValue)
        .unwrap();
    // Exact hits keep their value (even 0); between a non-positive value and another, NaN.
    close(&values(&out), &[0.0, NAN, 4.0, NAN]);
}

#[test]
fn log_methods_match_glow_formula() {
    // glow-rs' `loglog_interp`: exp(ln y0 + ln(y1/y0) ln(x/x0) / ln(x1/x0)).
    let (x0, x1, y0, y1, x): (f64, f64, f64, f64, f64) = (5.0, 40.0, 2.0e-16, 3.5e-17, 17.0);
    let want = (y0.ln() + (y1 / y0).ln() * ((x / x0).ln() / (x1 / x0).ln())).exp();
    let a = da(array![y0, y1].into_dyn(), array![x0, x1]);
    let out = a.interp("x", &target(&[x]), InterpMethod::LogLog).unwrap();
    close(&values(&out), &[want]);
}

#[test]
fn log_methods_reject_undefined_inputs() {
    let a = da(array![1.0, 2.0].into_dyn(), array![0.0, 1.0]);
    assert!(
        a.interp("x", &target(&[0.5]), InterpMethod::LogCoord)
            .is_err()
    );
    let c = da(array![10.0, 20.0].into_dyn(), array![1.0, 2.0])
        .with_unit("degC".parse::<Unit>().unwrap());
    assert!(
        c.interp("x", &target(&[1.5]), InterpMethod::LogValue)
            .is_err()
    );
    // Linear interpolation of the same °C data is fine.
    assert!(c.interp("x", &target(&[1.5]), InterpMethod::Linear).is_ok());
}

// ----- extrapolation -----

fn at(a: &DataArray, t: &[f64], opts: InterpOptions) -> Vec<f64> {
    values(&a.interp("x", &target(t), opts).unwrap())
}

#[test]
fn extrapolation_options() {
    let a = da(array![0.0, 10.0, 20.0].into_dyn(), array![0.0, 1.0, 2.0]);
    let linear = InterpOptions::new(InterpMethod::Linear);
    close(&at(&a, &[-1.0, 3.0, NAN], linear), &[NAN, NAN, NAN]);
    let extend = linear.extrapolate(Extrapolate::Extend);
    close(
        &at(&a, &[-1.0, 0.5, 3.0, NAN], extend),
        &[-10.0, 5.0, 30.0, NAN],
    );
    let edge = linear.extrapolate(Extrapolate::Edge);
    close(&at(&a, &[-1.0, 3.0], edge), &[0.0, 20.0]);
    let mixed = linear.below(Extrapolate::Edge).above(Extrapolate::Extend);
    close(&at(&a, &[-1.0, 3.0], mixed), &[0.0, 30.0]);
    // Nearest only ever takes edge values.
    let nearest = InterpOptions::new(InterpMethod::Nearest).extrapolate(Extrapolate::Extend);
    close(&at(&a, &[-5.0, 9.0], nearest), &[0.0, 20.0]);
}

#[test]
fn extrapolation_follows_the_method_law() {
    // y = x^2: the power law continues beyond the table on both sides.
    let p = da(array![1.0, 100.0].into_dyn(), array![1.0, 10.0]);
    let loglog = InterpOptions::new(InterpMethod::LogLog).extrapolate(Extrapolate::Extend);
    close(&at(&p, &[100.0, 0.1], loglog), &[10000.0, 0.01]);
    // n = exp(-z / 10) continues above the last level.
    let z = array![0.0, 10.0];
    let e = da(z.mapv(|z: f64| (-z / 10.0).exp()).into_dyn(), z);
    let logv = InterpOptions::new(InterpMethod::LogValue).above(Extrapolate::Extend);
    close(&at(&e, &[30.0], logv), &[(-3.0f64).exp()]);
    // Log coordinates: a non-positive target stays NaN even when extending.
    let c = da(array![0.0, 2.0].into_dyn(), array![1.0, 100.0]);
    let logc = InterpOptions::new(InterpMethod::LogCoord).extrapolate(Extrapolate::Extend);
    close(&at(&c, &[1000.0, 0.1, -1.0], logc), &[3.0, -1.0, NAN]);
}

#[test]
fn extrapolation_with_descending_and_single_coordinates() {
    let d = da(array![5.0, 3.0, 1.0].into_dyn(), array![30.0, 20.0, 10.0]);
    let extend = InterpOptions::new(InterpMethod::Linear).extrapolate(Extrapolate::Extend);
    close(&at(&d, &[40.0, 0.0], extend), &[7.0, -1.0]);
    let one = da(array![4.0].into_dyn(), array![1.0]);
    close(&at(&one, &[0.0, 2.0], extend), &[4.0, 4.0]);
}

#[test]
fn glow_style_table_lookup() {
    // Hold the first value below the table, continue the power law above it.
    let e = array![10.0, 100.0, 1000.0];
    let sigma = da(e.mapv(|e: f64| 1e-16 * (e / 10.0).powf(-0.8)).into_dyn(), e);
    let opts = InterpOptions::new(InterpMethod::LogLog)
        .below(Extrapolate::Edge)
        .above(Extrapolate::Extend);
    let got = at(&sigma, &[1.0, 30.0, 5000.0], opts);
    close(
        &got,
        &[1e-16, 1e-16 * 3f64.powf(-0.8), 1e-16 * 500f64.powf(-0.8)],
    );
}

#[test]
fn power_law_extrapolation_reproduces_glow() {
    // glow-rs' elastic tables: above the last energy, sigma_last * (e_last / e)^n with a
    // tabulated n unrelated to the table's last slope; below the first, the first value.
    let e = array![10.0, 100.0, 1000.0];
    let sigma = da(array![3.0e-16, 2.0e-16, 9.0e-17].into_dyn(), e);
    let n = 1.7;
    let opts = InterpOptions::new(InterpMethod::LogLog)
        .below(Extrapolate::Edge)
        .above(Extrapolate::PowerLaw(-n));
    let got = at(&sigma, &[2000.0, 5.0, 1000.0], opts);
    close(
        &got,
        &[9.0e-17 * (1000.0f64 / 2000.0).powf(n), 3.0e-16, 9.0e-17],
    );
    // Below, from the first point: y_first * (x / x_first)^k.
    let below = InterpOptions::new(InterpMethod::Linear).below(Extrapolate::PowerLaw(2.0));
    close(&at(&sigma, &[5.0, 0.0], below), &[3.0e-16 * 0.25, NAN]);
}

#[test]
fn power_law_needs_ratio_scales() {
    let opts = InterpOptions::new(InterpMethod::Linear).above(Extrapolate::PowerLaw(1.0));
    let c = da(array![10.0, 20.0].into_dyn(), array![1.0, 2.0])
        .with_unit("degC".parse::<Unit>().unwrap());
    assert!(c.interp("x", &target(&[3.0]), opts).is_err());
    let times = DataArray::from_array(["x"], array![1.0, 2.0].into_dyn())
        .unwrap()
        .with_dim_coord(
            "x",
            array![
                "2020-01-01T00:00:00".parse::<NaiveDateTime>().unwrap(),
                "2020-01-02T00:00:00".parse::<NaiveDateTime>().unwrap()
            ],
        )
        .unwrap();
    let at_t = Variable::new(
        ["x"],
        array!["2020-01-03T00:00:00".parse::<NaiveDateTime>().unwrap()].into_dyn(),
    )
    .unwrap();
    assert!(times.interp("x", &at_t, opts).is_err());
}
