//! ffill, bfill, interpolate_na (expected values from scipy and xarray, except the documented
//! deviations).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::Unit;
use annoray::{CoreError, DataArray, Dataset, FillMethod, InterpNaOptions, Label, Variable};
use ndarray::array;

const NAN: f64 = f64::NAN;

fn vals(d: &DataArray) -> Vec<f64> {
    d.to_f64().unwrap().iter().copied().collect()
}

fn close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        let ok = (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-9 * w.abs().max(1.0);
        assert!(ok, "{got:?} vs {want:?}");
    }
}

fn unit(s: &str) -> Unit {
    s.parse().unwrap()
}

const X: [f64; 10] = [0.0, 1.0, 2.5, 3.0, 4.5, 5.0, 7.0, 8.0, 9.5, 10.0];
const Y: [f64; 10] = [NAN, 1.0, NAN, 4.0, 0.5, NAN, NAN, 3.0, 2.0, NAN];

fn sample() -> DataArray {
    DataArray::from_array(["x"], ndarray::Array1::from(Y.to_vec()).into_dyn())
        .unwrap()
        .with_dim_coord("x", ndarray::Array1::from(X.to_vec()))
        .unwrap()
}

/// `Y` with the NaN positions (x = 0, 2.5, 5, 7, 10) replaced by `at`.
fn with(at: [f64; 5]) -> Vec<f64> {
    let mut out = Y.to_vec();
    for (i, v) in [0, 2, 5, 6, 9].into_iter().zip(at) {
        out[i] = v;
    }
    out
}

#[test]
fn methods_match_scipy() {
    let cases = [
        (
            FillMethod::Linear,
            [
                -0.5,
                3.25,
                0.8571428571428572,
                2.2857142857142856,
                1.6666666666666667,
            ],
        ),
        (
            FillMethod::Slinear,
            [
                -0.5,
                3.25,
                0.8571428571428571,
                2.2857142857142856,
                1.6666666666666665,
            ],
        ),
        (FillMethod::Nearest, [1.0, 4.0, 0.5, 3.0, 2.0]),
        (FillMethod::Zero, [1.0, 1.0, 0.5, 0.5, 2.0]),
        (
            FillMethod::Cubic,
            [
                -10.163441005541292,
                4.780064873631572,
                0.046046763076091665,
                1.84157769067892,
                0.3920169392260222,
            ],
        ),
        (
            FillMethod::Pchip,
            [
                -4.303571428571429,
                3.8772321428571437,
                0.6384839650145773,
                2.5043731778425657,
                1.4465608465608466,
            ],
        ),
        (
            FillMethod::Akima,
            [
                -3.2108564013840835,
                4.030303849480969,
                0.6188290353070716,
                2.6058571567422932,
                1.1486032883882342,
            ],
        ),
        (
            FillMethod::Makima,
            [
                -1.0140172829581993,
                4.053337268890675,
                0.6284740658837842,
                2.54326311616365,
                1.4819017057284207,
            ],
        ),
    ];
    for (m, want) in cases {
        let all = sample()
            .interpolate_na("x", &InterpNaOptions::new(m).extrapolate(true))
            .unwrap();
        close(&vals(&all), &with(want));
        // Without extrapolation the leading and trailing NaN stay.
        let inner = sample()
            .interpolate_na("x", &InterpNaOptions::new(m))
            .unwrap();
        close(&vals(&inner), &with([NAN, want[1], want[2], want[3], NAN]));
    }
}

#[test]
fn limit_and_max_gap() {
    let o = InterpNaOptions::new(FillMethod::Linear);
    let l = sample().interpolate_na("x", &o.clone().limit(1)).unwrap();
    close(&vals(&l), &with([NAN, 3.25, 0.8571428571428572, NAN, NAN]));
    // Gaps: 1 (leading, from x[0]), 2, 3.5, 0.5 (trailing, to x[-1]).
    let g = sample()
        .interpolate_na("x", &o.clone().max_gap(Label::Float(2.0)))
        .unwrap();
    close(&vals(&g), &with([NAN, 3.25, NAN, NAN, NAN]));
    let ge = sample()
        .interpolate_na("x", &o.clone().max_gap(Label::Float(2.0)).extrapolate(true))
        .unwrap();
    close(
        &vals(&ge),
        &with([-0.5, 3.25, NAN, NAN, 1.6666666666666667]),
    );
    assert!(sample().interpolate_na("x", &o.clone().limit(0)).is_err());
    // Positions instead of the coordinate.
    let p = sample()
        .interpolate_na("x", &o.clone().use_positions().max_gap(Label::Int(2)))
        .unwrap();
    close(&vals(&p), &with([NAN, 2.5, NAN, NAN, NAN]));
}

#[test]
fn max_gap_in_units_and_times() {
    let km = sample()
        .with_unit(unit("degC"))
        .with_coord(
            "x",
            Variable::new(["x"], ndarray::Array1::from(X.to_vec()).into_dyn())
                .unwrap()
                .with_unit(unit("km")),
        )
        .unwrap();
    let o = InterpNaOptions::new(FillMethod::Linear);
    let g = km
        .interpolate_na(
            "x",
            &o.clone().max_gap(Label::quantity(2500.0, "m").unwrap()),
        )
        .unwrap();
    close(&vals(&g), &with([NAN, 3.25, NAN, NAN, NAN]));
    assert_eq!(g.unit(), Some(&unit("degC")));
    assert!(
        km.interpolate_na("x", &o.clone().max_gap(Label::quantity(1.0, "s").unwrap()))
            .is_err()
    );

    // Daily data with a two-day and a three-day hole.
    let days = [
        "2000-01-01",
        "2000-01-02",
        "2000-01-03",
        "2000-01-04",
        "2000-01-05",
        "2000-01-06",
        "2000-01-07",
    ]
    .map(|s| s.parse::<annoray::Instant>().unwrap());
    let t = DataArray::from_array(["t"], array![0.0, NAN, 2.0, NAN, NAN, 5.0, 6.0].into_dyn())
        .unwrap()
        .with_dim_coord("t", ndarray::Array1::from(days.to_vec()))
        .unwrap();
    for gap in [
        Label::TimeDelta(annoray::Duration::from_hours(48).unwrap()),
        Label::Str("2 d".into()),
        Label::quantity(48.0, "h").unwrap(),
    ] {
        let f = t.interpolate_na("t", &o.clone().max_gap(gap)).unwrap();
        close(&vals(&f), &[0.0, 1.0, 2.0, NAN, NAN, 5.0, 6.0]);
    }
    let all = t
        .interpolate_na("t", &o.clone().max_gap(Label::Str("3 d".into())))
        .unwrap();
    close(&vals(&all), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(
        t.interpolate_na("t", &o.clone().max_gap(Label::Float(2.0)))
            .is_err()
    );
}

#[test]
fn coordinates_and_types() {
    let o = InterpNaOptions::new(FillMethod::Linear);
    // Not increasing.
    let rev = DataArray::from_array(["x"], array![1.0, NAN, 3.0].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![2.0, 1.0, 0.0])
        .unwrap();
    assert!(matches!(
        rev.interpolate_na("x", &o),
        Err(CoreError::InvalidReduction(_))
    ));
    // Another coordinate along the dimension.
    let other = rev
        .with_coord(
            "pos",
            Variable::new(["x"], array![0.0, 3.0, 4.0].into_dyn()).unwrap(),
        )
        .unwrap();
    close(
        &vals(
            &other
                .interpolate_na("x", &o.clone().use_coordinate("pos"))
                .unwrap(),
        ),
        &[1.0, 2.5, 3.0],
    );
    // No coordinate: positions.
    let bare =
        DataArray::from_array(["x"], array![1.0f32, f32::NAN, f32::NAN, 4.0].into_dyn()).unwrap();
    let f = bare.interpolate_na("x", &o).unwrap();
    assert_eq!(f.variable().dtype(), annoray::DType::F32);
    close(&vals(&f), &[1.0, 2.0, 3.0, 4.0]);
    // Integers have nothing to fill; times are refused.
    let ints = DataArray::from_array(["x"], array![1i32, 2].into_dyn()).unwrap();
    assert_eq!(
        ints.interpolate_na("x", &o).unwrap().variable().dtype(),
        annoray::DType::I32
    );
    // Cubic needs four valid points: otherwise left as is.
    let three = DataArray::from_array(["x"], array![1.0, NAN, 3.0, 4.0].into_dyn()).unwrap();
    close(
        &vals(
            &three
                .interpolate_na("x", &InterpNaOptions::new(FillMethod::Cubic))
                .unwrap(),
        ),
        &[1.0, NAN, 3.0, 4.0],
    );
}

#[test]
fn forward_and_backward_fill() {
    let a = DataArray::from_array(
        ["y", "x"],
        array![
            [NAN, 1.0, NAN, NAN, NAN, 5.0, NAN],
            [2.0, NAN, NAN, 3.0, NAN, NAN, NAN]
        ]
        .into_dyn(),
    )
    .unwrap()
    .with_unit(unit("m"));
    let f = a.ffill("x", None).unwrap();
    close(
        &vals(&f),
        &[
            NAN, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
        ],
    );
    assert_eq!(f.unit(), Some(&unit("m")));
    close(
        &vals(&a.ffill("x", Some(2)).unwrap()),
        &[
            NAN, 1.0, 1.0, 1.0, NAN, 5.0, 5.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, NAN,
        ],
    );
    close(
        &vals(&a.bfill("x", Some(1)).unwrap()),
        &[
            1.0, 1.0, NAN, NAN, 5.0, 5.0, NAN, 2.0, NAN, 3.0, 3.0, NAN, NAN, NAN,
        ],
    );
    assert!(a.ffill("x", Some(0)).is_err());
    assert!(a.ffill("z", None).is_err());
    // Times fill NaT.
    let d = |s: &str| s.parse::<annoray::Instant>().unwrap();
    let nat = annoray::Instant::NAT;
    let t = DataArray::from_array(
        ["x"],
        array![d("2000-01-01"), nat, d("2000-01-03")].into_dyn(),
    )
    .unwrap();
    let tf = t.ffill("x", None).unwrap();
    assert_eq!(tf.data::<annoray::Instant>().unwrap()[[1]], d("2000-01-01"));
    assert!(t.interpolate_na("x", &InterpNaOptions::default()).is_err());
}

#[test]
fn datasets() {
    let mut ds = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![0.0, 1.0, 2.0].into_dyn()).unwrap(),
        )
        .unwrap();
    ds.add_var(
        "a",
        Variable::new(["x"], array![1.0, NAN, 3.0].into_dyn()).unwrap(),
    )
    .unwrap();
    ds.add_var("b", Variable::new(["y"], array![NAN].into_dyn()).unwrap())
        .unwrap();
    let i = ds.interpolate_na("x", &InterpNaOptions::default()).unwrap();
    close(&vals(&i.get("a").unwrap().to_owned()), &[1.0, 2.0, 3.0]);
    close(&vals(&i.get("b").unwrap().to_owned()), &[NAN]);
    close(
        &vals(&ds.ffill("x", None).unwrap().get("a").unwrap().to_owned()),
        &[1.0, 1.0, 3.0],
    );
    close(
        &vals(&ds.bfill("x", None).unwrap().get("a").unwrap().to_owned()),
        &[1.0, 3.0, 3.0],
    );
}

#[test]
fn lazy_data_streams_in_slabs() {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let ds = annoray::io::open_dataset(&path).unwrap();
    let da = ds.get("temp").unwrap().to_owned();
    let dim = da.dims()[1].to_string();
    let mut all = annoray::io::open_dataset(&path).unwrap();
    all.load().unwrap();
    let loaded = all.get("temp").unwrap().to_owned();
    let before = annoray::stream_bytes();
    annoray::set_stream_bytes(1);
    let o = InterpNaOptions::new(FillMethod::Pchip);
    let pairs = [
        (
            da.ffill(&dim, None).unwrap(),
            loaded.ffill(&dim, None).unwrap(),
        ),
        (
            da.interpolate_na(&dim, &o).unwrap(),
            loaded.interpolate_na(&dim, &o).unwrap(),
        ),
    ];
    annoray::set_stream_bytes(before);
    for (lazy, eager) in pairs {
        close(&vals(&lazy), &vals(&eager));
    }
    assert!(da.variable().is_lazy());
}
