//! differentiate, integrate, cumulative_integrate, polyfit, polyval (expected values as
//! Python xarray gives them, except the documented deviations).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::Unit;
use annoray::{Covariance, DataArray, Dataset, PolyfitOptions, Variable, polyval};
use ndarray::{Array2, array};

const NAN: f64 = f64::NAN;

fn vals(d: &DataArray) -> Vec<f64> {
    d.to_f64().unwrap().iter().copied().collect()
}

fn var(ds: &Dataset, name: &str) -> Vec<f64> {
    vals(&ds.get(name).unwrap().to_owned())
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

/// `x³ − 2x` at uneven positions.
fn cubic() -> DataArray {
    let x = array![0.0, 1.0, 3.0, 4.5, 7.0];
    let y = x.mapv(|v: f64| v.powi(3) - 2.0 * v);
    DataArray::from_array(["x"], y.into_dyn())
        .unwrap()
        .with_dim_coord("x", x)
        .unwrap()
}

#[test]
fn derivatives_match_numpy_gradient() {
    let d = cubic();
    close(
        &vals(&d.differentiate("x", 1, None).unwrap()),
        &[-1.0, 3.0, 28.0, 62.5, 98.75],
    );
    close(
        &vals(&d.differentiate("x", 2, None).unwrap()),
        &[-5.0, 3.0, 28.0, 62.5, 135.0],
    );
    let even = DataArray::from_array(["t"], array![1.0, 4.0, 9.0, 16.0].into_dyn())
        .unwrap()
        .with_dim_coord("t", array![0.0, 2.0, 4.0, 6.0])
        .unwrap();
    close(
        &vals(&even.differentiate("t", 2, None).unwrap()),
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert!(d.differentiate("x", 3, None).is_err());
    assert!(
        d.isel([(
            "x",
            annoray::Indexer::Slice(annoray::PySlice::new(None, Some(2), None))
        )])
        .unwrap()
        .differentiate("x", 2, None)
        .is_err()
    );
    assert!(d.differentiate("nope", 1, None).is_err());
}

#[test]
fn integrals_by_trapezoids() {
    let d = cubic();
    close(&vals(&d.integrate(&["x"], None).unwrap()), &[610.75]);
    close(
        &vals(&d.cumulative_integrate(&["x"], None).unwrap()),
        &[0.0, -0.5, 19.5, 96.84375, 610.75],
    );
    let z = DataArray::from_array(
        ["y", "x"],
        array![[1.0, 2.0, NAN, 4.0], [0.0, 1.0, 4.0, 9.0]].into_dyn(),
    )
    .unwrap()
    .with_dim_coord("x", array![0.0, 1.0, 2.0, 4.0])
    .unwrap();
    let i = z.integrate(&["x"], None).unwrap();
    assert_eq!(i.dims(), ["y"]);
    assert!(!i.coords().contains_key("x"));
    close(&vals(&i), &[NAN, 16.0]);
    close(
        &vals(&z.cumulative_integrate(&["x"], None).unwrap()),
        &[0.0, 1.5, NAN, NAN, 0.0, 0.5, 3.0, 16.0],
    );
}

#[test]
fn calculus_carries_units() {
    let d = cubic()
        .with_unit(unit("m"))
        .with_coord(
            "x",
            Variable::new(["x"], array![0.0, 1.0, 3.0, 4.5, 7.0].into_dyn())
                .unwrap()
                .with_unit(unit("s")),
        )
        .unwrap();
    let v = d.differentiate("x", 1, None).unwrap();
    assert!(v.unit().unwrap().is_convertible(&unit("m/s")));
    let i = d.integrate(&["x"], None).unwrap();
    assert!(i.unit().unwrap().is_convertible(&unit("m s")));
    // An absolute temperature has a derivative in K per step but no integral.
    let t = cubic().with_unit(unit("degC"));
    let dt = t.differentiate("x", 1, None).unwrap();
    assert!(
        dt.unit()
            .unwrap()
            .converter_to(&unit("K"))
            .unwrap()
            .is_identity()
    );
    assert!(t.integrate(&["x"], None).is_err());
    assert!(t.cumulative_integrate(&["x"], None).is_err());
}

#[test]
fn time_coordinates_step_in_seconds() {
    let times =
        ["2000-01-01", "2000-01-02", "2000-01-04"].map(|s| s.parse::<annoray::Instant>().unwrap());
    let t = DataArray::from_array(["t"], array![0.0, 1.0, 4.0].into_dyn())
        .unwrap()
        .with_dim_coord("t", ndarray::Array1::from(times.to_vec()))
        .unwrap();
    let d = t.differentiate("t", 1, None).unwrap();
    close(
        &vals(&d),
        &[
            1.1574074074074073e-05,
            1.3503086419753087e-05,
            1.736111111111111e-05,
        ],
    );
    assert!(
        d.unit()
            .unwrap()
            .converter_to(&unit("1/s"))
            .unwrap()
            .is_identity()
    );
    let i = t.integrate(&["t"], Some(&unit("h"))).unwrap();
    close(&vals(&i), &[132.0]);
    assert!(
        i.unit()
            .unwrap()
            .converter_to(&unit("h"))
            .unwrap()
            .is_identity()
    );
    assert!(cubic().differentiate("x", 1, Some(&unit("s"))).is_err());
}

fn fit_sample() -> DataArray {
    DataArray::from_array(
        ["s", "x"],
        array![
            [1.0, 2.9, 9.2, 19.1, 32.8, 51.2],
            [2.0, NAN, 0.0, -1.0, -2.2, -2.9]
        ]
        .into_dyn(),
    )
    .unwrap()
    .with_dim_coord("x", array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    .unwrap()
    .with_dim_coord("s", array![10i64, 20])
    .unwrap()
}

#[test]
fn polyfit_matches_xarray() {
    let f = fit_sample()
        .polyfit("x", 2, &PolyfitOptions::new().full(true))
        .unwrap();
    let c = f.get("polyfit_coefficients").unwrap().to_owned();
    assert_eq!(c.dims(), ["degree", "s"]);
    close(
        &vals(&c),
        &[
            2.0017857142857167,
            0.01303387334315176,
            0.008214285714276863,
            -1.0654639175257727,
            0.9964285714285935,
            2.022533136966127,
        ],
    );
    close(
        &var(&f, "polyfit_residuals"),
        &[0.12807142857143142, 0.041737849779086886],
    );
    close(&var(&f, "x_matrix_rank"), &[3.0]);
    close(
        &var(&f, "x_singular_values"),
        &[1.636957456050574, 0.5527665599747656, 0.1217350288704827],
    );
    close(
        &f.coords()["degree"]
            .to_f64()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        &[2.0, 1.0, 0.0],
    );
    assert!(f.coords().contains_key("s"));
    // skipna = false: the lane with NaN gives NaN.
    let strict = fit_sample()
        .polyfit("x", 2, &PolyfitOptions::new().skipna(false))
        .unwrap();
    let s = var(&strict, "polyfit_coefficients");
    assert!(s[1].is_nan() && !s[0].is_nan());
}

#[test]
fn polyfit_covariance_and_weights() {
    let one = fit_sample()
        .isel([(
            "s",
            annoray::Indexer::Slice(annoray::PySlice::new(None, Some(1), None)),
        )])
        .unwrap()
        .to_owned();
    let c = one
        .polyfit("x", 2, &PolyfitOptions::new().cov(Covariance::Scaled))
        .unwrap();
    close(
        &var(&c, "polyfit_covariance"),
        &[
            0.0011434948979592134,
            -0.00571747448979607,
            0.0038116496598640547,
            -0.005717474489796069,
            0.03102682823129334,
            -0.02515688775510273,
            0.0038116496598640517,
            -0.02515688775510272,
            0.035067176870749193,
        ],
    );
    close(&var(&c, "polyfit_residuals"), &[0.12807142857143142]);
    let w = one
        .polyfit(
            "x",
            2,
            &PolyfitOptions::new()
                .cov(Covariance::Unscaled)
                .weights(vec![1.0, 2.0, 1.0, 1.0, 0.5, 1.0]),
        )
        .unwrap();
    close(
        &var(&w, "polyfit_coefficients"),
        &[2.0007647362089735, 0.049948444749943074, 0.9109812682591708],
    );
    close(
        &var(&w, "polyfit_covariance"),
        &[
            0.02655095377212597,
            -0.1366214126138521,
            0.10036088675030143,
            -0.13662141261385213,
            0.7564014435470063,
            -0.6135074755112598,
            0.10036088675030148,
            -0.6135074755112601,
            0.6770922838975797,
        ],
    );
    assert!(w.get("polyfit_residuals").is_err());
    // Deficient rank: numpy's minimum-norm solution.
    let short = DataArray::from_array(["x"], array![1.0, 2.0].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![0.0, 1.0])
        .unwrap();
    close(
        &var(
            &short.polyfit("x", 3, &PolyfitOptions::new()).unwrap(),
            "polyfit_coefficients",
        ),
        &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0],
    );
    assert!(
        short
            .polyfit("x", 1, &PolyfitOptions::new().weights(vec![1.0]))
            .is_err()
    );
}

#[test]
fn polyval_restores_units() {
    let z = array![0.0, 1.0, 2.0, 3.0];
    let temp = DataArray::from_array(
        ["z"],
        z.mapv(|v: f64| 15.0 - 6.5 * v + 0.1 * v * v).into_dyn(),
    )
    .unwrap()
    .with_unit(unit("degC"))
    .with_coord(
        "z",
        Variable::new(["z"], z.into_dyn())
            .unwrap()
            .with_unit(unit("km")),
    )
    .unwrap();
    let fit = temp.polyfit("z", 2, &PolyfitOptions::new()).unwrap();
    let coeffs = fit.get("polyfit_coefficients").unwrap().to_owned();
    close(&vals(&coeffs), &[0.1, -6.5, 15.0]);
    // Evaluate at heights given in metres: converted to km first; the result is in °C.
    let at = DataArray::new(
        Variable::new(["h"], array![500.0, 2500.0].into_dyn())
            .unwrap()
            .with_unit(unit("m")),
    );
    let t = polyval(&at, &coeffs, "degree").unwrap();
    assert_eq!(t.unit(), Some(&unit("degC")));
    close(&vals(&t), &[11.775, -0.625]);
    assert!(polyval(&at, &coeffs, "nope").is_err());
}

#[test]
fn datasets() {
    let mut ds = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![0.0, 1.0, 3.0].into_dyn()).unwrap(),
        )
        .unwrap();
    ds.add_var(
        "a",
        Variable::new(["x"], array![0.0, 2.0, 6.0].into_dyn()).unwrap(),
    )
    .unwrap();
    ds.add_var(
        "b",
        Variable::new(["y", "x"], Array2::from_elem((2, 3), 1.0).into_dyn()).unwrap(),
    )
    .unwrap();
    ds.add_var(
        "c",
        Variable::new(["y"], array![5.0, 6.0].into_dyn()).unwrap(),
    )
    .unwrap();
    close(
        &var(&ds.differentiate("x", 1, None).unwrap(), "a"),
        &[2.0, 2.0, 2.0],
    );
    let i = ds.integrate(&["x"], None).unwrap();
    close(&var(&i, "a"), &[9.0]);
    close(&var(&i, "b"), &[3.0, 3.0]);
    close(&var(&i, "c"), &[5.0, 6.0]);
    assert!(!i.coords().contains_key("x"));
    close(
        &var(&ds.cumulative_integrate(&["x"], None).unwrap(), "a"),
        &[0.0, 1.0, 9.0],
    );
    let f = ds.polyfit("x", 1, &PolyfitOptions::new()).unwrap();
    close(&var(&f, "a_polyfit_coefficients"), &[2.0, 0.0]);
    assert_eq!(
        f.get("b_polyfit_coefficients").unwrap().dims(),
        ["degree", "y"]
    );
    assert!(f.get("c_polyfit_coefficients").is_err());
}

#[test]
fn lazy_data_streams_in_slabs() {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let ds = annoray::io::open_dataset(&path).unwrap();
    let (name, dim) = ds
        .data_vars()
        .iter()
        .find(|(_, v)| v.is_lazy() && v.ndim() >= 2 && v.dtype().to_string().starts_with("float"))
        .map(|(k, v)| (k.clone(), v.dims()[1].to_string()))
        .unwrap();
    let da = ds.get(&name).unwrap().to_owned();
    let mut all = annoray::io::open_dataset(&path).unwrap();
    all.load().unwrap();
    let loaded = all.get(&name).unwrap().to_owned();
    if !da.coords().contains_key(dim.as_str()) {
        return;
    }
    let before = annoray::stream_bytes();
    annoray::set_stream_bytes(1);
    let pairs = [
        (
            da.differentiate(&dim, 1, None).unwrap(),
            loaded.differentiate(&dim, 1, None).unwrap(),
        ),
        (
            da.integrate(&[dim.as_str()], None).unwrap(),
            loaded.integrate(&[dim.as_str()], None).unwrap(),
        ),
    ];
    let fit = da.polyfit(&dim, 1, &PolyfitOptions::new()).unwrap();
    let fit_eager = loaded.polyfit(&dim, 1, &PolyfitOptions::new()).unwrap();
    annoray::set_stream_bytes(before);
    for (lazy, eager) in pairs {
        close(&vals(&lazy), &vals(&eager));
    }
    close(
        &var(&fit, "polyfit_coefficients"),
        &var(&fit_eager, "polyfit_coefficients"),
    );
    assert!(da.variable().is_lazy());
}

#[test]
fn cf_calendars_and_epoch_numbers() {
    use annoray::units::{Calendar, CfDate};
    // 360-day calendar: 2000-01-30 → 2000-02-01 is one day (no 31st), 2000-02-01 → 2000-03-01
    // is 30.
    let dates = [(1, 30), (2, 1), (3, 1)]
        .map(|(m, d)| CfDate::from_ymd(Calendar::Day360, 2000, m, d).unwrap());
    let t = DataArray::from_array(["t"], array![0.0, 2.0, 62.0].into_dyn())
        .unwrap()
        .with_coord(
            "t",
            Variable::new(["t"], ndarray::Array1::from(dates.to_vec()).into_dyn()).unwrap(),
        )
        .unwrap();
    let d = t.differentiate("t", 1, Some(&unit("d"))).unwrap();
    close(&vals(&d), &[2.0, 2.0, 2.0]);
    close(
        &vals(&t.integrate(&["t"], Some(&unit("d"))).unwrap()),
        &[1.0 + 960.0],
    );
    let fit = t.polyfit("t", 1, &PolyfitOptions::new()).unwrap();
    // Seconds since 1970-01-01 in that calendar: slope 2 per day.
    close(&var(&fit, "polyfit_coefficients")[..1], &[2.0 / 86400.0]);
    let back = polyval(
        &DataArray::new(t.coords()["t"].clone()),
        &fit.get("polyfit_coefficients").unwrap().to_owned(),
        "degree",
    )
    .unwrap();
    close(&vals(&back), &[0.0, 2.0, 62.0]);

    // Undecoded "days since" numbers: the step is a day, converted by time_unit.
    let n = DataArray::from_array(["t"], array![0.0, 2.0, 6.0].into_dyn())
        .unwrap()
        .with_coord(
            "t",
            Variable::new(["t"], array![0.0, 1.0, 3.0].into_dyn())
                .unwrap()
                .with_unit(unit("days since 2000-01-01")),
        )
        .unwrap();
    let per_day = n.differentiate("t", 1, None).unwrap();
    close(&vals(&per_day), &[2.0, 2.0, 2.0]);
    assert!(per_day.unit().unwrap().is_convertible(&unit("1/d")));
    let per_hour = n.differentiate("t", 1, Some(&unit("h"))).unwrap();
    close(&vals(&per_hour), &[2.0 / 24.0; 3]);
    assert!(n.integrate(&["t"], Some(&unit("m"))).is_err());
}
