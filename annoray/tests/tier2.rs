//! `equals`/`identical`/`broadcast_equals`, `broadcast`, `cov`/`corr`.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::Unit;
use annoray::{DataArray, Dataset, Name, broadcast, corr, cov};
use ndarray::array;

fn u(s: &str) -> Unit {
    s.parse().unwrap()
}

fn arr(v: ndarray::ArrayD<f64>) -> DataArray {
    DataArray::from_array(["x"], v)
        .unwrap()
        .with_dim_coord("x", array![0.0, 1.0, 2.0])
        .unwrap()
}

#[test]
fn equals_converts_units_identical_does_not() {
    let km = arr(array![1.0, f64::NAN, 2.5].into_dyn()).with_unit(u("km"));
    let m = arr(array![1000.0, f64::NAN, 2500.0].into_dyn()).with_unit(u("m"));
    assert!(km.equals(&m).unwrap(), "1 km equals 1000 m; NaN equals NaN");
    assert!(!km.identical(&m).unwrap());
    assert!(km.identical(&km.clone()).unwrap());
    let named = km.clone().with_name("n");
    assert!(km.equals(&named).unwrap() && !km.identical(&named).unwrap());
    let attrd = km.clone().with_attrs([("k", "v")]);
    assert!(km.equals(&attrd).unwrap() && !km.identical(&attrd).unwrap());
    let s = arr(array![1.0, f64::NAN, 2.5].into_dyn()).with_unit(u("s"));
    assert!(!km.equals(&s).unwrap());
    let other_x = km
        .clone()
        .with_dim_coord("x", array![5.0, 6.0, 7.0])
        .unwrap();
    assert!(!km.equals(&other_x).unwrap());
    let text = DataArray::from_array(["x"], array!["a", "b", "c"].mapv(String::from).into_dyn())
        .unwrap()
        .with_dim_coord("x", array![0.0, 1.0, 2.0])
        .unwrap();
    assert!(!km.equals(&text).unwrap());
    // Broadcast: a scalar against an array of that value.
    let three = arr(array![3.0, 3.0, 3.0].into_dyn());
    let scalar = DataArray::scalar(3.0).unwrap();
    assert!(three.broadcast_equals(&scalar).unwrap());
    assert!(!three.equals(&scalar).unwrap());
}

#[test]
fn dataset_equality_ignores_order_identical_checks_attrs() {
    let mut a = Dataset::new();
    a.add_var(
        "p",
        arr(array![1.0, 2.0, 3.0].into_dyn()).variable().clone(),
    )
    .unwrap();
    a.add_var(
        "q",
        arr(array![4.0, 5.0, 6.0].into_dyn()).variable().clone(),
    )
    .unwrap();
    let mut b = Dataset::new();
    b.add_var("q", a.data_vars()["q"].clone()).unwrap();
    b.add_var("p", a.data_vars()["p"].clone()).unwrap();
    assert!(a.equals(&b).unwrap() && a.identical(&b).unwrap());
    // NaN attributes equal each other (a _FillValue read twice).
    a.attrs_mut().insert("fill".into(), f64::NAN.into());
    b.attrs_mut().insert("fill".into(), f64::NAN.into());
    assert!(a.identical(&b).unwrap());
    b.attrs_mut().insert("extra".into(), "1".into());
    assert!(a.equals(&b).unwrap() && !a.identical(&b).unwrap());
}

#[test]
fn broadcast_prepends_missing_dims() {
    let x = arr(array![1.0, 2.0, 3.0].into_dyn());
    let y = DataArray::from_array(["y"], array![10.0, 20.0].into_dyn())
        .unwrap()
        .with_dim_coord("y", array![0.5, 1.5])
        .unwrap();
    let out = broadcast(&[&x, &y], &[]).unwrap();
    assert_eq!(out[0].dims(), ["x", "y"]);
    assert_eq!(out[1].dims(), ["x", "y"]);
    assert!(out[0].coords().contains_key("y") && out[1].coords().contains_key("x"));
    assert_eq!(
        out[0].to_f64().unwrap().iter().copied().collect::<Vec<_>>(),
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
    let like = x.broadcast_like(&y, &[]).unwrap();
    assert_eq!(like.dims(), ["x", "y"]);
    let excluded = x.broadcast_like(&y, &["y"]).unwrap();
    assert_eq!(excluded.dims(), ["x"]);
}

#[test]
fn covariance_and_correlation() {
    let a = arr(array![1.0, 2.0, 4.0].into_dyn()).with_unit(u("m"));
    let b = arr(array![2.0, 4.0, f64::NAN].into_dyn()).with_unit(u("s"));
    // Valid pairs: (1, 2), (2, 4): means 1.5 and 3; products 0.5·1 + 0.5·1 = 1; ddof 1 → 1.
    let c = cov(&a, &b, None, 1).unwrap();
    assert!((c.to_f64().unwrap().first().copied().unwrap() - 1.0).abs() < 1e-12);
    assert!(c.unit().unwrap().is_convertible(&u("m s")));
    let r = corr(&a, &b, None).unwrap();
    assert!((r.to_f64().unwrap().first().copied().unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(r.unit(), None);
    let dims = [Name::from("x")];
    assert_eq!(cov(&a, &a, Some(&dims), 0).unwrap().ndim(), 0);
}

#[test]
fn lazy_data_stays_lazy_and_streams() {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let open = || annoray::io::open_dataset(&path).unwrap();
    let ds = open();
    let t = ds.get("temp").unwrap().to_owned();
    let k = DataArray::scalar(280.0).unwrap().with_unit(u("K"));
    let cond = t.gt(&k).unwrap();
    let doubled = (&t * 2.0).unwrap();
    // Three-way where, broadcast: lazy, nothing read.
    let w = annoray::where_(&cond, &t, &doubled).unwrap();
    assert!(w.variable().is_lazy());
    let flag = ds.get("flag").unwrap().to_owned();
    let b = broadcast(&[&flag, &t], &[]).unwrap();
    assert!(b[0].variable().is_lazy());
    assert_eq!(
        b[0].dims(),
        ["lat", "time", "lon"],
        "order of first appearance"
    );
    // Equality: in slabs, never holding the whole array.
    assert!(t.equals(&open().get("temp").unwrap()).unwrap());
    assert!(t.variable().is_lazy(), "streamed, not cached");
    // Same results as on loaded data.
    let loaded = open().loaded().unwrap();
    let lt = loaded.get("temp").unwrap().to_owned();
    let lw = annoray::where_(&lt.gt(&k).unwrap(), &lt, &(&lt * 2.0).unwrap()).unwrap();
    assert!(w.identical(&lw).unwrap());
    let lb = broadcast(&[&loaded.get("flag").unwrap().to_owned(), &lt], &[]).unwrap();
    assert!(b[0].identical(&lb[0]).unwrap());
}

#[test]
fn calendar_conversion() {
    use annoray::units::Calendar;
    use annoray::{AlignOn, CalendarOptions, CoreError};
    let days: Vec<chrono::NaiveDateTime> = (0..366)
        .map(|d| {
            chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
                .unwrap()
                .and_hms_opt(6, 0, 0)
                .unwrap()
                + chrono::TimeDelta::days(d)
        })
        .collect();
    let n = days.len();
    let a = DataArray::from_array(
        ["time"],
        ndarray::Array1::from_shape_fn(n, |i| i as f64).into_dyn(),
    )
    .unwrap()
    .with_dim_coord("time", ndarray::Array1::from(days))
    .unwrap();
    // Feb 29 does not exist in noleap: dropped with its value.
    let nl = a
        .convert_calendar("time", &CalendarOptions::new(Calendar::NoLeap))
        .unwrap();
    assert_eq!(nl.len(), 365);
    assert_eq!(nl.to_f64().unwrap()[59], 60.0);
    // 360_day needs align_on; by year it keeps 360 dates, the time of day kept.
    assert!(matches!(
        a.convert_calendar("time", &CalendarOptions::new(Calendar::Day360)),
        Err(CoreError::InvalidReduction(_))
    ));
    let y = a
        .convert_calendar(
            "time",
            &CalendarOptions::new(Calendar::Day360).align_on(AlignOn::Year),
        )
        .unwrap();
    assert_eq!(y.len(), 360);
    let first = y
        .coord("time")
        .unwrap()
        .data::<annoray::units::CfDate>()
        .unwrap()[0];
    assert_eq!((first.month(), first.day(), first.hour()), (1, 1, 6));
    // Back to the standard calendar with gaps filled.
    let back = nl
        .convert_calendar(
            "time",
            &CalendarOptions::new(Calendar::Standard).missing(-1.0),
        )
        .unwrap();
    assert_eq!(back.len(), 366);
    assert_eq!(back.to_f64().unwrap()[59], -1.0);
    assert_eq!(
        back.coord("time").unwrap().dtype(),
        annoray::DType::DateTime
    );
}

#[test]
fn date_ranges_in_calendars() {
    use annoray::units::Calendar;
    let d = annoray::date_range_in("2000-02-27", None, Some(5), "D", Calendar::Day360).unwrap();
    let days: Vec<(u8, u8)> = d.iter().map(|t| (t.month(), t.day())).collect();
    assert_eq!(days, [(2, 27), (2, 28), (2, 29), (2, 30), (3, 1)]);
    let m = annoray::date_range_in(
        "2000-01-15",
        Some("2001-01-01"),
        None,
        "MS",
        Calendar::NoLeap,
    )
    .unwrap();
    assert_eq!(m.len(), 12);
    assert_eq!((m[0].month(), m[11].year()), (2, 2001));
    // A year means its first day, as in pandas.
    let y = annoray::date_range("2000", 2, "D").unwrap();
    assert_eq!(y[0].to_string(), "2000-01-01 00:00:00");
    assert!(annoray::date_range_in("2000-01-01", None, None, "D", Calendar::NoLeap).is_err());
}

#[test]
fn exponential_moving_statistics() {
    use annoray::WindowType;
    // α = 1/2 (span 3): mean of [2, 4] is (2·½ + 4) / (½ + 1) = 10/3.
    let a = DataArray::from_array(["t"], array![2.0, 4.0, f64::NAN].into_dyn()).unwrap();
    let r = a.rolling_exp("t", 3.0, WindowType::Span, 0.0).unwrap();
    let m = r.mean().unwrap().to_f64().unwrap();
    assert!((m[1] - 10.0 / 3.0).abs() < 1e-14);
    assert!((m[2] - 10.0 / 3.0).abs() < 1e-14, "a gap keeps the mean");
    assert!(
        r.var().unwrap().to_f64().unwrap()[0].is_nan(),
        "one value: no variance"
    );
    // Large offset, small spread: the variance stays exact.
    let off = 1e12;
    let stored = array![off, off + 1e-3, off - 1e-3];
    // The same data, shifted back exactly (as stored, not as written).
    let shifted = stored.mapv(|v| v - off);
    let b = DataArray::from_array(["t"], stored.into_dyn()).unwrap();
    let c = DataArray::from_array(["t"], shifted.into_dyn()).unwrap();
    let (vb, vc) = (
        b.rolling_exp("t", 0.5, WindowType::Alpha, 0.0)
            .unwrap()
            .var()
            .unwrap(),
        c.rolling_exp("t", 0.5, WindowType::Alpha, 0.0)
            .unwrap()
            .var()
            .unwrap(),
    );
    let (vb, vc) = (vb.to_f64().unwrap(), vc.to_f64().unwrap());
    assert!(
        ((vb[2] - vc[2]) / vc[2]).abs() < 1e-3,
        "{} vs {}",
        vb[2],
        vc[2]
    );
    assert!(a.rolling_exp("t", 0.5, WindowType::Span, 0.0).is_err());
    assert!(a.rolling_exp("t", 1.5, WindowType::Alpha, 0.0).is_err());
}
