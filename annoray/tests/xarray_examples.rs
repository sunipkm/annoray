//! Selection and arithmetic examples checked against Python xarray 2026.4.0
//! (expected values from `fixtures/xarray_examples.py`).
#![allow(clippy::unwrap_used, clippy::arithmetic_side_effects)]

use annoray::ndarray::{Array1, ArrayD, IxDyn, array};
use annoray::units::time::Calendar;
use annoray::{
    CfDate, CoreError, DType, DataArray, DataArrayView, Dataset, Label, LabelIndexer, Unit,
    UnitPolicy, Variable, isel, sel,
};
use annoray::{Duration, Instant};

fn date(y: i64, m: u8, d: u8) -> Instant {
    Instant::from_ymd(y, m, d).unwrap()
}

/// temp(time: 4, lat: 3) = arange(12), time = 2020-01-01, -01-15, -02-01, -03-01; lat = 10, 20, 30.
fn temp() -> DataArray {
    let data = ArrayD::from_shape_vec(IxDyn(&[4, 3]), (0..12).map(f64::from).collect()).unwrap();
    DataArray::from_array(["time", "lat"], data)
        .unwrap()
        .with_name("temp")
        .with_dim_coord(
            "time",
            array![
                date(2020, 1, 1),
                date(2020, 1, 15),
                date(2020, 2, 1),
                date(2020, 3, 1)
            ],
        )
        .unwrap()
        .with_dim_coord("lat", array![10.0, 20.0, 30.0])
        .unwrap()
}

fn vals<S: annoray::Repr>(da: &annoray::DataArrayBase<S>) -> Vec<f64> {
    da.data::<f64>().unwrap().iter().copied().collect()
}

fn dims<S: annoray::Repr>(da: &annoray::DataArrayBase<S>) -> Vec<&str> {
    da.dims().iter().map(|d| d.as_str()).collect()
}

#[test]
fn label_selection_matches_xarray() {
    let t = temp();
    let r = t.sel([("lat", 20.0)]).unwrap();
    assert_eq!(
        (dims(&r), vals(&r)),
        (vec!["time"], vec![1.0, 4.0, 7.0, 10.0])
    );

    let r = sel!(t[lat = 15.0:30.0]).unwrap();
    assert_eq!(vals(&r), vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0]);

    let desc = isel!(t[lat = ::-1]).unwrap();
    let r = sel!(desc[lat = 30.0:15.0]).unwrap();
    assert_eq!(vals(&r), vec![2.0, 1.0, 5.0, 4.0, 8.0, 7.0, 11.0, 10.0]);

    // Partial date strings select the whole period and keep the dimension.
    let r = sel!(t[time = "2020-01"]).unwrap();
    assert_eq!(
        (dims(&r), vals(&r)),
        (vec!["time", "lat"], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    );
    let r = sel!(t[time = "2020-01-10":"2020-02"]).unwrap();
    assert_eq!(vals(&r), vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let r = sel!(t[lat = nearest(24.0)]).unwrap();
    assert_eq!(vals(&r), vec![1.0, 4.0, 7.0, 10.0]);
    assert!(matches!(
        sel!(t[lat = nearest(24.0, tol = 3.0)]),
        Err(CoreError::LabelNotFound { .. })
    ));
    assert_eq!(
        vals(&sel!(t[lat = pad(25.0)]).unwrap()),
        vec![1.0, 4.0, 7.0, 10.0]
    );
    assert_eq!(
        vals(&sel!(t[lat = backfill(25.0)]).unwrap()),
        vec![2.0, 5.0, 8.0, 11.0]
    );
    assert_eq!(
        vals(&sel!(desc[lat = pad(25.0)]).unwrap()),
        vec![2.0, 5.0, 8.0, 11.0]
    );
    assert_eq!(
        vals(&sel!(desc[lat = backfill(25.0)]).unwrap()),
        vec![1.0, 4.0, 7.0, 10.0]
    );

    let r = sel!(t[time = nearest("2020-01-20")]).unwrap();
    assert_eq!((dims(&r), vals(&r)), (vec!["lat"], vec![3.0, 4.0, 5.0]));

    let r = isel!(t[time = -1, lat = ::-1]).unwrap();
    assert_eq!(vals(&r), vec![11.0, 10.0, 9.0]);

    let r = sel!(t[time = ::2, lat = 10.0:30.0:2]).unwrap();
    assert_eq!(vals(&r), vec![0.0, 2.0, 6.0, 8.0]);

    assert!(matches!(
        t.sel([("lat", 25.0)]),
        Err(CoreError::LabelNotFound { .. })
    ));
}

#[test]
fn coordinates_follow_selection() {
    let t = temp();
    let r = sel!(t[lat = 15.0:30.0, time = "2020-02"]).unwrap();
    assert_eq!(
        r.coord("lat")
            .unwrap()
            .data::<f64>()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec![20.0, 30.0]
    );
    assert_eq!(
        r.coord("time")
            .unwrap()
            .data::<Instant>()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec![date(2020, 2, 1)]
    );
    // Scalar selection keeps a 0-d coordinate, as xarray does.
    let r = t.sel([("lat", 20.0)]).unwrap();
    assert_eq!(r.coord("lat").unwrap().ndim(), 0);
}

#[test]
fn arithmetic_matches_xarray() {
    let t = temp();
    let w = DataArray::from_array(["lat"], array![1.0, 2.0, 3.0].into_dyn())
        .unwrap()
        .with_dim_coord("lat", array![10.0, 20.0, 30.0])
        .unwrap();
    let r = (&t + &w).unwrap();
    assert_eq!(dims(&r), vec!["time", "lat"]);
    assert_eq!(
        vals(&r),
        vec![
            1.0, 3.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0, 11.0, 10.0, 12.0, 14.0
        ]
    );

    let x = DataArray::from_array(["x"], array![1.0, 2.0].into_dyn()).unwrap();
    let r = (&w * &x).unwrap();
    assert_eq!(
        (dims(&r), vals(&r)),
        (vec!["lat", "x"], vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0])
    );

    // Datetime differences give durations.
    let time = t.coord("time").unwrap();
    let td = DataArray::new(time.clone());
    let first = DataArray::new(time.isel([("time", 0)]).unwrap().to_owned());
    let d = (&td - &first).unwrap();
    assert_eq!(d.dtype(), DType::TimeDelta);
    let days: Vec<i64> = d
        .data::<Duration>()
        .unwrap()
        .iter()
        .map(|d| d.nanos() / 86_400_000_000_000)
        .collect();
    assert_eq!(days, vec![0, 14, 31, 60]);

    // Integer division is true division; multiplication stays integer.
    let a = DataArray::from_array(["x"], array![7i32, -7, 9].into_dyn()).unwrap();
    let b = DataArray::from_array(["x"], array![2i32, 2, 3].into_dyn()).unwrap();
    assert_eq!(vals(&(&a / &b).unwrap()), vec![3.5, -3.5, 3.0]);
    let c = DataArray::from_array(["x"], array![2i32, 3, 4].into_dyn()).unwrap();
    let m = (&a * &c).unwrap();
    assert_eq!(
        m.data::<i32>().unwrap().iter().cloned().collect::<Vec<_>>(),
        vec![14, -21, 36]
    );
}

#[test]
fn arithmetic_errors() {
    let t = temp();
    // Disjoint coordinates on a shared dimension: the inner join is empty, as in xarray.
    let shifted = DataArray::from_array(["lat"], array![1.0, 2.0, 3.0].into_dyn())
        .unwrap()
        .with_dim_coord("lat", array![11.0, 21.0, 31.0])
        .unwrap();
    assert_eq!((&t + &shifted).unwrap().len_of("lat").unwrap(), 0);
    // Mismatched lengths.
    let short = DataArray::from_array(["lat"], array![1.0, 2.0].into_dyn()).unwrap();
    assert!(matches!(
        &t + &short,
        Err(CoreError::DimLengthMismatch { .. })
    ));
    // Integer overflow is an error, not a panic or a wrap.
    let big = DataArray::from_array(["x"], array![i8::MAX].into_dyn()).unwrap();
    assert!(matches!(&big + &big, Err(CoreError::ArithmeticOverflow(_))));
    // Strings don't do arithmetic.
    let s = DataArray::from_array(["x"], array!["a".to_owned()].into_dyn()).unwrap();
    assert!(matches!(&s + &s, Err(CoreError::UnsupportedDType { .. })));
}

#[test]
fn unit_aware_arithmetic() {
    let u = |s: &str| s.parse::<Unit>().unwrap();
    let dist = DataArray::from_array(["x"], array![1.0, 2.0].into_dyn())
        .unwrap()
        .with_unit(u("km"));
    let more = DataArray::from_array(["x"], array![500.0, 250.0].into_dyn())
        .unwrap()
        .with_unit(u("m"));
    let time = DataArray::from_array(["x"], array![2.0, 4.0].into_dyn())
        .unwrap()
        .with_unit(u("h"));

    // Strict by default: km + m must be rescaled explicitly...
    assert!(matches!(
        &dist + &more,
        Err(CoreError::Unit(annoray::UnitError::ScaleMismatch { .. }))
    ));
    // ...or with AutoRescale.
    let sum = dist
        .binary(annoray::BinOp::Add, &more, UnitPolicy::AutoRescale)
        .unwrap();
    assert_eq!(vals(&sum), vec![1.5, 2.25]);
    assert_eq!(sum.unit(), Some(&u("km")));
    let sum = (&dist.to_unit(&u("m")).unwrap() + &more).unwrap();
    assert_eq!(vals(&sum), vec![1500.0, 2250.0]);

    // Products and quotients combine units.
    let speed = (&dist / &time).unwrap();
    assert_eq!(speed.unit(), Some(&u("km/h")));
    assert_eq!(
        vals(&speed.to_unit(&u("m/s")).unwrap()),
        vec![500.0 / 3600.0, 500.0 / 3600.0]
    );
    assert!(matches!(&dist + &time, Err(CoreError::Unit(_))));

    // Scalars are dimensionless.
    assert_eq!(vals(&(&dist * 2.0).unwrap()), vec![2.0, 4.0]);
    assert_eq!((2.0 * &dist).unwrap().unit(), Some(&u("km")));
    assert!((&dist + 1.0).is_err());

    // Temperatures: °C − °C is a difference in K; °C + K difference stays °C.
    let c = DataArray::from_array(["x"], array![20.0, 25.0].into_dyn())
        .unwrap()
        .with_unit(u("degC"));
    let c0 = DataArray::from_array(["x"], array![15.0, 15.0].into_dyn())
        .unwrap()
        .with_unit(u("degC"));
    let diff = (&c - &c0).unwrap();
    assert_eq!((vals(&diff), diff.unit()), (vec![5.0, 10.0], Some(&u("K"))));
    let warmer = (&c0 + &diff).unwrap();
    assert_eq!(
        (vals(&warmer), warmer.unit()),
        (vec![20.0, 25.0], Some(&u("degC")))
    );
    assert!((&c + &c0).is_err());
    let f = DataArray::from_array(["x"], array![32.0, 212.0].into_dyn())
        .unwrap()
        .with_unit(u("degF"));
    let delta = c
        .binary(annoray::BinOp::Sub, &f, UnitPolicy::AutoRescale)
        .unwrap();
    assert_eq!(vals(&delta), vec![20.0, -75.0]);

    // Chaining with one `?`.
    let r = (&dist + &dist * &DataArray::from_array(["x"], array![1.0, 1.0].into_dyn()).unwrap())
        .unwrap();
    assert_eq!(vals(&r), vec![2.0, 4.0]);
}

#[test]
fn unit_aware_labels() {
    let height = DataArray::from_array(["z"], array![0.0, 1.0, 2.0].into_dyn())
        .unwrap()
        .with_coord(
            "z",
            Variable::new(["z"], array![0.0, 500.0, 1000.0].into_dyn())
                .unwrap()
                .with_unit("m".parse().unwrap()),
        )
        .unwrap();
    // 0.5 km is converted to 500 m before lookup.
    let r = sel!(height[z = { Label::quantity(0.5, "km").unwrap() }]).unwrap();
    assert_eq!(vals(&r), vec![1.0]);
    let r = height
        .sel([(
            "z",
            LabelIndexer::slice(Some(Label::quantity(0.4, "km").unwrap()), None),
        )])
        .unwrap();
    assert_eq!(vals(&r), vec![1.0, 2.0]);
    let r = height
        .sel([(
            "z",
            LabelIndexer::nearest_within(
                Label::quantity(0.45, "km").unwrap(),
                Label::quantity(100.0, "m").unwrap(),
            ),
        )])
        .unwrap();
    assert_eq!(vals(&r), vec![1.0]);
    assert!(matches!(
        sel!(height[z = { Label::quantity(1.0, "s").unwrap() }]),
        Err(CoreError::Unit(_))
    ));
}

#[test]
fn cf_calendar_coordinates() {
    let days: Vec<CfDate> = [(2001, 2, 28), (2001, 2, 29), (2001, 2, 30), (2001, 3, 1)]
        .iter()
        .map(|(y, m, d)| CfDate::from_ymd(Calendar::Day360, *y, *m, *d).unwrap())
        .collect();
    let da = DataArray::from_array(["time"], array![1.0, 2.0, 3.0, 4.0].into_dyn())
        .unwrap()
        .with_dim_coord("time", Array1::from(days))
        .unwrap();
    assert_eq!(
        vals(&sel!(da[time = "2001-02"]).unwrap()),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(vals(&sel!(da[time = "2001-02-30"]).unwrap()), vec![3.0]);
    assert_eq!(
        vals(&sel!(da[time = "2001-02-29":]).unwrap()),
        vec![2.0, 3.0, 4.0]
    );
    assert!(sel!(da[time = "2001-02-31"]).is_err());
}

#[test]
fn errors_are_results() {
    let t = temp();
    let plain = DataArray::from_array(["x"], array![1.0, 2.0].into_dyn()).unwrap();
    assert!(matches!(
        plain.sel([("x", 1.0)]),
        Err(CoreError::NoIndex(_))
    ));
    assert!(matches!(
        t.sel([("depth", 1.0)]),
        Err(CoreError::NoSuchDim(_))
    ));
    assert!(matches!(
        t.sel([("lat", "abc")]),
        Err(CoreError::InvalidLabel { .. })
    ));
    assert!(matches!(
        sel!(t[time = "not a date"]),
        Err(CoreError::InvalidLabel { .. })
    ));
    let unsorted = DataArray::from_array(["x"], array![1.0, 2.0, 3.0].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![3.0, 1.0, 2.0])
        .unwrap();
    assert!(matches!(
        sel!(unsorted[x = pad(1.5)]),
        Err(CoreError::NotMonotonic(_))
    ));
    assert_eq!(vals(&sel!(unsorted[x = 1.0]).unwrap()), vec![2.0]);
    assert_eq!(vals(&sel!(unsorted[x = 1.0:2.0]).unwrap()), vec![2.0, 3.0]);
}

#[test]
fn dataset_views_and_mutation() {
    let mut ds = Dataset::new()
        .with_coord(
            "lat",
            Variable::new(["lat"], array![10.0, 20.0, 30.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var("temp", temp().into_parts().1)
        .unwrap()
        .with_var(
            "elev",
            Variable::new(["lat"], array![100.0, 200.0, 300.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_attrs([("title", "test")]);
    assert_eq!(
        ds.sizes().into_iter().collect::<Vec<_>>(),
        vec![("lat".into(), 3), ("time".into(), 4)]
    );

    // Selection applies to every variable that has the dimension.
    let v = sel!(ds[lat = 20.0:]).unwrap();
    assert_eq!(
        v.data_vars()["elev"]
            .data::<f64>()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec![200.0, 300.0]
    );
    assert_eq!(v.data_vars()["temp"].shape(), &[4, 2]);
    let one: DataArrayView<'_> = v.get("temp").unwrap();
    assert_eq!(
        one.coords().keys().map(|k| k.as_str()).collect::<Vec<_>>(),
        vec!["lat"]
    );

    // Writes through a mutable view reach the owner; coordinates are untouched.
    {
        let mut m = sel!(mut ds[lat = 30.0]).unwrap();
        m.data_var_mut("elev")
            .unwrap()
            .data_mut::<f64>()
            .unwrap()
            .fill(-1.0);
        assert!(m.data_var_mut("lat").is_err()); // coordinates are not data variables
    }
    assert_eq!(
        ds.data_vars()["elev"]
            .data::<f64>()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec![100.0, 200.0, -1.0]
    );
    let mut temp_view = ds.get_mut("temp").unwrap();
    isel!(mut temp_view[time = 0])
        .unwrap()
        .data_mut::<f64>()
        .unwrap()
        .fill(0.5);
    assert_eq!(ds.data_vars()["temp"].data::<f64>().unwrap()[[0, 2]], 0.5);

    // Conflicting dimension lengths are rejected.
    let bad = Variable::new(["lat"], array![1.0, 2.0].into_dyn()).unwrap();
    assert!(matches!(
        ds.add_var("bad", bad),
        Err(CoreError::DimLengthMismatch { .. })
    ));
    assert!(matches!(
        ds.add_var(
            "elev",
            Variable::new(["lat"], array![0.0, 0.0, 0.0].into_dyn()).unwrap()
        ),
        Err(CoreError::NameConflict(_))
    ));
}

#[test]
fn display_looks_like_xarray() {
    let t = temp()
        .with_unit("K".parse().unwrap())
        .with_attrs([("long_name", "air temperature")]);
    let s = t.to_string();
    assert!(
        s.starts_with("<annoray.DataArray \"temp\" (time: 4, lat: 3)> Size: 96B"),
        "{s}"
    );
    assert!(s.contains("Units: K"), "{s}");
    assert!(
        s.contains("  * time (time) datetime 2020-01-01 00:00:00"),
        "{s}"
    );
    assert!(s.contains("  * lat  (lat) float64 10 20 30"), "{s}");
    assert!(s.contains("    long_name: air temperature"), "{s}");

    let ds = Dataset::new().with_var("t", t.into_parts().1).unwrap();
    let s = ds.to_string();
    assert!(s.contains("Dimensions:  (time: 4, lat: 3)"), "{s}");
    assert!(
        s.contains("    t (time, lat) float64 [K] 0 1 2 3 4 5 ..."),
        "{s}"
    );
}

#[test]
fn fancy_indexing_matches_xarray() {
    let t = temp();
    let r = t.take("time", &[3, 0, -1]).unwrap();
    assert_eq!(
        vals(&r),
        vec![9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 9.0, 10.0, 11.0]
    );
    assert_eq!(
        r.coord("time").unwrap().data::<Instant>().unwrap()[[1]],
        date(2020, 1, 1)
    );
    let r = t.sel_take("lat", [30.0, 10.0]).unwrap();
    assert_eq!(vals(&r), vec![2.0, 0.0, 5.0, 3.0, 8.0, 6.0, 11.0, 9.0]);
    let r = t
        .sel_take(
            "lat",
            [LabelIndexer::nearest(29.0), LabelIndexer::nearest(11.0)],
        )
        .unwrap();
    assert_eq!(vals(&r), vec![2.0, 0.0, 5.0, 3.0, 8.0, 6.0, 11.0, 9.0]);
    assert!(matches!(t.take("time", &[4]), Err(CoreError::Index(_))));
    // Partial dates expand to every matching position.
    let r = t.sel_take("time", ["2020-01", "2020-03-01"]).unwrap();
    assert_eq!(r.shape(), &[3, 3]);
    let ds = Dataset::new().with_var("temp", t.into_parts().1).unwrap();
    assert_eq!(ds.take("lat", &[0, 0]).unwrap().sizes()["lat"], 2);
}
