//! resample against Python xarray (`fixtures/resample.txt`, from `fixtures/gen_resample.py`).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use annoray::units::{Calendar, CfDate};
use annoray::{DataArray, Dataset, NaT, TypedArray, Variable, View};
use chrono::NaiveDateTime;
use ndarray::{Array1, ArrayD, IxDyn, array};

fn dt(s: &str) -> NaiveDateTime {
    s.parse().unwrap()
}

fn std_series() -> DataArray {
    let t: Array1<NaiveDateTime> = [
        "2000-01-01T06:00:00",
        "2000-01-15T00:00:00",
        "2000-01-31T00:00:00",
        "2000-01-31T12:00:00",
        "2000-02-10T00:00:00",
        "2000-04-02T00:00:00",
        "2000-07-01T00:00:00",
    ]
    .iter()
    .map(|s| dt(s))
    .collect();
    DataArray::from_array(
        ["time"],
        ArrayD::from_shape_vec(IxDyn(&[7]), (0..7).map(f64::from).collect()).unwrap(),
    )
    .unwrap()
    .with_dim_coord("time", t)
    .unwrap()
}

fn noleap_series() -> DataArray {
    let d = |m, day, h| {
        CfDate::from_ymd(
            Calendar::NoLeap,
            if m == 13 { 2002 } else { 2001 },
            if m == 13 { 1 } else { m },
            day,
        )
        .unwrap()
        .and_hms_nano(h, 0, 0, 0)
        .unwrap()
    };
    let t: Array1<CfDate> = vec![
        d(1, 31, 0),
        d(2, 28, 12),
        d(3, 1, 0),
        d(6, 30, 0),
        d(13, 1, 0),
    ]
    .into();
    DataArray::from_array(["time"], array![0.0, 1.0, 2.0, 3.0, 4.0].into_dyn())
        .unwrap()
        .with_dim_coord("time", t)
        .unwrap()
}

fn text(t: &TypedArray<View<'_>>) -> String {
    let v: Vec<String> = match t {
        TypedArray::F64(a) => a
            .iter()
            .map(|v| {
                if v.is_nan() {
                    "nan".into()
                } else {
                    format!("{v:?}")
                }
            })
            .collect(),
        #[allow(clippy::cast_precision_loss)]
        TypedArray::I64(a) => a.iter().map(|v| format!("{:?}", *v as f64)).collect(),
        TypedArray::DateTime(a) => a
            .iter()
            .map(|d| d.format("%Y-%m-%dT%H:%M:%S").to_string())
            .collect(),
        TypedArray::CfDate(a) => a
            .iter()
            .map(|d| {
                format!(
                    "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                    d.year(),
                    d.month(),
                    d.day(),
                    d.hour(),
                    d.minute(),
                    d.second()
                )
            })
            .collect(),
        _ => panic!("unexpected type"),
    };
    v.join(" ")
}

#[test]
fn matches_xarray() {
    let table = include_str!("../../fixtures/resample.txt");
    let mut checked = 0;
    for line in table.lines() {
        let f: Vec<&str> = line.split('|').collect();
        let da = match f[0] {
            "std" => std_series(),
            "noleap" => noleap_series(),
            other => panic!("unknown series {other}"),
        };
        let r = da
            .resample("time", f[1])
            .unwrap_or_else(|e| panic!("{line}: {e}"));
        let mean = r.mean().unwrap();
        assert_eq!(
            text(&mean.coord("time").unwrap().typed().unwrap()),
            f[2],
            "{} labels",
            f[1]
        );
        assert_eq!(
            text(&mean.variable().typed().unwrap()),
            f[3],
            "{} mean",
            f[1]
        );
        assert_eq!(
            text(&r.sum().unwrap().variable().typed().unwrap()),
            f[4],
            "{} sum",
            f[1]
        );
        assert_eq!(
            text(&r.count().unwrap().variable().typed().unwrap()),
            f[5],
            "{} count",
            f[1]
        );
        assert_eq!(
            text(&r.first().unwrap().variable().typed().unwrap()),
            f[6],
            "{} first",
            f[1]
        );
        checked += 1;
    }
    assert_eq!(checked, 22);
}

#[test]
fn nat_unsorted_and_datasets() {
    // NaT is in no bin; times need not be sorted (xarray requires a monotonic index).
    let t: Array1<NaiveDateTime> = vec![
        dt("2000-01-03T00:00:00"),
        NaiveDateTime::NAT,
        dt("2000-01-01T12:00:00"),
        dt("2000-01-03T06:00:00"),
    ]
    .into();
    let da = DataArray::from_array(["time"], array![1.0, 2.0, 3.0, 4.0].into_dyn())
        .unwrap()
        .with_dim_coord("time", t)
        .unwrap();
    let r = da.resample("time", "D").unwrap();
    assert_eq!(r.indices(), [vec![2], vec![], vec![0, 3]]);
    let sum = r.sum().unwrap();
    assert_eq!(
        sum.data::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [3.0, 0.0, 5.0]
    );

    // Datasets, with a second dimension and a variable without time.
    let ds = Dataset::new()
        .with_coord(
            "time",
            Variable::new(
                ["time"],
                std_series()
                    .coord("time")
                    .unwrap()
                    .data::<NaiveDateTime>()
                    .unwrap()
                    .to_owned(),
            )
            .unwrap(),
        )
        .unwrap()
        .with_var(
            "v",
            Variable::new(
                ["time", "y"],
                ArrayD::from_shape_vec(IxDyn(&[7, 2]), (0..14).map(f64::from).collect()).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
        .with_var(
            "c",
            Variable::new(["y"], array![7.0, 8.0].into_dyn()).unwrap(),
        )
        .unwrap();
    let m = ds.resample("time", "QS").unwrap().mean().unwrap();
    assert_eq!(m.data_vars()["v"].dims(), ["time", "y"]);
    assert_eq!(
        m.data_vars()["v"]
            .data::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [4.0, 5.0, 10.0, 11.0, 12.0, 13.0]
    );
    assert_eq!(m.data_vars()["c"].dims(), ["y"]);

    // Errors.
    assert!(std_series().resample("time", "fortnight").is_err());
    assert!(std_series().resample("time", "0D").is_err());
    assert!(std_series().resample("time", "MS-JAN").is_err());
    assert!(std_series().resample("nope", "D").is_err());
    assert!(noleap_series().resample("time", "W").is_err());
}
