//! groupby against Python xarray (`fixtures/groupby.txt`, from `fixtures/gen_groupby.py`).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::arithmetic_side_effects
)]

use chrono::{NaiveDate, NaiveDateTime};
use ndarray::{Array1, ArrayD, IxDyn, array};
use xr_core::xr_units::{Calendar, CfDate};
use xr_core::{BinOptions, CoreError, DataArray, Dataset, NaT, Name, TypedArray, Variable, View};

const NAN: f64 = f64::NAN;

fn lab() -> DataArray {
    let labels: Array1<String> = ["b", "a", "b", "c", "c", "a"]
        .map(String::from)
        .to_vec()
        .into();
    DataArray::from_array(["x"], array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![10i64, 20, 30, 40, 50, 60])
        .unwrap()
        .with_coord("lab", Variable::new(["x"], labels.into_dyn()).unwrap())
        .unwrap()
}

fn num() -> DataArray {
    DataArray::from_array(["x"], array![1.0, 2.0, 3.0, 4.0].into_dyn())
        .unwrap()
        .with_coord(
            "k",
            Variable::new(["x"], array![2.0, NAN, 1.0, 2.0].into_dyn()).unwrap(),
        )
        .unwrap()
}

fn months(n: u32) -> Array1<NaiveDateTime> {
    (0..n)
        .map(|i| {
            NaiveDate::from_ymd_opt(2000 + i32::try_from(i / 12).unwrap(), i % 12 + 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
        })
        .collect()
}

fn grid() -> DataArray {
    let values: Vec<f64> = (0..48).map(f64::from).collect();
    DataArray::from_array(
        ["time", "y"],
        ArrayD::from_shape_vec(IxDyn(&[24, 2]), values).unwrap(),
    )
    .unwrap()
    .with_dim_coord("time", months(24))
    .unwrap()
    .with_dim_coord("y", array![0i64, 1])
    .unwrap()
}

fn noleap() -> DataArray {
    let t: Array1<CfDate> = [1u8, 2, 3, 1, 2, 3]
        .iter()
        .map(|m| CfDate::from_ymd(Calendar::NoLeap, 2001, *m, 15).unwrap())
        .collect();
    DataArray::from_array(["time"], array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0].into_dyn())
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
        TypedArray::I64(a) => a.iter().map(i64::to_string).collect(),
        TypedArray::Str(a) => a.iter().cloned().collect(),
        _ => panic!("unexpected type"),
    };
    v.join(" ")
}

fn compute(case: &str) -> (DataArray, Option<&'static str>) {
    let lab = lab();
    let g = grid();
    match case {
        "lab_mean" => (lab.groupby("lab").unwrap().mean().unwrap(), Some("lab")),
        "lab_sum" => (lab.groupby("lab").unwrap().sum().unwrap(), Some("lab")),
        "lab_max" => (lab.groupby("lab").unwrap().max().unwrap(), Some("lab")),
        "lab_count" => (lab.groupby("lab").unwrap().count().unwrap(), Some("lab")),
        "lab_first" => (lab.groupby("lab").unwrap().first().unwrap(), Some("lab")),
        "lab_last" => (lab.groupby("lab").unwrap().last().unwrap(), Some("lab")),
        "num_sum" => (num().groupby("k").unwrap().sum().unwrap(), Some("k")),
        "bins_mean" => (
            lab.groupby_bins(
                "x",
                &[0.0, 25.0, 45.0, 100.0, 200.0],
                &BinOptions::default(),
            )
            .unwrap()
            .mean()
            .unwrap(),
            Some("x_bins"),
        ),
        "bins_left" => (
            lab.groupby_bins(
                "x",
                &[10.0, 30.0, 60.0],
                &BinOptions::default().right(false),
            )
            .unwrap()
            .sum()
            .unwrap(),
            Some("x_bins"),
        ),
        "bins_lowest" => (
            lab.groupby_bins(
                "x",
                &[10.0, 30.0, 60.0],
                &BinOptions::default().include_lowest(true),
            )
            .unwrap()
            .sum()
            .unwrap(),
            Some("x_bins"),
        ),
        "month_mean" => (
            g.groupby("time.month").unwrap().mean().unwrap(),
            Some("month"),
        ),
        "season_mean" => (
            g.groupby("time.season").unwrap().mean().unwrap(),
            Some("season"),
        ),
        "year_sum" => (g.groupby("time.year").unwrap().sum().unwrap(), Some("year")),
        "noleap_month" => (
            noleap().groupby("time.month").unwrap().mean().unwrap(),
            Some("month"),
        ),
        "anomaly" => {
            let clim = g.groupby("time.season").unwrap().mean().unwrap();
            (
                (&g.groupby("time.season").unwrap() - &clim).unwrap(),
                Some("season"),
            )
        }
        "map_anomaly" => (
            g.groupby("time.month")
                .unwrap()
                .map(|x| &x - &x.mean("time")?)
                .unwrap(),
            None,
        ),
        "map_reduce" => (
            g.groupby("time.year")
                .unwrap()
                .map(|x| x.max("time"))
                .unwrap(),
            Some("year"),
        ),
        _ => panic!("unknown case {case}"),
    }
}

#[test]
fn matches_xarray() {
    let table = include_str!("../../../fixtures/groupby.txt");
    let mut checked = 0;
    for line in table.lines() {
        let f: Vec<&str> = line.split('|').collect();
        let (out, label) = compute(f[0]);
        let dims: Vec<&str> = out.dims().iter().map(Name::as_str).collect();
        assert_eq!(dims.join(","), f[1], "{line}");
        if let Some(label) = label {
            let got = text(&out.coord(label).unwrap().typed().unwrap());
            if f[0] == "bins_lowest" {
                // pandas shows the widened first bin as "(9.999, 30.0]".
                assert_eq!(got, "[10, 30] (30, 60]", "{line}");
            } else {
                assert_eq!(got, f[2], "{line}");
            }
        }
        assert_eq!(text(&out.variable().typed().unwrap()), f[3], "{line}");
        checked += 1;
    }
    assert_eq!(checked, 17);
}

#[test]
fn nat_labels_and_datasets() {
    // NaT belongs to no group.
    let mut t = months(4);
    t[1] = NaiveDateTime::NAT;
    let da = DataArray::from_array(["time"], array![1.0, 2.0, 3.0, 4.0].into_dyn())
        .unwrap()
        .with_coord("t", Variable::new(["time"], t.into_dyn()).unwrap())
        .unwrap();
    let m = da.groupby("t.month").unwrap();
    assert_eq!(m.len(), 3);
    assert_eq!(m.indices(), [vec![0], vec![2], vec![3]]);

    // Dataset: variables without the grouped dimension are kept as they are (xarray repeats
    // them per group); numeric reductions drop strings.
    let ds = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![10i64, 20, 30].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_coord(
            "lab",
            Variable::new(
                ["x"],
                array!["a".to_owned(), "b".into(), "a".into()].into_dyn(),
            )
            .unwrap(),
        )
        .unwrap()
        .with_var(
            "v",
            Variable::new(["x"], array![1.0, 2.0, 3.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var(
            "c",
            Variable::new(["y"], array![7.0, 8.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var(
            "s",
            Variable::new(
                ["x"],
                array!["p".to_owned(), "q".into(), "r".into()].into_dyn(),
            )
            .unwrap(),
        )
        .unwrap();
    let r = ds.groupby("lab").unwrap().mean().unwrap();
    assert_eq!(
        r.data_vars()["v"]
            .data::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [2.0, 2.0]
    );
    assert_eq!(r.data_vars()["c"].dims(), ["y"]);
    assert!(!r.contains("s"));
    let first = ds.groupby("lab").unwrap().first().unwrap();
    assert_eq!(
        first.data_vars()["s"]
            .data::<String>()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        ["p", "q"]
    );

    // Errors.
    assert!(matches!(
        da.groupby("nope"),
        Err(CoreError::NoSuchVariable(_))
    ));
    assert!(da.groupby("t.fortnight").is_err());
    assert!(
        lab()
            .groupby_bins("x", &[5.0], &BinOptions::default())
            .is_err()
    );
    assert!(
        lab()
            .groupby_bins("x", &[5.0, 1.0], &BinOptions::default())
            .is_err()
    );
}

#[test]
fn dt_accessor() {
    let mut t = months(3);
    t[2] = NaiveDateTime::NAT;
    let v = Variable::new(["time"], t.into_dyn()).unwrap();
    let m = v.dt("month").unwrap();
    let m: Vec<f64> = m.data::<f64>().unwrap().iter().copied().collect();
    assert_eq!(m[..2], [1.0, 2.0]);
    assert!(m[2].is_nan());
    assert_eq!(
        v.dt("season")
            .unwrap()
            .data::<String>()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        ["DJF", "DJF", ""]
    );
    let v = Variable::new(["time"], months(2).into_dyn()).unwrap();
    // 2000-01-01 was a Saturday.
    assert_eq!(
        v.dt("dayofweek")
            .unwrap()
            .data::<i64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [5, 1]
    );
    assert_eq!(
        v.dt("dayofyear")
            .unwrap()
            .data::<i64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [1, 32]
    );
}

/// xarray puts a data array's group dimension where the grouped dimension was (datasets put
/// it first).
#[test]
fn group_dim_keeps_its_position() {
    let data = ArrayD::from_shape_fn(IxDyn(&[2, 3]), |ix| (ix[0] * 3 + ix[1]) as f64);
    let da = DataArray::from_array(["y", "x"], data)
        .unwrap()
        .with_coord(
            "lab",
            Variable::new(
                ["x"],
                array!["a", "b", "a"].map(|s| s.to_string()).into_dyn(),
            )
            .unwrap(),
        )
        .unwrap();
    let out = da.groupby("lab").unwrap().sum().unwrap();
    assert_eq!(out.dims(), [Name::from("y"), Name::from("lab")]);
    assert_eq!(
        out.data::<f64>().unwrap(),
        array![[2.0, 1.0], [8.0, 4.0]].into_dyn()
    );
    let first = da.groupby("lab").unwrap().first().unwrap();
    assert_eq!(first.dims(), [Name::from("y"), Name::from("lab")]);
    let mut ds = Dataset::new();
    ds.add_array("v", da).unwrap();
    let out = ds.groupby("lab").unwrap().sum().unwrap();
    assert_eq!(
        out.data_vars()["v"].dims(),
        [Name::from("lab"), Name::from("y")]
    );
}

#[test]
fn transpose() {
    let data = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |ix| {
        (ix[0] * 100 + ix[1] * 10 + ix[2]) as f64
    });
    let da = DataArray::from_array(["a", "b", "c"], data)
        .unwrap()
        .with_coord(
            "ab",
            Variable::new(
                ["a", "b"],
                ArrayD::from_shape_fn(IxDyn(&[2, 3]), |ix| (ix[0] * 10 + ix[1]) as i64),
            )
            .unwrap(),
        )
        .unwrap();
    let t = da.transpose(&["c", "a", "b"]).unwrap();
    assert_eq!(
        t.dims(),
        [Name::from("c"), Name::from("a"), Name::from("b")]
    );
    assert_eq!(t.data::<f64>().unwrap()[[3, 1, 2]], 123.0);
    let r = da.transpose(&[]).unwrap();
    assert_eq!(
        r.dims(),
        [Name::from("c"), Name::from("b"), Name::from("a")]
    );
    assert_eq!(r.coords()["ab"].dims(), [Name::from("b"), Name::from("a")]);
    assert_eq!(r.coords()["ab"].data::<i64>().unwrap()[[2, 1]], 12);
    let e = da.transpose(&["b", "..."]).unwrap();
    assert_eq!(
        e.dims(),
        [Name::from("b"), Name::from("a"), Name::from("c")]
    );
    assert!(matches!(
        da.transpose(&["a", "b"]),
        Err(CoreError::DimCountMismatch { .. })
    ));
    assert!(matches!(
        da.transpose(&["a", "b", "z"]),
        Err(CoreError::NoSuchDim(_))
    ));
    assert!(matches!(
        da.transpose(&["a", "a", "b"]),
        Err(CoreError::DuplicateDim(_))
    ));
    let mut ds = Dataset::new();
    ds.add_array("v", da).unwrap();
    let t = ds.transpose(&["c", "..."]).unwrap();
    assert_eq!(
        t.data_vars()["v"].dims(),
        [Name::from("c"), Name::from("a"), Name::from("b")]
    );
}
