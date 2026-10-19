//! clip, round, isin, dot (expected values as numpy and xarray give them).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::Unit;
use annoray::{ClipBound, DType, DataArray, Dataset, Label, Variable, dot};
use ndarray::array;

const NAN: f64 = f64::NAN;

fn vals(d: &DataArray) -> Vec<f64> {
    d.to_f64().unwrap().iter().copied().collect()
}

fn close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        let ok = (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * w.abs().max(1.0);
        assert!(ok, "{got:?} vs {want:?}");
    }
}

fn unit(s: &str) -> Unit {
    s.parse().unwrap()
}

/// `v` in `a`'s unit, as an `isin` label.
fn label(a: &DataArray, v: f64) -> Label {
    match a.unit() {
        Some(u) => Label::Quantity(v, u.clone()),
        None => Label::Float(v),
    }
}

/// `v` in `a`'s unit, as a clip bound.
fn bound(a: &DataArray, v: f64) -> Option<ClipBound> {
    Some(match a.unit() {
        Some(u) => ClipBound::Quantity(v, u.clone()),
        None => v.into(),
    })
}

fn da(v: ndarray::Array1<f64>) -> DataArray {
    DataArray::from_array(["x"], v.into_dyn()).unwrap()
}

#[test]
fn clip_scalars_types_and_units() {
    let a = da(array![1.0, NAN, 5.0, -3.0]).with_name("v");
    let c = a.clip(Some(0.0.into()), Some(4.0.into())).unwrap();
    close(&vals(&c), &[1.0, NAN, 4.0, 0.0]);
    assert_eq!(c.name().unwrap().as_str(), "v");
    close(
        &vals(&a.clip(None, Some(2.0.into())).unwrap()),
        &[1.0, NAN, 2.0, -3.0],
    );
    // min > max: max wins, as numpy.
    close(
        &vals(&a.clip(Some(4.0.into()), Some(2.0.into())).unwrap()),
        &[2.0, NAN, 2.0, 2.0],
    );
    // NaN bound gives NaN.
    close(&vals(&a.clip(Some(NAN.into()), None).unwrap()), &[NAN; 4]);

    let ints = DataArray::from_array(["x"], array![1i32, 5, 9].into_dyn()).unwrap();
    let ci = ints.clip(Some(2.0.into()), Some(6.0.into())).unwrap();
    assert_eq!(ci.variable().dtype(), DType::I32);
    close(&vals(&ci), &[2.0, 5.0, 6.0]);
    let cf = ints.clip(Some(2.5.into()), None).unwrap();
    assert_eq!(cf.variable().dtype(), DType::F64);
    close(&vals(&cf), &[2.5, 5.0, 9.0]);
    let f32s = DataArray::from_array(["x"], array![1.0f32, 5.0].into_dyn()).unwrap();
    assert_eq!(
        f32s.clip(Some(2.0.into()), None)
            .unwrap()
            .variable()
            .dtype(),
        DType::F32
    );

    // °C data clipped by bounds in K and °F (absolute conversions).
    let t = da(array![10.0, 30.0, -25.0]).with_unit(unit("degC"));
    let ct = t
        .clip(
            Some(ClipBound::Quantity(-4.0, unit("degF"))),
            Some(ClipBound::Quantity(293.15, unit("K"))),
        )
        .unwrap();
    close(&vals(&ct), &[10.0, 20.0, -20.0]);
    assert_eq!(ct.unit(), Some(&unit("degC")));
    assert!(
        t.clip(Some(ClipBound::Quantity(1.0, unit("m"))), None)
            .is_err()
    );
    // Bare numbers beside a unit: only NaN, or 0 for a ratio-scale unit.
    assert!(t.clip(Some(5.0.into()), None).is_err());
    assert!(t.clip(Some(0.0.into()), None).is_err()); // °C is not ratio-scale
    close(&vals(&t.clip(Some(NAN.into()), None).unwrap()), &[NAN; 3]);
    let m = da(array![-1.0, 2.0]).with_unit(unit("m"));
    close(&vals(&m.clip(Some(0.0.into()), None).unwrap()), &[0.0, 2.0]);
    assert!(m.clip(None, Some(1.0.into())).is_err());
}

#[test]
fn clip_by_arrays() {
    let a = DataArray::from_array(["y", "x"], array![[1.0, 2000.0], [3000.0, 4.0]].into_dyn())
        .unwrap()
        .with_unit(unit("m"));
    let hi = da(array![1.5, 2.5]).with_unit(unit("km"));
    let c = a.clip(None, Some(hi.into())).unwrap();
    close(&vals(&c), &[1.0, 2000.0, 1500.0, 4.0]);
    assert_eq!(c.unit(), Some(&unit("m")));
    // A bare-number array against data with a unit is refused.
    assert!(a.clip(None, Some(da(array![1.0, 2.0]).into())).is_err());
}

#[test]
fn round_half_to_even() {
    let a = da(array![0.5, 1.5, 2.5, -0.5, 1.2345, NAN]).with_unit(unit("m"));
    close(
        &vals(&a.round(0).unwrap()),
        &[0.0, 2.0, 2.0, -0.0, 1.0, NAN],
    );
    close(
        &vals(&a.round(2).unwrap()),
        &[0.5, 1.5, 2.5, -0.5, 1.23, NAN],
    );
    assert_eq!(a.round(1).unwrap().unit(), Some(&unit("m")));
    close(
        &vals(&da(array![15.0, 25.0, 14.0]).round(-1).unwrap()),
        &[20.0, 20.0, 10.0],
    );
    let ints = DataArray::from_array(["x"], array![15i32, 25, -15, 14].into_dyn()).unwrap();
    let r = ints.round(-1).unwrap();
    assert_eq!(r.variable().dtype(), DType::I32);
    close(&vals(&r), &[20.0, 20.0, -20.0, 10.0]);
    assert_eq!(vals(&ints.round(2).unwrap()), vals(&ints));
    let tiny = DataArray::from_array(["x"], array![120u8].into_dyn()).unwrap();
    assert!(tiny.round(-3).is_ok());
    // 126 rounds to 130, which an i8 cannot hold.
    let edge = DataArray::from_array(["x"], array![126i8].into_dyn()).unwrap();
    assert!(edge.round(-1).is_err());
}

#[test]
fn isin_compares_in_the_data_unit() {
    let a = da(array![1.0, 2.0, NAN, 3.0]).with_unit(unit("m"));
    let got = a
        .isin(&[
            Label::quantity(2.0, "m").unwrap(),
            Label::Float(NAN),
            Label::quantity(0.003, "km").unwrap(),
        ])
        .unwrap();
    assert_eq!(got.variable().dtype(), DType::Bool);
    assert_eq!(got.unit(), None);
    close(&vals(&got), &[0.0, 1.0, 0.0, 1.0]);
    // Bare numbers beside a unit: only NaN, or 0 for a ratio-scale unit.
    assert!(a.isin(&[Label::Float(2.0)]).is_err());
    assert!(a.isin(&[Label::Int(2)]).is_err());
    close(&vals(&a.isin(&[Label::Float(0.0)]).unwrap()), &[0.0; 4]);
    let c = da(array![0.0, 5.0]).with_unit(unit("degC"));
    assert!(c.isin(&[Label::Float(0.0)]).is_err());
    assert!(
        da(array![1.0])
            .isin(&[Label::quantity(1.0, "m").unwrap()])
            .is_err()
    );
    assert!(a.isin(&[Label::quantity(1.0, "s").unwrap()]).is_err());
    let s =
        DataArray::from_array(["x"], array!["a".to_owned(), "b".to_owned()].into_dyn()).unwrap();
    close(
        &vals(&s.isin(&[Label::Str("b".into()), Label::Int(1)]).unwrap()),
        &[0.0, 1.0],
    );
}

#[test]
fn dot_contracts_shared_dimensions() {
    let a = DataArray::from_array(["x", "y"], array![[1.0, 2.0], [3.0, 4.0]].into_dyn())
        .unwrap()
        .with_unit(unit("m"));
    let b = DataArray::from_array(
        ["y", "z"],
        array![[1.0, 0.0, 2.0], [0.5, 1.0, NAN]].into_dyn(),
    )
    .unwrap()
    .with_unit(unit("s"));
    let d = a.dot(&b, None).unwrap();
    assert_eq!(d.dims(), ["x", "z"]);
    close(&vals(&d), &[2.0, 2.0, NAN, 5.0, 4.0, NAN]);
    assert!(d.unit().unwrap().is_convertible(&unit("m s")));
    // Over every dimension.
    close(&vals(&a.dot(&a, Some(&["x", "y"])).unwrap()), &[30.0]);
    // Three operands: only dimensions shared by two or more are summed.
    let three = dot(
        &[
            &a,
            &a,
            &DataArray::from_array(["z"], array![1.0, 1.0].into_dyn()).unwrap(),
        ],
        None,
    );
    assert_eq!(three.unwrap().dims(), ["z"]);
    // Integers stay integers.
    let i = DataArray::from_array(["x"], array![1i64, 2, 3].into_dyn()).unwrap();
    let ii = i.dot(&i, None).unwrap();
    assert_eq!(ii.variable().dtype(), DType::I64);
    close(&vals(&ii), &[14.0]);
    assert!(a.dot(&b, Some(&["nope"])).is_err());
}

#[test]
fn datasets_and_laziness() {
    let mut ds = Dataset::new();
    ds.add_var(
        "a",
        Variable::new(["x"], array![0.25, 7.5].into_dyn()).unwrap(),
    )
    .unwrap();
    ds.add_var(
        "s",
        Variable::new(["x"], array!["p".to_owned(), "q".to_owned()].into_dyn()).unwrap(),
    )
    .unwrap();
    let r = ds.round(0).unwrap();
    close(&vals(&r.get("a").unwrap().to_owned()), &[0.0, 8.0]);
    let c = ds.clip(Some(1.0.into()), None).unwrap();
    close(&vals(&c.get("a").unwrap().to_owned()), &[1.0, 7.5]);
    let i = ds.isin(&[Label::Str("q".into())]).unwrap();
    close(&vals(&i.get("s").unwrap().to_owned()), &[0.0, 1.0]);

    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let lazy = annoray::io::open_dataset(&path)
        .unwrap()
        .get("temp")
        .unwrap()
        .to_owned();
    let mut all = annoray::io::open_dataset(&path).unwrap();
    all.load().unwrap();
    let eager = all.get("temp").unwrap().to_owned();
    for (l, e) in [
        (lazy.round(1).unwrap(), eager.round(1).unwrap()),
        (
            lazy.clip(bound(&lazy, 0.0), bound(&lazy, 10.0)).unwrap(),
            eager.clip(bound(&eager, 0.0), bound(&eager, 10.0)).unwrap(),
        ),
        (
            lazy.isin(&[label(&lazy, 0.0)]).unwrap(),
            eager.isin(&[label(&eager, 0.0)]).unwrap(),
        ),
    ] {
        assert!(l.variable().is_lazy());
        close(&vals(&l), &vals(&e));
    }
}
