//! Comparisons and logical operations (numpy semantics, with units).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::{Calendar, CfDate, Unit};
use annoray::{CmpOp, DType, DataArray, LogicOp};
use ndarray::{arr0, array};

fn bools(d: &DataArray) -> Vec<bool> {
    assert_eq!(d.variable().dtype(), DType::Bool);
    d.data::<bool>().unwrap().iter().copied().collect()
}

fn u(s: &str) -> Unit {
    s.parse().unwrap()
}

fn num(v: ndarray::Array1<f64>) -> DataArray {
    DataArray::from_array(["x"], v.into_dyn()).unwrap()
}

fn scalar(v: f64) -> DataArray {
    DataArray::from_array(Vec::<&str>::new(), arr0(v).into_dyn()).unwrap()
}

fn quantity(v: f64, unit: &str) -> DataArray {
    scalar(v).with_unit(u(unit))
}

#[test]
fn every_operator_with_nan() {
    let a = num(array![1.0, 2.0, f64::NAN, 4.0]);
    let two = scalar(2.0);
    assert_eq!(
        bools(&a.compare(CmpOp::Lt, &two).unwrap()),
        [true, false, false, false]
    );
    assert_eq!(
        bools(&a.compare(CmpOp::Le, &two).unwrap()),
        [true, true, false, false]
    );
    assert_eq!(
        bools(&a.compare(CmpOp::Gt, &two).unwrap()),
        [false, false, false, true]
    );
    assert_eq!(
        bools(&a.compare(CmpOp::Ge, &two).unwrap()),
        [false, true, false, true]
    );
    assert_eq!(
        bools(&a.compare(CmpOp::Eq, &two).unwrap()),
        [false, true, false, false]
    );
    // NaN != anything, NaN included.
    assert_eq!(
        bools(&a.compare(CmpOp::Ne, &two).unwrap()),
        [true, false, true, true]
    );
    assert_eq!(
        bools(&a.compare(CmpOp::Ne, &a).unwrap()),
        [false, false, true, false]
    );
    // Integers compare exactly, beyond f64 precision.
    let big = DataArray::from_array(["x"], array![i64::MAX - 1, i64::MAX].into_dyn()).unwrap();
    let edge = DataArray::from_array(Vec::<&str>::new(), arr0(i64::MAX - 1).into_dyn()).unwrap();
    assert_eq!(
        bools(&big.compare(CmpOp::Gt, &edge).unwrap()),
        [false, true]
    );
}

#[test]
fn units_convert_and_bare_zero_only() {
    let len = num(array![500.0, 1500.0]).with_unit(u("m"));
    assert_eq!(
        bools(&len.compare(CmpOp::Gt, &quantity(1.0, "km")).unwrap()),
        [false, true]
    );
    // A bare 0 against a ratio-scale unit; any other bare number is refused.
    let flux = num(array![-1.0, 2.0]).with_unit(u("W m-2"));
    assert_eq!(
        bools(&flux.compare(CmpOp::Gt, &scalar(0.0)).unwrap()),
        [false, true]
    );
    assert_eq!(
        bools(&scalar(0.0).compare(CmpOp::Lt, &flux).unwrap()),
        [false, true]
    );
    assert!(flux.compare(CmpOp::Gt, &scalar(5.0)).is_err());
    assert!(len.compare(CmpOp::Gt, &quantity(1.0, "s")).is_err());
    // Not for °C: 0 °C is not 0 °F.
    let t = num(array![-5.0, 5.0]).with_unit(u("degC"));
    assert!(t.compare(CmpOp::Gt, &scalar(0.0)).is_err());
    assert_eq!(
        bools(&t.compare(CmpOp::Gt, &quantity(273.15, "K")).unwrap()),
        [false, true]
    );
    assert_eq!(
        bools(&t.compare(CmpOp::Gt, &quantity(32.0, "degF")).unwrap()),
        [false, true]
    );
    // Logarithmic units: 0 dB is 0 in any of them.
    let gain = num(array![-3.0, 3.0]).with_unit(u("dB"));
    assert_eq!(
        bools(&gain.compare(CmpOp::Ge, &scalar(0.0)).unwrap()),
        [false, true]
    );
    assert_eq!(
        len.compare(CmpOp::Gt, &quantity(1.0, "km")).unwrap().unit(),
        None
    );
}

#[test]
fn text_dates_and_durations() {
    let s =
        DataArray::from_array(["x"], array!["a".to_owned(), "b".to_owned()].into_dyn()).unwrap();
    let b = DataArray::from_array(Vec::<&str>::new(), arr0("b".to_owned()).into_dyn()).unwrap();
    assert_eq!(bools(&s.compare(CmpOp::Eq, &b).unwrap()), [false, true]);
    assert_eq!(bools(&s.compare(CmpOp::Lt, &b).unwrap()), [true, false]);
    assert!(s.compare(CmpOp::Eq, &scalar(1.0)).is_err());

    let d = |y| annoray::Instant::from_ymd(y, 1, 1).unwrap();
    let nat = annoray::Instant::NAT;
    let t = DataArray::from_array(["t"], array![d(2000), nat, d(2002)].into_dyn()).unwrap();
    let mid = DataArray::from_array(Vec::<&str>::new(), arr0(d(2001)).into_dyn()).unwrap();
    assert_eq!(
        bools(&t.compare(CmpOp::Gt, &mid).unwrap()),
        [false, false, true]
    );
    assert_eq!(
        bools(&t.compare(CmpOp::Ne, &mid).unwrap()),
        [true, true, true]
    );

    let cf = |y| CfDate::from_ymd(Calendar::Day360, y, 2, 30).unwrap();
    let c = DataArray::from_array(["t"], array![cf(2000), cf(2002)].into_dyn()).unwrap();
    let cm = DataArray::from_array(Vec::<&str>::new(), arr0(cf(2001)).into_dyn()).unwrap();
    assert_eq!(bools(&c.compare(CmpOp::Lt, &cm).unwrap()), [true, false]);
    let other = DataArray::from_array(
        Vec::<&str>::new(),
        arr0(CfDate::from_ymd(Calendar::NoLeap, 2001, 1, 1).unwrap()).into_dyn(),
    )
    .unwrap();
    assert!(c.compare(CmpOp::Lt, &other).is_err());
}

#[test]
fn alignment_broadcasting_and_coordinates() {
    let a = num(array![1.0, 2.0, 3.0])
        .with_dim_coord("x", array![10.0, 20.0, 30.0])
        .unwrap();
    let b = num(array![2.0, 2.0])
        .with_dim_coord("x", array![20.0, 30.0])
        .unwrap();
    let r = a.compare(CmpOp::Ge, &b).unwrap();
    assert_eq!(bools(&r), [true, true]);
    assert_eq!(
        r.coords()["x"]
            .to_f64()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [20.0, 30.0]
    );
    let col = DataArray::from_array(["y"], array![1.5, 2.5].into_dyn()).unwrap();
    let grid = num(array![1.0, 2.0, 3.0]).compare(CmpOp::Gt, &col).unwrap();
    assert_eq!(grid.dims(), ["x", "y"]);
    assert_eq!(bools(&grid), [false, false, true, false, true, true]);
}

#[test]
fn logical_operations() {
    let p = DataArray::from_array(["x"], array![true, true, false, false].into_dyn()).unwrap();
    let q = DataArray::from_array(["x"], array![true, false, true, false].into_dyn()).unwrap();
    assert_eq!(
        bools(&p.logical(LogicOp::And, &q).unwrap()),
        [true, false, false, false]
    );
    assert_eq!(
        bools(&p.logical(LogicOp::Or, &q).unwrap()),
        [true, true, true, false]
    );
    assert_eq!(
        bools(&p.logical(LogicOp::Xor, &q).unwrap()),
        [false, true, true, false]
    );
    assert_eq!(bools(&p.logical_not().unwrap()), [false, false, true, true]);
    assert!(num(array![1.0]).logical_not().is_err());
    // Typical mask: a range test.
    let v = num(array![0.5, 1.5, 2.5]);
    let mask = v
        .compare(CmpOp::Gt, &scalar(1.0))
        .unwrap()
        .logical(LogicOp::And, &v.compare(CmpOp::Lt, &scalar(2.0)).unwrap())
        .unwrap();
    assert_eq!(bools(&mask), [false, true, false]);
}

#[test]
fn lazy_data_against_a_scalar_stays_lazy() {
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
    let unit = lazy.unit().cloned();
    let threshold = match &unit {
        Some(u) => scalar(0.0).with_unit(u.clone()),
        None => scalar(0.0),
    };
    let l = lazy.compare(CmpOp::Gt, &threshold).unwrap();
    assert!(l.variable().is_lazy());
    let e = eager.compare(CmpOp::Gt, &threshold).unwrap();
    assert_eq!(bools(&l), bools(&e));
}

#[test]
fn methods_and_operators() -> annoray::Result<()> {
    let v = num(array![0.5, 1.5, 2.5, f64::NAN]);
    // Comparison methods take arrays, plain numbers and quantities.
    assert_eq!(bools(&v.gt(1.0)?), [false, true, true, false]);
    assert_eq!(bools(&v.le(&scalar(1.5))?), [true, true, false, false]);
    assert_eq!(bools(&v.eq_elem(2.5)?), [false, false, true, false]);
    assert_eq!(bools(&v.ne_elem(&v)?), [false, false, false, true]);
    let len = num(array![500.0, 1500.0]).with_unit(u("m"));
    let km: annoray::units::Quantity = annoray::units::Quantity::new(1.0, "km")?;
    assert_eq!(bools(&len.ge(&km)?), [false, true]);
    assert!(len.lt(3.0).is_err(), "a bare nonzero number against metres");
    // Logical operators, chained as arithmetic is.
    let mask = (&v.gt(1.0)? & v.lt(2.0))?;
    assert_eq!(bools(&mask), [false, true, false, false]);
    assert_eq!(
        bools(&(&v.lt(1.0)? | v.gt(2.0))?),
        [true, false, true, false]
    );
    assert_eq!(bools(&(&mask ^ &mask)?), [false; 4]);
    assert_eq!(bools(&(!&mask)?), [true, false, true, true]);
    assert!((!&v).is_err(), "not booleans");
    // A typical use: mask, then fill.
    let kept = v.where_(&mask)?;
    assert!(kept.to_f64()?[0].is_nan());
    assert_eq!(kept.to_f64()?[1], 1.5);
    Ok(())
}

#[test]
fn cross_unit_comparisons_read_as_the_decimals() {
    let km = |v: f64| scalar(v).with_unit(u("km"));
    let m = |v: f64| scalar(v).with_unit(u("m"));
    assert_eq!(
        bools(&km(1.2).compare(CmpOp::Gt, &m(1000.0)).unwrap()),
        [true]
    );
    assert_eq!(
        bools(&m(1000.0).compare(CmpOp::Lt, &km(1.2)).unwrap()),
        [true]
    );
    assert_eq!(
        bools(&km(1.2).compare(CmpOp::Gt, &m(1200.0)).unwrap()),
        [false]
    );
    // Equality holds as written, whichever side has the larger unit.
    for i in 1..2000_i32 {
        let (a, b) = (km(f64::from(i) / 10.0), m(f64::from(i) * 100.0));
        assert_eq!(
            bools(&a.compare(CmpOp::Eq, &b).unwrap()),
            [true],
            "{i}/10 km"
        );
        assert_eq!(bools(&b.compare(CmpOp::Eq, &a).unwrap()), [true], "{i}00 m");
        assert_eq!(bools(&b.compare(CmpOp::Le, &a).unwrap()), [true], "{i}00 m");
    }
}
