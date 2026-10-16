//! Dataset arithmetic, comparisons and elementwise functions (xarray's dataset operators).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::Unit;
use annoray::{BinOp, CmpOp, DataArray, Dataset, MathFn, Variable};
use annoray_units::UnitPolicy;
use ndarray::array;

fn vals(ds: &Dataset, k: &str) -> Vec<f64> {
    ds.get(k)
        .unwrap()
        .to_f64()
        .unwrap()
        .iter()
        .copied()
        .collect()
}

fn u(s: &str) -> Unit {
    s.parse().unwrap()
}

/// `a` (m) along x, `b` (s) along x and y, a string variable, x coordinate 0..3.
fn sample() -> Dataset {
    let mut ds = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![0.0, 1.0, 2.0].into_dyn()).unwrap(),
        )
        .unwrap();
    ds.add_var(
        "a",
        Variable::new(["x"], array![1.0, 2.0, f64::NAN].into_dyn())
            .unwrap()
            .with_unit(u("m")),
    )
    .unwrap();
    ds.add_var(
        "b",
        Variable::new(
            ["x", "y"],
            array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn(),
        )
        .unwrap()
        .with_unit(u("s")),
    )
    .unwrap();
    ds.attrs_mut().insert("title".into(), "t".into());
    ds
}

#[test]
fn scalars_reach_every_variable() {
    let ds = sample();
    let twice = (&ds * 2.0).unwrap();
    assert_eq!(vals(&twice, "a")[..2], [2.0, 4.0]);
    assert_eq!(vals(&twice, "b"), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    assert_eq!(twice.get("a").unwrap().unit(), Some(&u("m")));
    assert_eq!(
        twice.attrs()["title"],
        "t".into(),
        "attributes kept, as xarray"
    );
    // Units: a bare number cannot be added to metres.
    assert!((&ds + 1.0).is_err());
    let neg = (-&ds).unwrap();
    assert_eq!(vals(&neg, "b")[0], -1.0);
    let inv = (1.0 / &ds).unwrap();
    assert!(
        inv.get("a")
            .unwrap()
            .unit()
            .unwrap()
            .is_convertible(&u("1/m"))
    );
}

#[test]
fn datasets_combine_common_variables_after_alignment() {
    let ds = sample();
    let mut other = Dataset::new()
        .with_coord(
            "x",
            Variable::new(["x"], array![1.0, 2.0].into_dyn()).unwrap(),
        )
        .unwrap();
    other
        .add_var(
            "a",
            Variable::new(["x"], array![10.0, 20.0].into_dyn())
                .unwrap()
                .with_unit(u("m")),
        )
        .unwrap();
    other
        .add_var(
            "c",
            Variable::new(["x"], array![0.0, 0.0].into_dyn()).unwrap(),
        )
        .unwrap();
    let sum = (&ds + &other).unwrap();
    // Only the common variable, on the common labels.
    assert_eq!(
        sum.data_vars()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["a"]
    );
    assert_eq!(vals(&sum, "a")[0], 12.0);
    assert!(vals(&sum, "a")[1].is_nan());
    assert_eq!(
        sum.coords()["x"]
            .to_f64()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [1.0, 2.0]
    );
    // With quantities in other units.
    let mut km = other.clone();
    km.add_var(
        "d",
        Variable::new(["x"], array![1.0, 1.0].into_dyn()).unwrap(),
    )
    .unwrap();
    let mixed = ds.binary(BinOp::Sub, &km, UnitPolicy::AutoRescale).unwrap();
    assert_eq!(vals(&mixed, "a")[0], -8.0);
}

#[test]
fn data_arrays_broadcast_against_every_variable() {
    let ds = sample();
    let scale = DataArray::from_array(["y"], array![1.0, 10.0].into_dyn()).unwrap();
    let r = (&ds * &scale).unwrap();
    assert_eq!(r.get("a").unwrap().dims(), ["x", "y"]);
    assert_eq!(vals(&r, "b"), [1.0, 20.0, 3.0, 40.0, 5.0, 60.0]);
}

#[test]
fn comparisons_masks_and_functions() {
    let ds = sample();
    let pos = ds
        .compare_array(
            CmpOp::Gt,
            &DataArray::from_array(Vec::<&str>::new(), ndarray::arr0(0.0).into_dyn()).unwrap(),
        )
        .unwrap();
    let a: Vec<bool> = pos
        .get("a")
        .unwrap()
        .data::<bool>()
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(a, [true, true, false]);
    let nulls = ds.isnull().unwrap();
    let na: Vec<bool> = nulls
        .get("a")
        .unwrap()
        .data::<bool>()
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(na, [false, false, true]);
    let masked = ds.where_dataset(&ds.notnull().unwrap()).unwrap();
    assert_eq!(vals(&masked, "b"), vals(&ds, "b"));
    let roots = ds.math(MathFn::Sqrt);
    assert!(roots.is_err(), "sqrt of metres is not a unit");
    let sq = ds.pow(2.0).unwrap();
    assert!(
        sq.get("b")
            .unwrap()
            .unit()
            .unwrap()
            .is_convertible(&u("s2"))
    );
    let mapped = ds
        .map_data_vars(true, |v| v.binary_scalar(BinOp::Mul, 3.0))
        .unwrap();
    assert_eq!(mapped.attrs()["title"], "t".into());
}

#[test]
fn lazy_datasets_stay_lazy() {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/xr_nc4.nc");
    let mut ds = annoray::io::open_dataset(&path).unwrap();
    // Text cannot be doubled (as in xarray): keep the numbers.
    let text: Vec<String> = ds
        .data_vars()
        .iter()
        .filter(|(_, v)| !v.dtype().to_string().starts_with("float"))
        .map(|(k, _)| k.to_string())
        .collect();
    for k in &text {
        ds.remove(k).unwrap();
    }
    assert!(ds.data_vars().values().any(|v| v.is_lazy()));
    let doubled = (&ds * 2.0).unwrap();
    for (k, v) in doubled.data_vars() {
        assert_eq!(v.is_lazy(), ds.data_vars()[k].is_lazy(), "{k}");
    }
}
