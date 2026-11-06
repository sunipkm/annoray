//! Values are shared between clones and copied on write; dimension-named variables are indexes.
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::sync::Arc;

use annoray::ndarray::{ArrayD, ArrayViewD, IxDyn, array};
use annoray::{DataArray, Dataset, ExternalBuffer, InterpMethod, TypedArray, Unit, Variable, View};

fn var() -> Variable {
    Variable::new(["x"], array![1.0f64, 2.0, 3.0].into_dyn()).unwrap()
}

fn pointer(v: &Variable) -> *const f64 {
    v.data::<f64>().unwrap().as_ptr()
}

#[test]
fn clones_share_until_written() {
    let a = var();
    let mut b = a.clone();
    assert_eq!(pointer(&a), pointer(&b));
    b.data_mut::<f64>().unwrap()[[0]] = 9.0;
    assert_ne!(pointer(&a), pointer(&b));
    assert_eq!(a.data::<f64>().unwrap()[[0]], 1.0);
    assert_eq!(b.data::<f64>().unwrap()[[0]], 9.0);
}

#[test]
fn unshared_writes_do_not_copy() {
    let mut a = var();
    let before = pointer(&a);
    a.data_mut::<f64>().unwrap()[[0]] = 9.0;
    assert_eq!(pointer(&a), before);
}

#[test]
fn isel_mut_copies_shared_values_too() {
    let a = var();
    let mut b = a.clone();
    let mut slice = b.isel_mut([("x", 0)]).unwrap();
    slice.data_mut::<f64>().unwrap()[IxDyn(&[])] = 7.0;
    assert_eq!(a.data::<f64>().unwrap()[[0]], 1.0);
    assert_eq!(b.data::<f64>().unwrap()[[0]], 7.0);
}

#[test]
fn to_owned_of_a_whole_view_shares_and_of_a_selection_copies() {
    let a = var();
    assert_eq!(pointer(&a.view().to_owned()), pointer(&a));
    assert_eq!(pointer(&a.to_owned()), pointer(&a));
    let part = annoray::isel!(a[x = 1:]).unwrap().to_owned();
    assert_eq!(
        part.data::<f64>().unwrap().as_slice(),
        Some(&[2.0, 3.0][..])
    );
    assert_ne!(pointer(&part), pointer(&a));
}

#[test]
fn deep_copies_own_their_values() {
    let a = var();
    assert_ne!(pointer(&a.deep_copy()), pointer(&a));
    let ds = Dataset::new().with_var("v", a).unwrap();
    assert_ne!(
        pointer(ds.deep_copy().data_vars().get("v").unwrap()),
        pointer(ds.data_vars().get("v").unwrap())
    );
    assert_eq!(
        pointer(ds.clone().data_vars().get("v").unwrap()),
        pointer(ds.data_vars().get("v").unwrap())
    );
    let da = DataArray::new(var());
    assert_ne!(pointer(da.deep_copy().variable()), pointer(da.variable()));
}

#[test]
fn dataset_children_and_untouched_variables_are_shared() {
    let other = Variable::new(["y"], array![1.0f64, 2.0].into_dyn()).unwrap();
    let ds = Dataset::new()
        .with_var("v", var())
        .unwrap()
        .with_var("w", other)
        .unwrap();
    let held = pointer(ds.data_vars().get("v").unwrap());
    assert_eq!(pointer(ds.get("v").unwrap().to_owned().variable()), held);
    // Selecting along another dimension leaves `v` untouched: shared, not copied.
    let picked = ds.isel([("y", 0)]).unwrap().to_owned();
    assert_eq!(pointer(picked.data_vars().get("v").unwrap()), held);
}

#[test]
fn keep_alive_outlives_the_variable_and_stops_writes_in_place() {
    let mut a = var();
    let before = pointer(&a);
    let keep = a.keep_alive();
    a.data_mut::<f64>().unwrap()[[0]] = 9.0;
    assert_ne!(pointer(&a), before);
    drop(a);
    let kept = keep.downcast::<TypedArray<annoray::Owned>>().unwrap();
    assert!(matches!(&*kept, TypedArray::F64(x) if x[[0]] == 1.0));
}

/// A buffer that is not annoray's.
struct Foreign(ArrayD<f64>);

impl ExternalBuffer for Foreign {
    fn typed(&self) -> TypedArray<View<'_>> {
        let v: ArrayViewD<'_, f64> = self.0.view();
        TypedArray::F64(v)
    }
}

#[test]
fn external_buffers_read_in_place_and_copy_on_write() {
    let source = array![[1.0f64, 2.0], [3.0, 4.0]].into_dyn();
    let at = source.as_ptr();
    let mut v = Variable::from_external(["y", "x"], Arc::new(Foreign(source))).unwrap();
    assert_eq!(v.shape(), &[2, 2]);
    assert_eq!(v.dtype(), annoray::DType::F64);
    assert_eq!(pointer(&v), at);
    let shared = v.clone();
    assert_eq!(pointer(&shared), at);
    assert_eq!(v.isel([("y", 1)]).unwrap().data::<f64>().unwrap()[[0]], 3.0);
    assert_ne!(pointer(&v.deep_copy()), at);
    v.data_mut::<f64>().unwrap()[[0, 0]] = 9.0;
    assert_ne!(pointer(&v), at);
    assert_eq!(v.data::<f64>().unwrap()[[0, 0]], 9.0);
    assert_eq!(shared.data::<f64>().unwrap()[[0, 0]], 1.0);
    assert_eq!(pointer(&shared), at);
    let t = v.clone().into_typed().unwrap();
    assert!(matches!(t, TypedArray::F64(x) if x[[0, 0]] == 9.0));
}

#[test]
fn external_buffers_must_match_their_dimensions() {
    let source = array![1.0f64].into_dyn();
    assert!(Variable::from_external(["y", "x"], Arc::new(Foreign(source))).is_err());
}

#[test]
fn a_variable_named_after_its_dimension_is_its_coordinate() {
    let mut ds = Dataset::new();
    ds.add_var("x", var()).unwrap();
    assert!(ds.coords().contains_key("x"));
    assert!(ds.data_vars().is_empty());
    ds.add_var("v", var()).unwrap();
    assert!(ds.data_vars().contains_key("v"));
    assert!(ds.add_var("x", var()).is_err());
    // A data variable of that name, replaced by the index.
    let mut ds = Dataset::new();
    ds.add_var(
        "q",
        Variable::new(["y"], array![1.0f64].into_dyn()).unwrap(),
    )
    .unwrap();
    ds.set_array(
        "q",
        DataArray::new(Variable::new(["q"], array![1.0f64, 2.0].into_dyn()).unwrap()),
    )
    .unwrap();
    assert!(ds.coords().contains_key("q"));
    assert!(!ds.data_vars().contains_key("q"));
}

#[test]
fn set_unit_of_changes_the_unit_only() {
    let mut ds = Dataset::new().with_var("v", var()).unwrap();
    let km: Unit = "km".parse().unwrap();
    assert!(ds.set_unit_of("v", Some(km.clone())).unwrap().is_none());
    assert_eq!(ds.data_vars().get("v").unwrap().unit(), Some(&km));
    assert!(ds.set_unit_of("nothing", None).is_err());
}

#[test]
fn interp_at_a_scalar_loses_the_dimension_and_keeps_a_scalar_coordinate() {
    let a = DataArray::new(var()).with_coord("x", var()).unwrap();
    let at = Variable::new(Vec::<&str>::new(), ndarray_scalar(1.5)).unwrap();
    let out = a.interp("x", &at, InterpMethod::Linear).unwrap();
    assert_eq!(out.ndim(), 0);
    assert_eq!(out.variable().data::<f64>().unwrap()[IxDyn(&[])], 1.5);
    assert_eq!(out.coords().get("x").unwrap().ndim(), 0);
    let ds = Dataset::new()
        .with_coord("x", var())
        .unwrap()
        .with_var("v", var())
        .unwrap();
    let out = ds.interp("x", &at, InterpMethod::Linear).unwrap();
    assert_eq!(out.data_vars().get("v").unwrap().ndim(), 0);
    assert_eq!(out.coords().get("x").unwrap().ndim(), 0);
}

fn ndarray_scalar(x: f64) -> ArrayD<f64> {
    ArrayD::from_elem(IxDyn(&[]), x)
}

#[test]
fn boolean_mask_positions() {
    let x = Variable::new(["x"], array![0.0f64, 1.0, 2.0].into_dyn()).unwrap();
    let a = DataArray::new(var()).with_coord("x", x.clone()).unwrap();
    let mask = DataArray::new(Variable::new(["x"], array![true, false, true].into_dyn()).unwrap())
        .with_coord("x", x)
        .unwrap();
    assert_eq!(a.mask_positions("x", &mask).unwrap(), vec![0, 2]);
    let wrong = Variable::new(["x"], array![5.0f64, 6.0, 7.0].into_dyn()).unwrap();
    assert!(
        a.mask_positions("x", &mask.clone().with_coord("x", wrong).unwrap())
            .is_err()
    );
    let short = DataArray::new(Variable::new(["x"], array![true].into_dyn()).unwrap());
    assert!(a.mask_positions("x", &short).is_err());
}
