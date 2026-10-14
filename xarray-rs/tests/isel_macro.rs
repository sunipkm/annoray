//! Integration tests for the `isel!` macro.
#![allow(clippy::unwrap_used, clippy::arithmetic_side_effects)]

use xarray_rs::ndarray::{ArrayD, IxDyn};
use xarray_rs::{CoreError, IndexError, Variable, isel};

fn var() -> Variable {
    // time=2, lat=3, lon=4; value = 100*t + 10*y + x
    let a = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |ix| {
        (ix[0] * 100 + ix[1] * 10 + ix[2]) as i64
    });
    Variable::new(["time", "lat", "lon"], a).unwrap()
}

fn values(v: &xarray_rs::VariableView<'_>) -> Vec<i64> {
    v.data::<i64>().unwrap().iter().copied().collect()
}

#[test]
fn positional_forms() {
    let v = var();
    assert_eq!(values(&isel!(v[0, 1, ::-1]).unwrap()), vec![13, 12, 11, 10]);
    assert_eq!(values(&isel!(v[-1, -1, 1:3]).unwrap()), vec![121, 122]);
    assert_eq!(values(&isel!(v[.., 0, 0]).unwrap()), vec![0, 100]);
    assert_eq!(values(&isel!(v[1, ..., 3]).unwrap()), vec![103, 113, 123]);
    assert_eq!(isel!(v[]).unwrap().shape(), &[2, 3, 4]);
    assert_eq!(isel!(v[0,]).unwrap().shape(), &[3, 4]);
    // Rust ranges are accepted as well as Python slices.
    assert_eq!(isel!(v[0, 1..=2, ..2]).unwrap().shape(), &[2, 2]);
}

#[test]
fn named_forms() {
    let v = var();
    let s = isel!(v[lon = ::2, time = 1]).unwrap();
    assert_eq!(s.dims(), ["lat", "lon"]);
    assert_eq!(values(&s), vec![100, 102, 110, 112, 120, 122]);
    let s = isel!(v[lat = -2:, lon = 3]).unwrap();
    assert_eq!(s.dims(), ["time", "lat"]);
    assert_eq!(values(&s), vec![13, 23, 113, 123]);
}

#[test]
fn expressions_as_bounds() {
    let v = var();
    let i: usize = 1;
    let n = 2;
    let s = isel!(v[lon = i:i + n, lat = {v.len_of("lat").unwrap() - 1}]).unwrap();
    assert_eq!(s.dims(), ["time", "lon"]);
    assert_eq!(values(&s), vec![21, 22, 121, 122]);
}

#[test]
fn mutable_views_write_through() {
    let mut v = var();
    {
        let mut m = isel!(mut v[time = 1, lat = ::-1]).unwrap();
        let mut corner = isel!(mut m[0, -1]).unwrap();
        corner.data_mut::<i64>().unwrap().fill(-7);
    }
    assert_eq!(v.data::<i64>().unwrap()[[1, 2, 3]], -7);
}

#[test]
fn runtime_errors_are_results() {
    let v = var();
    assert!(matches!(
        isel!(v[time = 5]),
        Err(CoreError::Index(IndexError::OutOfBounds { .. }))
    ));
    assert_eq!(
        isel!(v[depth = 0]).unwrap_err(),
        CoreError::NoSuchDim("depth".into())
    );
    assert_eq!(
        isel!(v[::0]).unwrap_err(),
        CoreError::Index(IndexError::ZeroStep)
    );
    assert_eq!(
        isel!(v[{ u64::MAX }]).unwrap_err(),
        CoreError::Index(IndexError::Overflow)
    );
    assert_eq!(
        isel!(v[0, 0, 0, 0]).unwrap_err(),
        CoreError::Index(IndexError::TooManyIndexers { given: 4, ndim: 3 })
    );
}

#[test]
fn compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}

#[test]
fn receivers_auto_deref() {
    fn through_ref(v: &Variable) -> usize {
        isel!(v[0]).unwrap().ndim()
    }
    fn through_mut(v: &mut Variable) {
        isel!(mut v[0, 0, 0])
            .unwrap()
            .data_mut::<i64>()
            .unwrap()
            .fill(9);
    }
    let mut v = var();
    assert_eq!(through_ref(&v), 2);
    through_mut(&mut v);
    assert_eq!(v.data::<i64>().unwrap()[[0, 0, 0]], 9);
    let mut boxed = Box::new(var());
    assert_eq!(isel!(boxed[.., 1]).unwrap().shape(), &[2, 3]);
    assert_eq!(isel!(mut boxed[1]).unwrap().shape(), &[3, 4]);
}
