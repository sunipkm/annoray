//! `DataArray::put`: a copy with values replaced, checked for units, types and shape.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects
)]

use annoray::units::Unit;
use annoray::{CoreError, DataArray, Name, Positions, Variable};
use ndarray::{ArrayD, IxDyn, array};

fn grid() -> DataArray {
    let data = ArrayD::from_shape_fn(IxDyn(&[5, 3]), |ix| (ix[0] * 10 + ix[1]) as f32);
    DataArray::new(Variable::new(["x", "y"], data).unwrap())
        .with_name("v")
        .with_coord(
            "x",
            Variable::new(["x"], array![0.0, 1.0, 2.0, 3.0, 4.0].into_dyn()).unwrap(),
        )
        .unwrap()
}

fn vals(d: &DataArray) -> Vec<f32> {
    d.variable()
        .data::<f32>()
        .unwrap()
        .iter()
        .copied()
        .collect()
}

fn scalar(v: f64) -> DataArray {
    DataArray::new(Variable::new(Vec::<Name>::new(), ArrayD::from_elem(IxDyn(&[]), v)).unwrap())
}

fn at(d: &str, p: Positions) -> (Name, Positions) {
    (Name::from(d), p)
}

#[test]
fn writes_a_copy() {
    let a = grid();
    let b = a.put(&[at("x", Positions::At(1))], &scalar(-1.0)).unwrap();
    assert_eq!(&vals(&b)[3..6], &[-1.0, -1.0, -1.0]);
    assert_eq!(vals(&a)[3], 10.0, "the original is unchanged");
    assert_eq!(b.dtype(), annoray::DType::F32);
    assert_eq!(b.name().map(|n| n.as_str()), Some("v"));
    assert!(b.coords().contains_key("x"));
}

#[test]
fn strided_scattered_and_reversed_positions() {
    let a = grid();
    let col = DataArray::new(Variable::new(["x"], array![7.0f32, 8.0, 9.0].into_dyn()).unwrap());
    // Evenly spaced (forward and backward) and scattered positions agree with numpy.
    for (pos, expect) in [
        (vec![0, 2, 4], [(0, 7.0), (2, 8.0), (4, 9.0)]),
        (vec![4, 2, 0], [(4, 7.0), (2, 8.0), (0, 9.0)]),
        (vec![3, 0, 1], [(3, 7.0), (0, 8.0), (1, 9.0)]),
    ] {
        let b = a
            .put(
                &[
                    at("x", Positions::Each(pos.clone())),
                    at("y", Positions::At(2)),
                ],
                &col,
            )
            .unwrap();
        let v = vals(&b);
        for (x, want) in expect {
            assert_eq!(v[x * 3 + 2], want, "{pos:?}");
            assert_eq!(
                v[x * 3 + 1],
                (x * 10 + 1) as f32,
                "{pos:?}: other columns kept"
            );
        }
    }
}

#[test]
fn units_convert_and_bare_numbers_need_zero() {
    let k = Unit::parse("K").unwrap();
    let a = DataArray::new(
        Variable::new(["x"], array![1.0f64, 2.0, 3.0].into_dyn())
            .unwrap()
            .with_unit(k),
    );
    let mk = DataArray::new(
        Variable::new(Vec::<Name>::new(), ArrayD::from_elem(IxDyn(&[]), 500.0))
            .unwrap()
            .with_unit(Unit::parse("mK").unwrap()),
    );
    let b = a.put(&[at("x", Positions::At(0))], &mk).unwrap();
    assert_eq!(b.variable().data::<f64>().unwrap()[0], 0.5);
    assert!(a.put(&[at("x", Positions::At(0))], &scalar(0.0)).is_ok());
    assert!(matches!(
        a.put(&[at("x", Positions::At(0))], &scalar(1.0)),
        Err(CoreError::Unit(_))
    ));
}

#[test]
fn element_type_never_changes() {
    let ints = DataArray::new(Variable::new(["x"], array![1u8, 2, 3].into_dyn()).unwrap());
    let p = [at("x", Positions::At(0))];
    let b = ints.put(&p, &scalar(7.0)).unwrap();
    assert_eq!(b.variable().data::<u8>().unwrap()[0], 7);
    assert!(matches!(
        ints.put(&p, &scalar(1.5)),
        Err(CoreError::Cast { .. })
    ));
    assert!(matches!(
        ints.put(&p, &scalar(300.0)),
        Err(CoreError::Cast { .. })
    ));
    assert!(matches!(
        ints.put(&p, &scalar(f64::NAN)),
        Err(CoreError::Cast { .. })
    ));
    let flag = DataArray::new(
        Variable::new(Vec::<Name>::new(), ArrayD::from_elem(IxDyn(&[]), true)).unwrap(),
    );
    assert!(matches!(
        ints.put(&p, &flag),
        Err(CoreError::DTypeMismatch { .. })
    ));
}

#[test]
fn shape_and_coordinates_are_checked() {
    let a = grid();
    let p = |v: Vec<usize>| [at("x", Positions::Each(v))];
    let short = DataArray::new(Variable::new(["x"], array![1.0f32, 2.0].into_dyn()).unwrap());
    assert!(matches!(
        a.put(&p(vec![0, 1, 2]), &short),
        Err(CoreError::DimLengthMismatch { .. })
    ));
    assert!(matches!(
        a.put(&p(vec![0, 0]), &short),
        Err(CoreError::Put(_))
    ));
    assert!(matches!(
        a.put(&p(vec![0, 9]), &short),
        Err(CoreError::Put(_))
    ));
    // A dimension the selection drops cannot be in the value.
    assert!(matches!(
        a.put(&[at("x", Positions::At(0))], &short),
        Err(CoreError::Put(_))
    ));
    // The value's coordinate must label the selected places.
    let labelled = short
        .clone()
        .with_coord(
            "x",
            Variable::new(["x"], array![3.0, 4.0].into_dyn()).unwrap(),
        )
        .unwrap();
    assert!(a.put(&p(vec![3, 4]), &labelled).is_ok());
    assert!(matches!(
        a.put(&p(vec![0, 1]), &labelled),
        Err(CoreError::NotAligned(_))
    ));
}
