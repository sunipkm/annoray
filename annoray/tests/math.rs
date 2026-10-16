//! Unit-aware elementwise math (numpy's unary ufuncs) and powers.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::units::Unit;
use annoray::{DType, DataArray, MathFn};
use ndarray::array;

fn vals(d: &DataArray) -> Vec<f64> {
    d.to_f64().unwrap().iter().copied().collect()
}

fn close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * w.abs().max(1.0),
            "{got:?} vs {want:?}"
        );
    }
}

fn u(s: &str) -> Unit {
    s.parse().unwrap()
}

fn da(v: ndarray::Array1<f64>, unit: &str) -> DataArray {
    DataArray::from_array(["x"], v.into_dyn())
        .unwrap()
        .with_unit(u(unit))
}

#[test]
fn roots_and_powers_transform_units() {
    let a = da(array![4.0, 9.0], "m2");
    let r = a.math(MathFn::Sqrt).unwrap();
    close(&vals(&r), &[2.0, 3.0]);
    assert!(
        r.unit()
            .unwrap()
            .converter_to(&u("m"))
            .unwrap()
            .is_identity()
    );
    assert!(da(array![4.0], "m").math(MathFn::Sqrt).is_err());
    let s = da(array![2.0], "m").math(MathFn::Square).unwrap();
    assert!(
        s.unit()
            .unwrap()
            .converter_to(&u("m2"))
            .unwrap()
            .is_identity()
    );
    let p = da(array![2.0], "m").pow(3.0).unwrap();
    close(&vals(&p), &[8.0]);
    assert!(
        p.unit()
            .unwrap()
            .converter_to(&u("m3"))
            .unwrap()
            .is_identity()
    );
    let q = a.pow(0.5).unwrap();
    assert!(
        q.unit()
            .unwrap()
            .converter_to(&u("m"))
            .unwrap()
            .is_identity()
    );
    assert!(da(array![2.0], "m").pow(1.5).is_err());
    close(&vals(&da(array![4.0], "1").pow(1.5).unwrap()), &[8.0]);
    let inv = da(array![4.0], "s").math(MathFn::Reciprocal).unwrap();
    assert!(inv.unit().unwrap().is_convertible(&u("Hz")));
}

#[test]
fn transcendental_functions_need_plain_numbers_or_angles() {
    assert!(da(array![1.0], "m").math(MathFn::Exp).is_err());
    // Percent is converted to a plain number first.
    let e = da(array![50.0], "percent").math(MathFn::Exp).unwrap();
    close(&vals(&e), &[0.5f64.exp()]);
    assert_eq!(e.unit(), None);
    let s = da(array![90.0, 180.0], "degree").math(MathFn::Sin).unwrap();
    close(&vals(&s), &[1.0, std::f64::consts::PI.sin()]);
    assert!(da(array![1.0], "rad").math(MathFn::Exp).is_err());
    let a = da(array![1.0], "1").math(MathFn::Arcsin).unwrap();
    close(&vals(&a), &[std::f64::consts::FRAC_PI_2]);
    assert_eq!(a.unit(), Some(&u("rad")));
    // Without a unit: plain numpy.
    let plain = DataArray::from_array(["x"], array![0.0, 1.0].into_dyn()).unwrap();
    close(
        &vals(&plain.math(MathFn::Log1p).unwrap()),
        &[0.0, 2f64.ln()],
    );
}

#[test]
fn signs_rounding_and_types() {
    let c = da(array![-1.5, 2.5], "degC");
    assert!(c.math(MathFn::Neg).is_err());
    assert!(c.math(MathFn::Abs).is_err());
    let f = c.math(MathFn::Rint).unwrap();
    close(&vals(&f), &[-2.0, 2.0]);
    assert_eq!(f.unit(), Some(&u("degC")));
    let n = da(array![-1.5, 2.5], "m").math(MathFn::Neg).unwrap();
    close(&vals(&n), &[1.5, -2.5]);
    let ints = DataArray::from_array(["x"], array![-3i32, 4].into_dyn()).unwrap();
    let ab = ints.math(MathFn::Abs).unwrap();
    assert_eq!(ab.variable().dtype(), DType::I32);
    close(&vals(&ab), &[3.0, 4.0]);
    assert_eq!(
        ints.math(MathFn::Sqrt).unwrap().variable().dtype(),
        DType::F64
    );
    let min = DataArray::from_array(["x"], array![i8::MIN].into_dyn()).unwrap();
    assert!(min.math(MathFn::Abs).is_err());
    let f32s = DataArray::from_array(["x"], array![4.0f32].into_dyn()).unwrap();
    assert_eq!(
        f32s.math(MathFn::Sqrt).unwrap().variable().dtype(),
        DType::F32
    );
    let b = DataArray::from_array(["x"], array![true].into_dyn()).unwrap();
    assert!(b.math(MathFn::Sqrt).is_err());
    close(
        &vals(
            &da(array![-2.0, 0.0, f64::NAN], "m")
                .math(MathFn::Sign)
                .unwrap(),
        ),
        &[-1.0, 0.0, f64::NAN],
    );
    assert_eq!(MathFn::from_numpy("absolute"), Some(MathFn::Abs));
}

#[test]
fn lazy_data_stays_lazy() {
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
    let l = lazy.math(MathFn::Floor).unwrap();
    assert!(l.variable().is_lazy());
    close(&vals(&l), &vals(&eager.math(MathFn::Floor).unwrap()));
}

#[test]
fn rename_dimensions_and_coordinates() {
    let a = DataArray::from_array(["x", "y"], ndarray::Array2::<f64>::zeros((2, 3)).into_dyn())
        .unwrap()
        .with_dim_coord("x", array![1.0, 2.0])
        .unwrap();
    let r = a.rename_dims(&[("x", "lon"), ("y", "lat")]).unwrap();
    assert_eq!(r.dims(), ["lon", "lat"]);
    assert_eq!(r.coords()["lon"].dims(), ["lon"]);
    assert!(a.rename_dims(&[("nope", "z")]).is_err());
    assert!(a.rename_dims(&[("x", "y")]).is_err());
}
