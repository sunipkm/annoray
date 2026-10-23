//! Complex numbers (`complex64`, `complex128`): arithmetic, functions, reductions, casts,
//! comparisons, missing values, and storage in Zarr and netCDF-4 (h5py's `{r, i}` compound).
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::arithmetic_side_effects
)]

use std::path::PathBuf;

use annoray::io::zarr::{ZarrFormat, ZarrOptions};
use annoray::io::{Format, ToNetcdf, ToZarr, WriteOptions, open_dataset};
use annoray::num_complex::Complex;
use annoray::units::Unit;
use annoray::{
    CmpOp, CoreError, DType, DataArray, Dataset, MathFn, ReduceOptions, Reduction, Variable,
};
use ndarray::{ArrayD, IxDyn};

type C = Complex<f64>;

fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

fn da(v: Vec<C>) -> DataArray {
    let n = v.len();
    DataArray::from_array(["x"], ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap()).unwrap()
}

fn values(d: &DataArray) -> Vec<C> {
    d.data::<C>().unwrap().iter().copied().collect()
}

fn reals(d: &DataArray) -> Vec<f64> {
    d.data::<f64>().unwrap().iter().copied().collect()
}

fn close(a: C, b: C) -> bool {
    (a - b).norm() <= 1e-12 * a.norm().max(b.norm()).max(1.0)
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("annoray-complex-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn element_types() {
    let a = da(vec![c(1.0, 2.0)]);
    assert_eq!(a.dtype(), DType::C128);
    assert_eq!(DType::C64.to_string(), "complex64");
    assert_eq!(DType::C128.to_string(), "complex128");
    let narrow = DataArray::from_array(
        ["x"],
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![Complex::<f32>::new(1.0, 2.0)]).unwrap(),
    )
    .unwrap();
    assert_eq!(narrow.dtype(), DType::C64);
}

#[test]
fn arithmetic_promotes_like_numpy() {
    let a = da(vec![c(1.0, 2.0), c(-3.0, 0.5)]);
    let b = da(vec![c(0.5, -1.0), c(2.0, 2.0)]);
    let sum = (&a + &b).unwrap();
    assert_eq!(values(&sum), [c(1.5, 1.0), c(-1.0, 2.5)]);
    let prod = (&a * &b).unwrap();
    assert_eq!(
        values(&prod),
        [c(1.0, 2.0) * c(0.5, -1.0), c(-3.0, 0.5) * c(2.0, 2.0)]
    );
    let quot = (&a / &b).unwrap();
    assert!(close(values(&quot)[0], c(1.0, 2.0) / c(0.5, -1.0)));
    // A real operand is promoted; complex64 with float32 stays complex64.
    let real = DataArray::from_array(["x"], ndarray::array![2.0, 4.0].into_dyn()).unwrap();
    assert_eq!(values(&(&a * &real).unwrap()), [c(2.0, 4.0), c(-12.0, 2.0)]);
    let a32 = DataArray::from_array(
        ["x"],
        ndarray::array![Complex::<f32>::new(1.0, 1.0)].into_dyn(),
    )
    .unwrap();
    let r32 = DataArray::from_array(["x"], ndarray::array![2.0f32].into_dyn()).unwrap();
    assert_eq!((&a32 * &r32).unwrap().dtype(), DType::C64);
    assert_eq!(
        (&a32 * &real.isel([("x", 0..1)]).unwrap().to_owned())
            .unwrap()
            .dtype(),
        DType::C128
    );
    // No order: no maximum.
    assert!(matches!(
        a.binary(annoray::BinOp::Max, &b, annoray_units::UnitPolicy::Strict),
        Err(CoreError::UnsupportedDType { .. })
    ));
}

#[test]
fn units_scale_both_parts() {
    let m: Unit = "m".parse().unwrap();
    let km: Unit = "km".parse().unwrap();
    let a = da(vec![c(1.0, 2.0)]).with_unit(m.clone());
    let b = da(vec![c(1.0, -1.0)]).with_unit(km);
    let sum = a
        .binary(
            annoray::BinOp::Add,
            &b,
            annoray_units::UnitPolicy::AutoRescale,
        )
        .unwrap();
    assert_eq!(sum.unit(), Some(&m));
    assert_eq!(values(&sum), [c(1001.0, -998.0)]);
}

#[test]
fn functions() {
    let z = c(3.0, 4.0);
    let a = da(vec![z]);
    assert_eq!(reals(&a.math(MathFn::Abs).unwrap()), [5.0]);
    assert_eq!(reals(&a.math(MathFn::Real).unwrap()), [3.0]);
    assert_eq!(reals(&a.math(MathFn::Imag).unwrap()), [4.0]);
    assert_eq!(reals(&a.math(MathFn::Angle).unwrap()), [4f64.atan2(3.0)]);
    // The angle of data with a unit is in radians (unitless data stays unitless).
    let volts = a.clone().with_unit("V".parse().unwrap());
    assert_eq!(
        volts
            .math(MathFn::Angle)
            .unwrap()
            .unit()
            .unwrap()
            .to_string(),
        "rad"
    );
    assert_eq!(
        volts.math(MathFn::Abs).unwrap().unit().unwrap().to_string(),
        "V"
    );
    assert_eq!(values(&a.math(MathFn::Conj).unwrap()), [c(3.0, -4.0)]);
    assert!(close(values(&a.math(MathFn::Exp).unwrap())[0], z.exp()));
    assert!(close(values(&a.math(MathFn::Sqrt).unwrap())[0], z.sqrt()));
    assert!(close(values(&a.math(MathFn::Log).unwrap())[0], z.ln()));
    assert!(close(values(&a.pow(2.0).unwrap())[0], z * z));
    // exp(iφ): the phase factor of optics.
    let phase = da(vec![c(0.0, std::f64::consts::FRAC_PI_2)]);
    assert!(close(
        values(&phase.math(MathFn::Exp).unwrap())[0],
        c(0.0, 1.0)
    ));
    // complex64 parts stay float32.
    let narrow = DataArray::from_array(
        ["x"],
        ndarray::array![Complex::<f32>::new(3.0, 4.0)].into_dyn(),
    )
    .unwrap();
    assert_eq!(narrow.math(MathFn::Abs).unwrap().dtype(), DType::F32);
    // Real data: the real part and the conjugate are the data (integers stay integers).
    let ints = DataArray::from_array(["x"], ndarray::array![-2i32, 3].into_dyn()).unwrap();
    assert_eq!(ints.math(MathFn::Conj).unwrap().dtype(), DType::I32);
    assert_eq!(
        ints.math(MathFn::Imag)
            .unwrap()
            .data::<i32>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [0, 0]
    );
    let r = DataArray::from_array(["x"], ndarray::array![-2.0, 3.0].into_dyn()).unwrap();
    assert_eq!(
        reals(&r.math(MathFn::Angle).unwrap()),
        [std::f64::consts::PI, 0.0]
    );
    // Refused: no order, no rounding.
    for f in [MathFn::Floor, MathFn::Sign, MathFn::Cbrt] {
        assert!(
            matches!(a.math(f), Err(CoreError::UnsupportedDType { .. })),
            "{f:?}"
        );
    }
}

#[test]
fn reductions() {
    let a = da(vec![c(1.0, 2.0), c(3.0, -4.0), c(f64::NAN, 0.0)]);
    let opts = ReduceOptions::default();
    let sum = a.reduce(Reduction::Sum, "x", &opts).unwrap();
    assert_eq!(values(&sum), [c(4.0, -2.0)]);
    let mean = a.reduce(Reduction::Mean, "x", &opts).unwrap();
    assert_eq!(values(&mean), [c(2.0, -1.0)]);
    let prod = a.reduce(Reduction::Prod, "x", &opts).unwrap();
    assert_eq!(values(&prod), [c(1.0, 2.0) * c(3.0, -4.0)]);
    // var: the mean squared modulus of the deviations (numpy), real.
    let var = a.reduce(Reduction::Var, "x", &opts).unwrap();
    assert_eq!(var.dtype(), DType::F64);
    let want =
        ((c(1.0, 2.0) - c(2.0, -1.0)).norm_sqr() + (c(3.0, -4.0) - c(2.0, -1.0)).norm_sqr()) / 2.0;
    assert!((reals(&var)[0] - want).abs() < 1e-12);
    let count = a.reduce(Reduction::Count, "x", &opts).unwrap();
    assert_eq!(
        count
            .data::<i64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [2]
    );
    // Without skipping, a missing value makes the result missing.
    let strict = ReduceOptions::default().skipna(false);
    assert!(
        values(&a.reduce(Reduction::Sum, "x", &strict).unwrap())[0]
            .re
            .is_nan()
    );
    for op in [
        Reduction::Min,
        Reduction::Max,
        Reduction::Median,
        Reduction::ArgMax,
    ] {
        assert!(
            matches!(
                a.reduce(op, "x", &opts),
                Err(CoreError::UnsupportedDType { .. })
            ),
            "{op:?}"
        );
    }
}

#[test]
fn casts() {
    let r = DataArray::from_array(["x"], ndarray::array![1.5, -2.0].into_dyn()).unwrap();
    assert_eq!(
        values(&r.astype(DType::C128).unwrap()),
        [c(1.5, 0.0), c(-2.0, 0.0)]
    );
    let a = da(vec![c(1.5, 0.0), c(-2.0, 0.0)]);
    assert_eq!(reals(&a.astype(DType::F64).unwrap()), [1.5, -2.0]);
    let narrow = a.astype(DType::C64).unwrap();
    assert_eq!(narrow.dtype(), DType::C64);
    assert_eq!(narrow.astype(DType::C128).unwrap().dtype(), DType::C128);
    // The imaginary part must be zero to drop it.
    let b = da(vec![c(1.0, 1.0)]);
    assert!(matches!(b.astype(DType::F64), Err(CoreError::Cast { .. })));
    assert_eq!(
        b.astype(DType::Bool)
            .unwrap()
            .data::<bool>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [true]
    );
}

#[test]
fn comparisons_and_missing_values() {
    let a = da(vec![c(1.0, 2.0), c(3.0, 0.0), c(f64::NAN, 1.0)]);
    let b = da(vec![c(1.0, 2.0), c(3.0, 1.0), c(f64::NAN, 1.0)]);
    let eq = a.compare(CmpOp::Eq, &b).unwrap();
    assert_eq!(
        eq.data::<bool>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [true, false, false]
    );
    let ne = a.compare(CmpOp::Ne, &b).unwrap();
    assert_eq!(
        ne.data::<bool>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [false, true, true]
    );
    assert!(matches!(
        a.compare(CmpOp::Lt, &b),
        Err(CoreError::UnsupportedDType { .. })
    ));
    // Against a real number: equal when the imaginary part is zero.
    let three = DataArray::from_array(["x"], ndarray::array![1.0, 3.0, 0.0].into_dyn()).unwrap();
    assert_eq!(
        a.compare(CmpOp::Eq, &three)
            .unwrap()
            .data::<bool>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [false, true, false]
    );
    assert_eq!(
        a.isnull()
            .unwrap()
            .data::<bool>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [false, false, true]
    );
    assert_eq!(values(&a.fillna(0.0).unwrap())[2], c(0.0, 0.0));
    assert!(a.equals(&a).unwrap());
    assert!(!a.equals(&b).unwrap());
}

fn sample() -> Dataset {
    let field: ArrayD<C> = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |ix| {
        c(ix[0] as f64 - 1.0, 0.25 * ix[1] as f64)
    });
    let mut field = field;
    field[[1, 2]] = c(f64::NAN, f64::NAN);
    let narrow: ArrayD<Complex<f32>> = ArrayD::from_shape_fn(IxDyn(&[4]), |ix| {
        Complex::new(ix[0] as f32, -(ix[0] as f32))
    });
    let mut ds = Dataset::new();
    ds.add_var(
        "field",
        Variable::new(["y", "x"], field)
            .unwrap()
            .with_unit("V/m".parse().unwrap()),
    )
    .unwrap();
    ds.add_var("narrow", Variable::new(["x"], narrow).unwrap())
        .unwrap();
    ds
}

fn assert_same(a: &Dataset, b: &Dataset, what: &str) {
    for name in ["field", "narrow"] {
        let (x, y) = (&a.data_vars()[name], &b.data_vars()[name]);
        assert_eq!(x.dims(), y.dims(), "{what}: {name}");
        assert_eq!(x.dtype(), y.dtype(), "{what}: {name}");
        assert_eq!(x.unit(), y.unit(), "{what}: {name}");
        assert!(
            annoray::DataArray::from(x.to_owned())
                .equals(&annoray::DataArray::from(y.to_owned()))
                .unwrap(),
            "{what}: {name} values"
        );
    }
}

#[test]
fn zarr_round_trips() {
    let ds = sample();
    for format in [ZarrFormat::V2, ZarrFormat::V3] {
        let dir = tmp(&format!("rt-{format:?}.zarr"));
        let _ = std::fs::remove_dir_all(&dir);
        ds.to_zarr_with(&dir, &ZarrOptions::format(format)).unwrap();
        let mut back = open_dataset(&dir).unwrap();
        back.load().unwrap();
        assert_same(&ds, &back, &format!("{format:?}"));
    }
}

#[test]
fn netcdf4_round_trips_as_h5py_compounds() {
    let ds = sample();
    let path = tmp("rt.nc");
    ds.to_netcdf(&path).unwrap();
    let back = open_dataset(&path).unwrap();
    assert_same(&ds, &back, "netCDF-4");
    // netCDF4-python's complex numbers (nc-complex: a compound `{r, i}` named
    // `_PFNC_DOUBLE_COMPLEX_TYPE`), from fixtures/gen_complex.py.
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/nc/complex.nc");
    let theirs = open_dataset(fixture).unwrap();
    let f = &theirs.data_vars()["field"];
    assert_eq!(f.dtype(), DType::C128);
    assert_eq!(f.unit().unwrap().to_string(), "V/m");
    let z = f.data::<C>().unwrap();
    for ((y, x), v) in
        [(0, 0), (0, 3), (2, 1)]
            .into_iter()
            .zip([c(-1.0, 0.0), c(-1.0, 0.75), c(1.0, 0.25)])
    {
        assert_eq!(z[[y, x]], v);
    }
    assert!(z[[1, 2]].re.is_nan() && z[[1, 2]].im.is_nan());
    // netCDF-3 has no complex type.
    let classic = tmp("rt3.nc");
    assert!(
        ds.to_netcdf_with(&classic, &WriteOptions::format(Format::Netcdf3Classic))
            .is_err()
    );
}
