//! A test extension module for `annoray-pyo3`: round trips of annoray values between Python
//! and this extension's Rust (see `crates/annoray-pyo3/tests/test_interop.py`).

use annoray::num_complex::Complex;
use annoray::units::dims::{Length, Time, Velocity};
use annoray::units::{Calendar, CfDate, Dyn};
use annoray::{AttrValue, DataArray, Dataset, Duration, Instant, TimeZone, Variable};
use annoray_pyo3::{
    ArDataArray, ArDataArrayRef, ArDataTree, ArDataset, ArQuantity, ArUnit, annoray_module,
    unit_error, value_error,
};
use ndarray::{ArrayD, IxDyn};
use pyo3::prelude::*;

/// The `annoray-pyo3` version, after checking the `annoray` package against it.
#[pyfunction]
fn check(py: Python<'_>) -> PyResult<&'static str> {
    annoray_module(py)?;
    Ok(annoray_pyo3::VERSION)
}

#[pyfunction]
fn unit_roundtrip(u: ArUnit) -> ArUnit {
    u
}

#[pyfunction]
fn quantity_roundtrip(q: ArQuantity) -> ArQuantity {
    q
}

#[pyfunction]
fn complex_quantity_roundtrip(q: ArQuantity<Dyn, Complex<f64>>) -> ArQuantity<Dyn, Complex<f64>> {
    q
}

/// A length: its dimension checked when it arrives.
#[pyfunction]
fn length_roundtrip(q: ArQuantity<Length>) -> ArQuantity<Length> {
    q
}

/// The time to cover `distance` at `speed`, dimensions checked by the compiler.
#[pyfunction]
fn travel_time(
    py: Python<'_>,
    distance: ArQuantity<Length>,
    speed: ArQuantity<Velocity>,
) -> PyResult<ArQuantity<Time>> {
    // Quantity's operators check units and return a `Result`; they never panic.
    #[allow(clippy::arithmetic_side_effects)]
    let time = (&distance.0 / &speed.0).map_err(|e| unit_error(py, e))?;
    Ok(ArQuantity(time))
}

/// `da` converted to `unit` (an `annoray.Unit` or a unit string) in Rust.
#[pyfunction]
fn convert(py: Python<'_>, da: ArDataArray, unit: ArUnit) -> PyResult<ArDataArray> {
    let out = da.0.to_unit(&unit.0).map_err(|e| unit_error(py, e))?;
    Ok(ArDataArray(out))
}

#[pyfunction]
fn dataarray_roundtrip(da: ArDataArray) -> ArDataArray {
    da
}

/// Back to Python by reference (`&ArDataArray`): the values copied twice.
#[pyfunction]
fn dataarray_by_reference(py: Python<'_>, da: ArDataArray) -> PyResult<Bound<'_, PyAny>> {
    (&da).into_pyobject(py)
}

#[pyfunction]
fn dataset_roundtrip(ds: ArDataset) -> ArDataset {
    ds
}

#[pyfunction]
fn tree_roundtrip(tree: ArDataTree) -> ArDataTree {
    tree
}

/// Read in place, then copied into Rust and back.
#[pyfunction]
fn dataarray_ref_roundtrip(da: ArDataArrayRef<'_>) -> PyResult<ArDataArray> {
    Ok(ArDataArray(da.to_dataarray()?))
}

/// The address of the `float64` values `da` lends (to compare with numpy's).
#[pyfunction]
fn view_address(da: ArDataArrayRef<'_>) -> PyResult<usize> {
    let view = da.variable()?.view::<f64>()?;
    Ok(view.as_array().as_ptr().addr())
}

/// The sum of `da`'s `float64` values, read in place.
#[pyfunction]
fn view_sum(da: ArDataArrayRef<'_>) -> PyResult<f64> {
    Ok(da.variable()?.view::<f64>()?.as_array().sum())
}

/// The names, kinds and units of `da`'s variable and coordinates, read in place.
#[pyfunction]
fn describe(da: ArDataArrayRef<'_>) -> PyResult<Vec<(String, String, Option<String>)>> {
    let v = da.variable()?;
    let mut out = vec![(
        da.name()?.unwrap_or_default(),
        v.kind()?,
        v.unit()?.map(|u| u.to_string()),
    )];
    for (name, c) in da.coords()? {
        out.push((name, c.kind()?, c.unit()?.map(|u| u.to_string())));
    }
    Ok(out)
}

/// `n` values `0, 1, …` in metres, made in Rust.
#[pyfunction]
fn make_big(n: usize) -> PyResult<ArDataArray> {
    #[allow(clippy::cast_precision_loss)]
    let values = ArrayD::from_shape_fn(IxDyn(&[n]), |i| i[0] as f64);
    let var = Variable::new(["x"], values)
        .map_err(value_error)?
        .with_unit("m".parse().map_err(value_error)?);
    Ok(ArDataArray(DataArray::new(var).with_name("big")))
}

fn var<T: annoray::Element>(dims: &[&str], shape: &[usize], v: Vec<T>) -> PyResult<Variable> {
    let a = ArrayD::from_shape_vec(IxDyn(shape), v).map_err(value_error)?;
    Variable::new(dims.iter().copied(), a).map_err(value_error)
}

/// A dataset made in Rust with every element type, units, a time zone, CF dates in a
/// non-standard calendar (to the nanosecond) and attributes of every kind.
#[pyfunction]
fn make_samples() -> PyResult<ArDataset> {
    let x = ["x"];
    let mut ds = Dataset::new();
    let unit = |s: &str| s.parse::<annoray::Unit>().map_err(value_error);
    let time = vec![
        Instant::from_ymd_hms_nano((2024, 7, 1), (2, 0, 0), 5).map_err(value_error)?,
        Instant::NAT,
        Instant::from_ymd(2024, 7, 2).map_err(value_error)?,
    ];
    let mut t = var(&x, &[3], time)?;
    t.set_time_zone(Some(
        "America/New_York"
            .parse::<TimeZone>()
            .map_err(value_error)?,
    ))
    .map_err(value_error)?;
    ds.add_coord("time", t).map_err(value_error)?;
    ds.add_coord(
        "x",
        var(&x, &[3], vec![10.0, 20.0, 30.0])?.with_unit(unit("km")?),
    )
    .map_err(value_error)?;
    let add = |ds: &mut Dataset, name: &str, v: Variable| ds.add_var(name, v).map_err(value_error);
    add(
        &mut ds,
        "f64",
        var(&x, &[3], vec![1.5, f64::NAN, -2.0])?.with_unit(unit("degC")?),
    )?;
    add(&mut ds, "f32", var(&x, &[3], vec![1.5_f32, 2.0, 3.25])?)?;
    add(
        &mut ds,
        "c128",
        var(
            &x,
            &[3],
            vec![
                Complex::new(1.0, 2.0),
                Complex::new(0.0, -1.0),
                Complex::new(f64::NAN, 0.0),
            ],
        )?
        .with_unit(unit("V")?),
    )?;
    add(
        &mut ds,
        "c64",
        var(&x, &[3], vec![Complex::new(1.0_f32, 2.0); 3])?,
    )?;
    add(&mut ds, "i8", var(&x, &[3], vec![-1_i8, 0, 127])?)?;
    add(&mut ds, "i16", var(&x, &[3], vec![-1_i16, 0, 32_767])?)?;
    add(&mut ds, "i32", var(&x, &[3], vec![-1_i32, 0, i32::MAX])?)?;
    add(&mut ds, "i64", var(&x, &[3], vec![i64::MIN, 0, i64::MAX])?)?;
    add(&mut ds, "u8", var(&x, &[3], vec![0_u8, 1, 255])?)?;
    add(&mut ds, "u16", var(&x, &[3], vec![0_u16, 1, 65_535])?)?;
    add(&mut ds, "u32", var(&x, &[3], vec![0_u32, 1, u32::MAX])?)?;
    add(&mut ds, "u64", var(&x, &[3], vec![0_u64, 1, u64::MAX])?)?;
    add(&mut ds, "bool", var(&x, &[3], vec![true, false, true])?)?;
    add(
        &mut ds,
        "text",
        var(
            &x,
            &[3],
            vec!["a".to_owned(), "ångström µm".to_owned(), String::new()],
        )?,
    )?;
    let hours = |h| Duration::from_hours(h).ok_or_else(|| value_error("overflow"));
    add(
        &mut ds,
        "duration",
        var(&x, &[3], vec![hours(1)?, Duration::NAT, hours(-2)?])?,
    )?;
    let cf = |y, m, d, ns| {
        CfDate::from_ymd(Calendar::NoLeap, y, m, d)
            .and_then(|c| c.and_hms_nano(1, 2, 3, ns))
            .map_err(value_error)
    };
    add(
        &mut ds,
        "noleap",
        var(
            &x,
            &[3],
            vec![
                cf(1, 2, 28, 7)?,
                CfDate::NAT,
                cf(9999, 12, 31, 999_999_999)?,
            ],
        )?,
    )?;
    add(
        &mut ds,
        "scalar",
        var::<f64>(&[], &[], vec![42.0])?.with_unit(unit("hPa")?),
    )?;
    add(
        &mut ds,
        "grid",
        var(&["x", "y"], &[3, 2], (0..6).map(f64::from).collect())?,
    )?;
    *ds.attrs_mut() = [
        ("title", AttrValue::Str("samples".into())),
        ("history", AttrValue::Strs(vec!["one".into(), "two".into()])),
        ("i8", AttrValue::I8(vec![-1])),
        ("i16", AttrValue::I16(vec![1, 2])),
        ("i32", AttrValue::I32(vec![3])),
        ("i64", AttrValue::I64(vec![4, 5])),
        ("u8", AttrValue::U8(vec![6])),
        ("u16", AttrValue::U16(vec![7])),
        ("u32", AttrValue::U32(vec![8])),
        ("u64", AttrValue::U64(vec![9])),
        ("f32", AttrValue::F32(vec![0.5])),
        ("f64", AttrValue::F64(vec![0.25, 0.75])),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect();
    Ok(ArDataset(ds))
}

/// The nanoseconds of `ds["noleap"]`'s CF dates as Rust sees them (`None` for NaT): what
/// cftime objects (microseconds) cannot show.
#[pyfunction]
fn noleap_nanoseconds(ds: ArDataset) -> PyResult<Vec<Option<u32>>> {
    let var =
        ds.0.data_vars()
            .get("noleap")
            .ok_or_else(|| value_error("no noleap"))?;
    let dates = var.data::<CfDate>().map_err(value_error)?;
    Ok(dates
        .iter()
        .map(|d| (!d.is_nat()).then(|| d.nanosecond()))
        .collect())
}

/// Whether annoray-pyo3 at `rust` accepts the annoray package at `python`.
#[pyfunction]
fn compatible(rust: &str, python: &str) -> bool {
    annoray_pyo3::compatible(rust, python)
}

/// Round trips of annoray values through annoray-pyo3.
#[pymodule]
fn annoray_pyo3_test(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check, m)?)?;
    m.add_function(wrap_pyfunction!(compatible, m)?)?;
    m.add_function(wrap_pyfunction!(unit_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(quantity_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(complex_quantity_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(length_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(travel_time, m)?)?;
    m.add_function(wrap_pyfunction!(convert, m)?)?;
    m.add_function(wrap_pyfunction!(dataarray_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(dataarray_by_reference, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(tree_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(dataarray_ref_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(view_address, m)?)?;
    m.add_function(wrap_pyfunction!(view_sum, m)?)?;
    m.add_function(wrap_pyfunction!(describe, m)?)?;
    m.add_function(wrap_pyfunction!(make_big, m)?)?;
    m.add_function(wrap_pyfunction!(make_samples, m)?)?;
    m.add_function(wrap_pyfunction!(noleap_nanoseconds, m)?)?;
    Ok(())
}
