//! annoray quick start: build, inspect, select, compute with units, reduce.
//!
//! After xarray's "Quick overview". The Python twin is
//! `annoray-python/examples/quickstart.py`; both print the same lines.
//!
//! ```text
//! cargo run --example quickstart
//! ```
// annoray's operators return `Result` (unit and shape errors are values, never panics); the
// workspace's `arithmetic_side_effects` lint cannot tell.
#![allow(clippy::arithmetic_side_effects)]

use std::error::Error;

use annoray::ndarray::{ArrayD, IxDyn, array};
use annoray::{
    DataArray, DataArrayBase, Dataset, ReduceOptions, Reduction, Repr, Variable, isel, sel,
};

/// Values with three decimals, as the Python twin prints them.
fn show<S: Repr>(label: &str, da: &DataArrayBase<S>) -> Result<(), annoray::CoreError> {
    let v: Vec<String> = da.to_f64()?.iter().map(|v| format!("{v:.3}")).collect();
    println!("{label}: [{}]", v.join(", ").replace("NaN", "nan"));
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // A DataArray: values, dimension names, coordinates and a unit.
    let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(f64::from).collect())?;
    let da = DataArray::from_array(["x", "y"], data)?
        .with_dim_coord("x", array![10.0, 20.0])?
        .with_dim_coord("y", array![1.0, 2.0, 3.0])?
        .with_name("height")
        .with_unit("m".parse()?);
    println!("dims: {}", da.dims().join(", "));
    let sizes: Vec<String> = da.sizes().map(|(k, n)| format!("{k}={n}")).collect();
    println!("sizes: {}", sizes.join(", "));
    println!("units: {}", da.unit().ok_or("no unit")?);

    // Selection by position (isel) and by label (sel); label slices include both ends.
    show("isel x=0", &isel!(da[x = 0])?)?;
    show("sel x=20", &sel!(da[x = 20.0])?)?;
    show("sel y=1.5:3", &sel!(da[y = 1.5:3.0])?)?;
    show("sel y~2.4 (nearest)", &sel!(da[y = nearest(2.4)])?)?;

    // Arithmetic carries units, and refuses what does not make sense.
    let seconds = DataArray::scalar(2.0)?.with_unit("s".parse()?);
    let unit = |r: &DataArray| r.unit().map(ToString::to_string).unwrap_or_default();
    println!("da * 2 -> {}", unit(&(&da * 2.0)?));
    println!("da * da -> {}", unit(&(&da * &da)?));
    println!("da / 2 s -> {}", unit(&(&da / &seconds)?));
    match &da + &seconds {
        Ok(_) => println!("da + 2 s -> (no error?)"),
        Err(e) => println!("da + 2 s -> error: {e}"),
    }

    // Reductions, by dimension name.
    show("mean over x", &da.mean("x")?)?;
    let opts = ReduceOptions::default().ddof(1);
    show(
        "std over y (ddof=1)",
        &da.reduce(Reduction::Std, "y", &opts)?,
    )?;
    show("sum", &da.sum(..)?)?;

    // Unit conversion.
    show("in cm", &da.to_unit(&"cm".parse()?)?)?;

    // A Dataset: variables sharing dimensions; arithmetic broadcasts by name.
    let mut ds = Dataset::new().with_attrs([("title", "quick start")]);
    ds.add_array("height", da)?;
    ds.add_var(
        "width",
        Variable::new(["y"], array![1.0, 2.0, 4.0].into_dyn())?.with_unit("m".parse()?),
    )?;
    let area = (&ds.get("height")? * &ds.get("width")?)?;
    println!(
        "area units: {}, dims: {}",
        unit(&area),
        area.dims().join(", ")
    );
    show("area", &area)?;
    println!("{ds}");
    Ok(())
}
