//! Prints a small dataset, xarray style: `cargo run --example show`.
use chrono::NaiveDate;
use xarray_rs::ndarray::{ArrayD, IxDyn, array};
use xarray_rs::{DataArray, Dataset, Variable, sel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = |d| {
        NaiveDate::from_ymd_opt(2020, 1, d)
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .ok_or("bad date")
    };
    let temp = DataArray::from_array(
        ["time", "lat"],
        ArrayD::from_shape_vec(
            IxDyn(&[3, 2]),
            vec![280.1, 281.5, 279.9, 282.0, 280.4, 283.3],
        )?,
    )?
    .with_name("temp")
    .with_unit("K".parse()?)
    .with_attrs([("long_name", "air temperature")])
    .with_dim_coord("time", array![t(1)?, t(2)?, t(3)?])?
    .with_dim_coord("lat", array![45.0, 50.0])?;
    println!("{temp}");
    let mut ds = Dataset::new().with_attrs([("title", "demo")]);
    ds.add_array("temp", temp)?;
    ds.add_var(
        "elev",
        Variable::new(["lat"], array![120.0, 340.0].into_dyn())?.with_unit("m".parse()?),
    )?;
    println!("{ds}");
    println!("{}", sel!(ds[time = "2020-01-02":, lat = 50.0])?);
    Ok(())
}
