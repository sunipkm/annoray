//! Toy weather data: climatologies, anomalies, resampling, rolling means, missing values.
//!
//! After xarray's "Toy weather data" example, with deterministic noise instead of random
//! numbers so the Python twin (`annoray-python/examples/weather.py`) prints the same lines.
//!
//! ```text
//! cargo run --example weather
//! ```
// annoray's operators return `Result` (never panic), which `arithmetic_side_effects` cannot
// tell; the casts are small loop counters.
#![allow(clippy::cast_precision_loss, clippy::arithmetic_side_effects)]

use std::error::Error;
use std::f64::consts::PI;

use annoray::ndarray::{Array1, Array2};
use annoray::{
    DataArray, DataArrayBase, Dataset, Repr, RollingOptions, Variable, date_range, isel, sel,
};

/// Values with three decimals, as the Python twin prints them.
fn show<S: Repr>(label: &str, da: &DataArrayBase<S>) -> Result<(), annoray::CoreError> {
    let v: Vec<String> = da.to_f64()?.iter().map(|v| format!("{v:.3}")).collect();
    println!("{label}: [{}]", v.join(", ").replace("NaN", "nan"));
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Two years of daily minimum and maximum temperatures at three places.
    let time = Variable::new(["time"], date_range("2000-01-01", 731, "D")?.into_dyn())?;
    let doy = time.dt("dayofyear")?.to_f64()?;
    let base: Vec<f64> = doy
        .iter()
        .map(|d| 10.0 + 15.0 * (2.0 * PI * (d / 365.25 - 0.28)).sin())
        .collect();
    let at = |i: usize| base.get(i).copied().unwrap_or(f64::NAN);
    let noise = |i: usize, j: usize| 3.0 * (0.7 * (7 * i + 13 * j) as f64).sin();
    let tmin = Array2::from_shape_fn((base.len(), 3), |(i, j)| {
        // Some missing readings at the first place.
        if j == 0 && i % 50 == 0 {
            f64::NAN
        } else {
            at(i) + noise(i, j)
        }
    });
    let tmax = Array2::from_shape_fn((base.len(), 3), |(i, j)| at(i) + 10.0 + noise(i, j + 5));

    let mut ds = Dataset::new();
    ds.add_coord("time", time)?;
    let places: Array1<String> = ["IA", "IN", "IL"].map(String::from).to_vec().into();
    ds.add_coord("location", Variable::new(["location"], places.into_dyn())?)?;
    for (name, values) in [("tmin", tmin), ("tmax", tmax)] {
        let v = Variable::new(["time", "location"], values.into_dyn())?.with_unit("degC".parse()?);
        ds.add_var(name, v)?;
    }
    let sizes: Vec<String> = ds.sizes().iter().map(|(k, n)| format!("{k}={n}")).collect();
    println!("sizes: {}", sizes.join(", "));
    let tmin: DataArray = ds.get("tmin")?.to_owned();
    let tmax: DataArray = ds.get("tmax")?.to_owned();

    // Missing values: counted, skipped by reductions, or filled.
    show("missing tmin per place", &tmin.isnull()?.sum("time")?)?;
    show("mean tmin (skipping NaN)", &tmin.mean("time")?)?;

    // Monthly climatology, and each day's anomaly from it.
    let clim = tmin.groupby("time.month")?.mean()?;
    show("IA tmin by month", &sel!(clim[location = "IA"])?)?;
    let anom = (&tmin.groupby("time.month")? - &clim)?;
    show(
        "IN anomaly, first week of June 2000",
        &sel!(anom[location = "IN", time = "2000-06-01":"2000-06-07"])?,
    )?;

    // Resampling to monthly means, and a centred 7-day rolling mean.
    let monthly = tmax.resample("time", "MS")?.mean()?;
    let monthly = sel!(monthly[location = "IN"])?;
    show("IN monthly tmax, 2000 H1", &isel!(monthly[time = 0:6])?)?;
    let centred = RollingOptions::default().center(true);
    let weekly = tmax.rolling("time", 7, &centred)?.mean()?;
    let weekly = sel!(weekly[location = "IL"])?;
    show("IL 7-day tmax, first days", &isel!(weekly[time = 0:6])?)?;

    // The daily range, and units.
    show("mean daily range", &(&tmax - &tmin)?.mean("time")?)?;
    show(
        "mean tmax in degF",
        &tmax.mean("time")?.to_unit(&"degF".parse()?)?,
    )?;
    let filled = tmin.fillna(-99.0)?;
    let filled = sel!(filled[location = "IA"])?;
    show("filled tmin, IA, days 0-2", &isel!(filled[time = 0:3])?)?;
    Ok(())
}
