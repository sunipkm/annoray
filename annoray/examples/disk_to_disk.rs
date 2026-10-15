//! Disk to disk: compressed netCDF out, lazy reads, streamed computation and reductions.
//!
//! Data opened from files stays on disk: selections, arithmetic, unit conversion and
//! reductions read it chunk by chunk, so it never has to fit in memory. The Python twin is
//! `annoray-python/examples/disk_to_disk.py`; both print the same lines.
//!
//! ```text
//! cargo run --release --example disk_to_disk
//! ```
// annoray's operators return `Result` (never panic), which `arithmetic_side_effects` cannot
// tell; the casts are small loop counters.
#![allow(clippy::cast_precision_loss, clippy::arithmetic_side_effects)]

use std::error::Error;
use std::f64::consts::PI;
use std::path::PathBuf;

use annoray::io::{MfOptions, ToNetcdf, glob, open_dataset, open_mfdataset};
use annoray::ndarray::{Array1, Array3};
use annoray::{DataArray, DataArrayBase, Dataset, Repr, Variable, date_range, isel};

/// Values with three decimals, as the Python twin prints them.
fn show<S: Repr>(label: &str, da: &DataArrayBase<S>) -> Result<(), annoray::CoreError> {
    let v: Vec<String> = da.to_f64()?.iter().map(|v| format!("{v:.3}")).collect();
    println!("{label}: [{}]", v.join(", ").replace("NaN", "nan"));
    Ok(())
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

/// A scratch directory, removed when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Two years of daily air temperature on a small grid, in kelvin.
fn dataset() -> Result<Dataset, Box<dyn Error>> {
    // As numpy's linspace(-45, 45, 20): start + i * step, the last value exactly 45.
    let step = 90.0 / 19.0;
    let lat: Vec<f64> = (0..20)
        .map(|i| {
            if i == 19 {
                45.0
            } else {
                f64::from(i) * step - 45.0
            }
        })
        .collect();
    let lon: Vec<f64> = (0..30).map(|i| f64::from(i) * 12.0).collect();
    let air = Array3::from_shape_fn((731, 20, 30), |(t, y, x)| {
        let la = lat.get(y).copied().unwrap_or(f64::NAN);
        let lo = lon.get(x).copied().unwrap_or(f64::NAN);
        288.0 - 30.0 * la.to_radians().sin().abs()
            + 8.0 * (2.0 * PI * t as f64 / 365.25).sin()
            + lo.to_radians().cos()
    });
    let mut ds = Dataset::new();
    let time = date_range("2000-01-01", 731, "D")?;
    ds.add_coord("time", Variable::new(["time"], time.into_dyn())?)?;
    ds.add_coord("lat", Variable::new(["lat"], Array1::from(lat).into_dyn())?)?;
    ds.add_coord("lon", Variable::new(["lon"], Array1::from(lon).into_dyn())?)?;
    let air = Variable::new(["time", "lat", "lon"], air.into_dyn())?.with_unit("K".parse()?);
    ds.add_var("air", air)?;
    // Compressed, chunked netCDF-4.
    let enc = ds.encoding_of_mut("air")?;
    enc.deflate = Some(4);
    enc.chunks = Some(vec![30, 20, 30]);
    Ok(ds)
}

fn main() -> Result<(), Box<dyn Error>> {
    let tmp = Scratch(std::env::temp_dir().join(format!("annoray-example-{}", std::process::id())));
    std::fs::create_dir_all(&tmp.0)?;
    let path = |name: &str| tmp.0.join(name);

    // One file, and one file per year.
    let ds = dataset()?;
    ds.to_netcdf(path("air.nc"))?;
    isel!(ds[time = 0:366])?.to_netcdf(path("air_2000.nc"))?;
    isel!(ds[time = 366:731])?.to_netcdf(path("air_2001.nc"))?;
    println!("wrote air.nc, air_2000.nc, air_2001.nc");

    // Opening reads metadata only; values stay on disk until used.
    let on_disk = open_dataset(path("air.nc"))?;
    let air: DataArray = on_disk.get("air")?.to_owned();
    println!("opened: lazy={}", yes(air.variable().is_lazy()));

    // Arithmetic and unit conversion stay lazy; writing computes chunk by chunk.
    let celsius = air.to_unit(&"degC".parse()?)?;
    let anomaly = (&celsius - &celsius.mean("time")?)?;
    let unit = anomaly.unit().ok_or("no unit")?.to_string();
    println!(
        "anomaly: lazy={}, units={unit}",
        yes(anomaly.variable().is_lazy())
    );
    let mut out = Dataset::new();
    out.add_array("anomaly", anomaly)?;
    out.to_netcdf(path("anomaly.nc"))?;
    let back = open_dataset(path("anomaly.nc"))?;
    let back = back.get("anomaly")?;
    show(
        "anomaly at (lat 0, lon 0), first days",
        &isel!(back[time = 0:4, lat = 10, lon = 0])?,
    )?;

    // Reductions of on-disk data stream through a memory budget (here 1 MiB at a time).
    annoray::set_stream_bytes(1 << 20);
    let mean = air.mean("time")?;
    show(
        "time-mean air, lat 0..3, lon 0",
        &isel!(mean[lat = 0:3, lon = 0])?,
    )?;
    let std = air.std("time")?;
    show(
        "time-std air, lat 0..3, lon 0",
        &isel!(std[lat = 0:3, lon = 0])?,
    )?;
    let monthly = air.groupby("time.month")?.mean()?;
    show(
        "monthly mean at (lat 10, lon 0)",
        &isel!(monthly[lat = 10, lon = 0])?,
    )?;

    // Many files as one dataset, combined by their coordinates.
    let files = glob(path("air_*.nc").to_str().ok_or("path")?)?;
    let both = open_mfdataset(&files, &MfOptions::default())?.dataset;
    let sizes: Vec<String> = both
        .sizes()
        .iter()
        .map(|(k, n)| format!("{k}={n}"))
        .collect();
    println!("multi-file sizes: {}", sizes.join(", "));
    let one = air.mean(..)?.to_f64()?;
    let two = both.get("air")?.mean(..)?.to_f64()?;
    let means: Vec<String> = one.iter().chain(&two).map(|v| format!("{v:.3}")).collect();
    println!("overall mean (one file, two files): [{}]", means.join(", "));
    Ok(())
}
