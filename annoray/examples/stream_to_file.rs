//! Streaming to a new file: a netCDF-4 file made from a template and filled in as the
//! values are produced, readable while it is written.
//!
//! `NetcdfWriter::create` writes the template's dimensions, variables, attributes, encodings
//! and coordinates; the data variables read as fill values until written. Each step is
//! appended along the unlimited `time` dimension, a region is rewritten in place, and the
//! file is read back at any point. The Python twin is
//! `annoray-python/examples/stream_to_file.py`; both print the same lines.
//!
//! ```text
//! cargo run --release --example stream_to_file
//! ```
// annoray's operators return `Result` (never panic), which `arithmetic_side_effects` cannot
// tell; the casts are small loop counters.
#![allow(clippy::cast_precision_loss, clippy::arithmetic_side_effects)]

use std::error::Error;
use std::path::PathBuf;

use annoray::io::{NetcdfWriter, WriteOptions, open_dataset};
use annoray::ndarray::{Array1, Array3};
use annoray::{DataArrayBase, Dataset, Repr, Variable, date_range, isel};

const NLAT: usize = 4;
const NLON: usize = 6;

/// Values with two decimals, as the Python twin prints them.
fn show<S: Repr>(label: &str, da: &DataArrayBase<S>) -> Result<(), annoray::CoreError> {
    let v: Vec<String> = da.to_f64()?.iter().map(|v| format!("{v:.2}")).collect();
    println!("{label}: [{}]", v.join(", "));
    Ok(())
}

/// A scratch directory, removed when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `hours` hourly fields of temperature from hour `first`, in kelvin.
fn fields(first: usize, hours: usize) -> Result<Dataset, Box<dyn Error>> {
    let temp = Array3::from_shape_fn((hours, NLAT, NLON), |(h, la, lo)| {
        (280.0 + (first + h) as f64 + 0.5 * la as f64 + 0.1 * lo as f64) as f32
    });
    let start = format!("2000-01-01T{first:02}:00");
    let time = date_range(&start, hours, "h")?;
    let mut ds = Dataset::new();
    ds.add_coord("time", Variable::new(["time"], time.into_dyn())?)?;
    let temp = Variable::new(["time", "lat", "lon"], temp.into_dyn())?.with_unit("K".parse()?);
    ds.add_var("temp", temp)?;
    Ok(ds)
}

fn main() -> Result<(), Box<dyn Error>> {
    let tmp = Scratch(std::env::temp_dir().join(format!("annoray-stream-{}", std::process::id())));
    std::fs::create_dir_all(&tmp.0)?;
    let path = tmp.0.join("run.nc");

    // The template: no time steps yet, the grid's coordinates, and how to store the values.
    let mut template = fields(0, 0)?;
    let lat = Array1::from_iter((0..NLAT).map(|i| -30.0 + 20.0 * i as f64));
    let lon = Array1::from_iter((0..NLON).map(|i| 60.0 * i as f64));
    template.add_coord("lat", Variable::new(["lat"], lat.into_dyn())?)?;
    template.add_coord("lon", Variable::new(["lon"], lon.into_dyn())?)?;
    template.encoding_mut().unlimited_dims = vec!["time".into()];
    template.encoding_of_mut("time")?.time_units = Some("hours since 2000-01-01".into());
    template.encoding_of_mut("temp")?.deflate = Some(4);

    let mut w = NetcdfWriter::create(&path, &template, &WriteOptions::default())?;
    println!(
        "created run.nc: time={} (unlimited), lat={NLAT}, lon={NLON}",
        w.dim_len("time")?
    );

    // Each hour's field is appended as soon as it exists.
    for hour in 0..6 {
        w.append(&fields(hour, 1)?, "time")?;
        if hour == 2 {
            // The file so far, opened lazily while writing goes on.
            let now = w.dataset()?;
            let temp = now.get("temp")?;
            show(
                "after 3 hours, temp at (lat 0, lon 0)",
                &isel!(temp[lat = 0, lon = 0])?,
            )?;
        }
    }
    println!("appended 6 hours: time={}", w.dim_len("time")?);

    // A correction to part of hour 2: the region's own coordinates are left as they are.
    let mut fix = Dataset::new();
    let warm = Array3::<f32>::from_elem((1, NLAT, 3), 300.0);
    fix.add_var(
        "temp",
        Variable::new(["time", "lat", "lon"], warm.into_dyn())?.with_unit("K".parse()?),
    )?;
    w.write(&fix, &[("time".into(), 2), ("lon".into(), 3)])?;
    println!("rewrote hour 2 at lon 3..6");
    w.close()?;

    let back = open_dataset(&path)?;
    let temp = back.get("temp")?;
    show(
        "temp at (lat 1, lon 4), hour by hour",
        &isel!(temp[lat = 1, lon = 4])?,
    )?;
    let time = back.coords().get("time").ok_or("time")?;
    println!(
        "on disk: time={}, units={}, time units={}",
        back.sizes().get("time").copied().unwrap_or(0),
        temp.unit().map_or_else(String::new, ToString::to_string),
        time.encoding().time_units.as_deref().unwrap_or("")
    );
    Ok(())
}
