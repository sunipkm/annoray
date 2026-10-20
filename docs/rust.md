# Rust crates

The Rust API is documented on docs.rs, versioned with each crates.io release.

| Crate | API | |
|---|---|---|
| `annoray` | [docs.rs/annoray](https://docs.rs/annoray) | Data model, operations, CF encoding and decoding, netCDF/HDF5/Zarr I/O. |
| `annoray-units` | [docs.rs/annoray-units](https://docs.rs/annoray-units) | Units: UDUNITS-2/CF grammar, exact conversion, CF calendars, dimension-typed quantities. |
| `annoray-hdf5` | [docs.rs/annoray-hdf5](https://docs.rs/annoray-hdf5) | Pure-Rust HDF5 reader and writer. |
| `annoray-zarr` | [docs.rs/annoray-zarr](https://docs.rs/annoray-zarr) | Pure-Rust Zarr v2/v3 reader and writer for local stores. |
| `annoray-codecs` | [docs.rs/annoray-codecs](https://docs.rs/annoray-codecs) | Compression codecs shared by the HDF5 and Zarr crates. |
| `annoray-macros` | [docs.rs/annoray-macros](https://docs.rs/annoray-macros) | `isel!` / `sel!`, re-exported by `annoray`. |

The standalone crates do not depend on `annoray`:

```text
annoray ──┬── annoray-units
          ├── annoray-macros
          ├── annoray-hdf5 ──┐
          └── annoray-zarr ──┴── annoray-codecs
```

## Storage representations

`Variable`, `DataArray` and `Dataset` are generic over a representation, as ndarray's arrays
are:

| owned | shared view | mutable view |
|---|---|---|
| `Variable` | `VariableView<'a>` | `VariableViewMut<'a>` |
| `DataArray` | `DataArrayView<'a>` | `DataArrayViewMut<'a>` |
| `Dataset` | `DatasetView<'a>` | `DatasetViewMut<'a>` |

Selections return views; the `_mut` variants write in place under the borrow checker.
`DataArray::put` returns a copy with values replaced, converted into the array's unit and
element type.

## Dates, times and time zones

Date-times are `annoray::Instant`, durations `annoray::Duration`: 64-bit nanosecond counts
(since 1970-01-01 UTC for instants), `#[repr(transparent)]` over `i64`, with numpy's
`datetime64[ns]` and `timedelta64[ns]` layout and range (1677-09-21 to 2262-04-11). `i64::MIN`
is NaT. Their arithmetic is checked (`checked_add`, `duration_since`, …); instants parse from
ISO 8601 (`"2024-07-01T02:00Z"`, with an offset or none for UTC). Dates outside the range, and
dates in other CF calendars, are `CfDate`s.

A variable of date-times carries an optional `TimeZone`: UTC, a fixed offset (`"+05:30"`) or an
IANA zone (`"America/New_York"`) from the system's time zone database, or from a copy built in
with the `tzdb-bundle` feature. The zone sets the wall-clock times of the instants (components,
resampling bins, date-string selection, display) and is kept through operations; the instants
themselves are unchanged by it.

```rust
use annoray::ndarray::Array1;
use annoray::{
    DataArray, Disambiguation, Duration, Instant, TimeZone, Variable, date_range_tz,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t: Instant = "2024-07-01T02:00Z".parse()?;
    let later = t
        .checked_add(Duration::from_hours(6).ok_or("overflow")?)
        .ok_or("overflow")?;

    // Wall-clock hours in New York: 22 (the evening before) and 4.
    let ny: TimeZone = "America/New_York".parse()?;
    let times = Variable::new(["time"], Array1::from(vec![t, later]).into_dyn())?
        .with_time_zone(ny.clone())?;
    println!("{:?}", times.dt("hour")?.data::<i64>()?);

    // Local midnights across a daylight-saving change (24 h, then 23 h apart).
    let days = date_range_tz("2024-03-09", 3, "D", &ny)?;
    let time = Variable::new(["time"], days.into_dyn())?.with_time_zone(ny)?;
    let da = DataArray::from_array(["time"], Array1::from(vec![1.0, 2.0, 3.0]).into_dyn())?
        .with_coord("time", time)?;
    println!("{da}"); // 2024-03-09 00:00:00-05:00 2024-03-10 00:00:00-05:00 2024-03-11 00:00:00-04:00

    // The same instants shown in UTC, and their UTC wall-clock times read as Paris times.
    let utc = times.tz_convert(None)?;
    let paris = utc.tz_localize(Some("Europe/Paris".parse()?), Disambiguation::Compatible)?;
    println!("{paris}"); // 2024-07-01 02:00:00+02:00, 2024-07-01 08:00:00+02:00
    Ok(())
}
```

Mixing zones in `concat`, `merge` or alignment is an error. Files hold the zone in a
`time_zone` attribute. The [time chapter](user-guide/time.md) covers the same model from
Python.

## Errors

No function panics on bad input: every failure is a `Result`. `unwrap`, `expect`, panicking
indexing and unchecked arithmetic are denied by lint across the workspace.
