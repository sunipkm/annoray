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

## Complex numbers

`DType::C64` and `DType::C128` hold `num_complex::Complex<f32>` and `Complex<f64>` (re-exported
as `annoray::num_complex`), numpy's `complex64` and `complex128`. They support `+ - * /` with
any numeric operand, the complex functions of `MathFn` (with `Conj`, `Real`, `Imag` and
`Angle`), `sum`/`prod`/`mean` (complex) and `var`/`std` (real), casts, `==`/`!=`, missing
values (NaN in either part), and storage in netCDF-4/HDF5 and Zarr. Unit conversion
(`to_unit`) scales both parts, and `complex64` stays `complex64`; a unit with an offset (°C)
cannot be converted. Casts to a real type fail for a value with a non-zero imaginary part.
`where` and filling missing values work as for floats. They have no order: `min`, `max`,
sorting and `<` are errors. Interpolation (`interp`, `interpolate_na`), rolling windows and
cumulative reductions are not supported for them yet, and refuse them with an error.
netCDF-4/HDF5 complex variables are read when the file is opened, not lazily; Zarr's are read
lazily.

A single complex value with a unit is a `Quantity<K, Complex<f64>>` (`Quantity`'s second
parameter is its number type, `f64` by default). It follows the rules of real quantities, typed
dimensions included, with the differences of complex arrays: conversion scales both parts (a
unit with an offset is `UnitError::ComplexOffset`), and there is no order (`<` does not
compile). `re`, `im` and `norm` are real quantities in the same unit, `arg` is in radians, and
a real quantity becomes a complex one with `into()`.

```rust
use annoray::num_complex::Complex;
use annoray::units::Quantity;
use annoray::units::dims::{Current, Voltage};

let v = Quantity::<Voltage, _>::new(Complex::new(3.0, 4.0), "V")?;
let i = Quantity::<Current, _>::new(Complex::new(0.0, 2.0), "A")?;
let z = (&v / &i)?; // an impedance, dimension checked by the compiler
assert_eq!(z.value(), Complex::new(2.0, -1.5));
assert_eq!(v.to("mV")?.value(), Complex::new(3000.0, 4000.0));
assert_eq!(v.norm().to_string(), "5 V");
# Ok::<(), annoray::units::UnitError>(())
```

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
