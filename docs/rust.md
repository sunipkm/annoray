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
| `annoray-pyo3` | [docs.rs/annoray-pyo3](https://docs.rs/annoray-pyo3) | Exchange annoray values with the `annoray` Python package from other PyO3 extension modules. |

The standalone crates do not depend on `annoray`; `annoray-pyo3` builds on it:

```text
annoray-pyo3 ── annoray ──┬── annoray-units
                          ├── annoray-macros
                          ├── annoray-hdf5 ──┐
                          └── annoray-zarr ──┴── annoray-codecs
```

## Python extension modules

An extension module written with PyO3 exchanges annoray values with the `annoray` Python
package through `annoray-pyo3`: its function arguments and results are `ArUnit`,
`ArQuantity`, `ArDataArray`, `ArDataset` and `ArDataTree`, which convert to and from the
package's own `Unit`, `Quantity`, `DataArray`, `Dataset` and `DataTree` objects. Values cross
as numpy arrays, copied once into the side that keeps them; `ArDataArrayRef` reads an array's
numbers in place. The first conversion checks that the installed `annoray` is compatible.

```rust
use annoray_pyo3::{ArDataArray, ArUnit, unit_error};
use pyo3::prelude::*;

/// `da` converted to `unit` (an `annoray.Unit` or a unit string) in Rust.
#[pyfunction]
fn convert(py: Python<'_>, da: ArDataArray, unit: ArUnit) -> PyResult<ArDataArray> {
    let out = da.0.to_unit(&unit.0).map_err(|e| unit_error(py, e))?;
    Ok(ArDataArray(out))
}
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
`data::<T>()` gives an ndarray view in any layout; `values_slice::<T>()` gives the values as
one row-major slice, borrowed when they are stored that way and copied otherwise.
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

Quantities are `Copy`: the unit is a `UnitId`, a handle to a unit kept once per spelling in a
process-wide table, so a unit string is parsed once and a handle copies as a pointer. The table
only grows; `set_intern_cap` bounds it (65,536 units by default), and interning past the cap is
`UnitError::InternFull`, while plain `Unit` values keep working.

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

## Writing files

How a variable is stored (chunks, compression, packing, fill values) is its `Encoding`, read
from files and used when writing. Two ways set it without copying data:

- `WriteOptions::compress` gives a default for the whole file: a `Compress` applies to every
  variable whose encoding sets no compression, from 256 values up (`min_elements`), except
  strings and variable-length variables, which netCDF-C cannot filter. Chunk shapes are
  netCDF-C's defaults unless an encoding sets them; netCDF-3 ignores the option.
- `encoding_of_mut` on a view (`ds.view()`) sets one variable's encoding for this write: the
  view borrows the values and has its own encodings.

```rust
use annoray::io::{Format, ToNetcdf, WriteOptions};
use annoray::{Compress, Compression};

let opts = WriteOptions::format(Format::Netcdf4).compress(Compress::deflate(4));
ds.to_netcdf_with(&path, &opts)?;

let mut view = ds.view();
let p = view.encoding_of_mut("p")?;
p.compression = Some(Compression::Lz4);
p.chunks = Some(vec![25, 50]);
view.to_netcdf_with(&path, &opts)?;
```

Zarr stores compress every array with `ZarrOptions::compressor` (Zstandard by default).

## Units from files

A `units` attribute that parses becomes the variable's unit (`Variable::unit`) and is removed
from its attributes; writing puts it back. One that does not parse stays a plain attribute, the
variable has no unit, and decoding warns (`Opened::warnings`: `units "…" kept as an attribute:
…`); the attribute is written back unchanged. `DecodeOptions::decode_units = false` keeps every
`units` attribute as text.

## Schemas

`annoray::schema::Schema` states what a dataset must hold: its variables and coordinates
(dimensions, element types, units, sizes, attributes) and attributes. `check` returns a
`SchemaError` listing every problem found; only metadata is read. Variables not named are
allowed unless the schema is `closed()`.

```rust
use annoray::dims::Length;
use annoray::schema::{Schema, Var};

let schema = Schema::new()
    .attr_eq("format", "prism-map")
    .attr_in("version", ["1", "2"])
    .coord("x", Var::of::<f64>(["x"]).dimension::<Length>())
    .var("intensity", Var::of::<f32>(["y", "x"]).unit("1"))
    .optional("mask", Var::of::<bool>(["y", "x"]));
schema.check(&ds)?;
```

`#[derive(Schema)]` states the same of a struct whose fields borrow from the dataset:
`ArrayView`s of the values (their element type and dimensionality checked), `&VariableBase`
for the unit with the values, attributes (`&str`, `String`, `f64`, `i64`), and `Option`s of
these for what may be absent. `view` checks the dataset, then borrows; lazy variables are read
then, and only those named.

```rust
use annoray::dims::Length;
use annoray::ndarray::{ArrayView1, ArrayView2};

#[derive(annoray::schema::Schema)]
#[schema(attr_eq("format", "prism-map"), attr_in("version", "1", "2"))]
struct MapData<'a> {
    #[schema(coord, dims = ["x"], dimension = Length)]
    x: ArrayView1<'a, f64>,
    #[schema(dims = ["y", "x"], unit = "1")]
    intensity: ArrayView2<'a, f32>,
    #[schema(dims = ["y", "x"])]
    mask: Option<ArrayView2<'a, bool>>,
    #[schema(attr)]
    version: &'a str,
}

let map = MapData::view(&ds)?; // every problem at once, as `annoray::Error::Schema`
let first: f32 = map.intensity[[0, 0]];
let schema = MapData::schema(); // the same requirements as a `Schema`
```

## Files from untrusted sources

`OpenOptions::limits` holds a file to a `ReadLimits`: the most bytes of one read (a slice,
a whole variable, or one stored chunk), the bytes of values the file declares in all, values
and dimensions per variable, variables, and attributes per variable or group. The metadata is
checked before any values are read, and every read after that; a breach is
`CoreError::Limit` (`Error::core` finds it inside the wrappers). netCDF-4 string, compound
and variable-length variables are read while opening: one over `max_read_bytes` is skipped
with a warning instead. `ReadLimits::untrusted()` suits files from users or the network.

```rust
use annoray::CoreError;
use annoray::io::{OpenOptions, ReadLimits, open_dataset_with};

let opts = OpenOptions::default().limits(ReadLimits::untrusted().max_total_bytes(64 << 20));
match open_dataset_with(&path, &opts) {
    Ok(opened) => { /* … */ }
    Err(e) if matches!(e.core(), Some(CoreError::Limit { .. })) => { /* refuse the upload */ }
    Err(e) => return Err(e.into()),
}
```

The readers' fuzz targets are in `fuzz/` (`cargo +nightly fuzz run read_bytes`).

## Errors

No function panics on bad input: every failure is a `Result`. `unwrap`, `expect`, panicking
indexing and unchecked arithmetic are denied by lint across the workspace.
