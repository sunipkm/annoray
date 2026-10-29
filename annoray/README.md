# annoray

**Annotated arrays**: the [xarray](https://xarray.dev) data model in Rust, with first-class
physical units and pure-Rust netCDF-3 (CDF-1/2/5), netCDF-4, HDF5 and Zarr (v2/v3) backends.

- **Storage-generic containers:** `Variable`, `DataArray` and `Dataset` are generic over a
  GAT-based representation (owned, view, mutable view), mirroring `ndarray`; selections
  borrow instead of copying.
- **Labelled indexing and alignment:** `isel!`/`sel!` with Python slice syntax, name-based
  broadcasting, `groupby`, `resample`, `rolling`, `rolling_exp`, `interp`, `concat`, `merge`,
  `DataTree`.
- **Checked updates:** mutable views (`isel_mut`, `data_mut`) write in place under the borrow
  checker; `DataArray::put` returns a copy with values replaced, converted into the array's
  unit and element type.
- **Units:** binary operations check dimensional consistency and propagate units; exact
  conversion; CF `units` attributes decode into units.
- **Date-times and time zones:** `Instant` and `Duration` are 64-bit nanosecond counts, the
  layout of numpy's `datetime64[ns]`/`timedelta64[ns]`; a date-time variable carries a
  `TimeZone` (UTC, a fixed offset or an IANA zone) that sets its wall-clock components,
  resampling bins and display (`tz_convert`, `tz_localize`, `date_range_tz`).
- **Out-of-core evaluation:** lazy memory-mapped sources, lazy elementwise composition and
  chunk-aligned streaming reductions within a configurable memory budget. Writing borrows
  values that encoding leaves unchanged, so a dataset is written without a second copy.
- **Total error handling:** no panics in library code (`unwrap`, indexing and unchecked
  arithmetic are denied by lint); every failure is a `Result`.

> Pre-1.0: the API is unstable.

```rust,no_run
use annoray::io::open_dataset;
use annoray::sel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ds = open_dataset("air_temperature.nc")?; // lazy, memory-mapped
    let air = ds.get("air")?;
    let point = sel!(air[lat = nearest(40.0), lon = nearest(260.0)])?;
    let warmest = point.max("time")?;
    println!("{}", warmest.to_unit(&"degC".parse()?)?);
    Ok(())
}
```

Dimension-typed quantities move the dimensional check to compile time:

```rust
use annoray::dims::{Length, Time, Velocity};
use annoray::Quantity;

let d = Quantity::<Length>::new(2.5, "km")?;
let t = Quantity::<Time>::new(30.0, "min")?;
let v: Quantity<Velocity> = (&d / &t)?; // L/T: type-checked as Velocity
assert!((v.to("m/s")?.value() - 1.3889).abs() < 1e-4);
# Ok::<(), annoray::UnitError>(())
```

Further examples: [`examples/`](https://github.com/sunipkm/annoray/tree/main/annoray/examples)
(each paired with a Python twin of identical output). Deliberate departures from xarray's
behaviour are listed in
[differences from xarray](https://github.com/sunipkm/annoray/blob/main/docs/xarray-differences.md);
benchmarks in [performance](https://github.com/sunipkm/annoray/blob/main/docs/performance.md).

## Features

| Feature | Default | |
|---|---|---|
| `netcdf4` | yes | netCDF-4 and HDF5 files: read, written, and written in place (`NetcdfWriter`). Without it, netCDF-3 alone. |
| `zarr` | yes | Zarr v2 and v3 directory stores. |
| `rayon` | yes | Data-parallel reductions, arithmetic, rolling windows, quantiles, CF decoding, and HDF5/Zarr chunk (de)compression. |
| `zstd`, `lz4` | yes | Zstandard and LZ4 chunk codecs (HDF5 filters 32015/32004, Zarr `zstd`/`lz4`, inside Blosc), pure Rust. |
| `uom` | no | Arrays and `sel` labels of uom SI quantities: `to_typed`, `from_typed` (values rescaled into coherent SI). |
| `whippyunits` | no | Arrays and `sel` labels of whippyunits quantities; `view_typed` reinterprets data in exactly the quantity's unit without copying. |
| `netcdf-c` | no | netCDF-C/HDF5 C libraries as an alternative backend (static build; requires cmake). |
| `tzdb-bundle` | no | The IANA time zone database built in, for systems without one (always on Windows); otherwise time zones come from the system's database. |

A format left out is an error that names its feature. Format support is what sets the size of
a program: `open_dataset` detects formats at run time, so it links every reader built in, and
writing picks its writer from a run-time format. A program that opens and writes netCDF-3
files is about 0.9 MB with `default-features = false` and 1.3 MB with `netcdf4` and `zarr`
(release, LTO, stripped, `opt-level = "z"`; macOS arm64). Code a program never calls (a writer,
a codec) is left out by the linker either way.

[`annoray-units`](https://docs.rs/annoray-units) and [`annoray-hdf5`](https://docs.rs/annoray-hdf5)
are standalone crates, re-exported as `annoray::units` and `annoray::hdf5`;
[`annoray-zarr`](https://docs.rs/annoray-zarr) (Zarr stores, behind `annoray::io::zarr`) and
[`annoray-codecs`](https://docs.rs/annoray-codecs) (the compression codecs both file formats
use) are standalone too.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at the licensee's
option.
