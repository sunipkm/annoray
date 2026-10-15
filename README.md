# annoray

**Annotated arrays**: the xarray data model (labelled N-d arrays with named dimensions,
coordinates and attributes) in Rust, with first-class physical units and Python bindings.
I/O goes through pure-Rust netCDF-3 (CDF-1/2/5), netCDF-4 and HDF5 backends; no C
dependencies.

- **Labelled indexing and alignment:** positional and label-based selection (`isel`/`sel`,
  inclusive label slices, nearest/pad/backfill), name-based broadcasting and outer/inner/exact
  joins, `groupby`, `resample` (all CF calendars), `rolling`, `interp`, `concat`, `merge`.
- **Units in the type system of the data:** binary operations check dimensional consistency
  and propagate units; conversions use exact prime-factorised scales; affine (°C) and epoch
  units are handled; CF `units` attributes are decoded into units on open.
- **Out-of-core evaluation:** lazy, memory-mapped open; elementwise operations compose into
  lazy sources and reductions stream chunk-aligned slabs (single-pass moments) within a
  configurable memory budget, including disk-to-disk pipelines.
- **Total error handling:** malformed input, corrupt files and unit mismatches surface as
  `Result` errors; library code does not panic.

> Pre-1.0: the API is unstable.

## Rust

```toml
[dependencies]
annoray = "0.1"
```

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

## Python

```sh
pip install annoray            # "annoray[xarray]" adds xarray interoperability
```

```python
import annoray as ar

ds = ar.open_dataset("air_temperature.nc")
point = ds["air"].sel(lat=40.0, lon=260.0, method="nearest")
print(point.max("time").to("degC"))
```

## Crates

| Crate | |
|---|---|
| [`annoray`](annoray) | Data model, CF encode/decode, netCDF I/O backends. |
| [`annoray-units`](crates/annoray-units) | Standalone units library: UDUNITS-2/CF grammar, exact conversion, CF calendars, dimension-typed quantities. |
| [`annoray-hdf5`](crates/annoray-hdf5) | Standalone pure-Rust HDF5 reader/writer. |
| [`annoray-macros`](crates/annoray-macros) | `isel!`/`sel!` proc macros, re-exported by `annoray`. |

The Python bindings (PyO3, maturin) are in [`annoray-python`](annoray-python). The examples
are Rust/Python pairs with identical output: [`annoray/examples`](annoray/examples),
[`annoray-python/examples`](annoray-python/examples).

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at the licensee's
option.
