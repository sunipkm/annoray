# annoray

**Annotated arrays**: the [xarray](https://xarray.dev) data model in Rust, with first-class
physical units and pure-Rust netCDF-3 (CDF-1/2/5), netCDF-4 and HDF5 backends.

- **Storage-generic containers:** `Variable`, `DataArray` and `Dataset` are generic over a
  GAT-based representation (owned, view, mutable view), mirroring `ndarray`; selections
  borrow instead of copying.
- **Labelled indexing and alignment:** `isel!`/`sel!` with Python slice syntax, name-based
  broadcasting, `groupby`, `resample`, `rolling`, `interp`, `concat`, `merge`, `DataTree`.
- **Units:** binary operations check dimensional consistency and propagate units; exact
  conversion; CF `units` attributes decode into units.
- **Out-of-core evaluation:** lazy memory-mapped sources, lazy elementwise composition and
  chunk-aligned streaming reductions within a configurable memory budget.
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
(each paired with a Python twin of identical output).

## Features

| Feature | Default | |
|---|---|---|
| `rayon` | yes | Data-parallel reductions, arithmetic, CF decoding and chunk (de)compression. |
| `zstd`, `lz4` | yes | Zstandard (filter 32015) and LZ4 (filter 32004) chunk codecs, pure Rust. |
| `uom` | no | Arrays and `sel` labels of uom SI quantities: `to_typed`, `from_typed` (values rescaled into coherent SI). |
| `whippyunits` | no | Arrays and `sel` labels of whippyunits quantities; `view_typed` reinterprets data in exactly the quantity's unit without copying. |
| `netcdf-c` | no | netCDF-C/HDF5 C libraries as an alternative backend (static build; requires cmake). |

[`annoray-units`](https://docs.rs/annoray-units) and [`annoray-hdf5`](https://docs.rs/annoray-hdf5)
are standalone crates, re-exported as `annoray::units` and `annoray::hdf5`.

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at the licensee's
option.
