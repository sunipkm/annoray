# annoray

**Annotated arrays**: the [xarray](https://xarray.dev) data model with first-class physical
units, implemented in Rust (PyO3). I/O uses pure-Rust netCDF-3, netCDF-4 and HDF5 backends;
the only runtime dependency is NumPy.

- **Labelled indexing and alignment:** `isel`, `sel` (exact, nearest, pad, backfill),
  name-based broadcasting, `groupby`, `resample`, `rolling`, `interp`, `concat`, `merge`,
  `DataTree`.
- **Units:** binary operations check dimensional consistency and propagate units; `.to(...)`
  converts exactly; CF `units` attributes decode into units.
- **Out-of-core evaluation:** lazy memory-mapped open; lazy elementwise composition and
  streaming reductions within a configurable memory budget (`set_stream_bytes`), including
  disk-to-disk pipelines.
- **Dimension annotations:** `Quantity[Velocity]` / `DataArray[Velocity]` type hints,
  enforced at call time by `@checked`; derived dimensions are declared as `Dim` subclasses.
- **Zero-copy NumPy interop:** `.values` exposes the Rust buffer as a read-only array.

> Pre-1.0: the API is unstable.

```sh
pip install annoray            # "annoray[xarray]" adds xarray interoperability
```

```python
import numpy as np
import annoray as ar

da = ar.DataArray(
    np.arange(6.0).reshape(2, 3),
    dims=("x", "y"),
    coords={"x": [10.0, 20.0], "y": [1.0, 2.0, 3.0]},
    units="m",
)
seconds = ar.DataArray(2.0, units="s")
print((da / seconds).units)  # m s-1
print(da.sel(y=2.4, method="nearest").mean("x").to("cm").values)
try:
    da + seconds
except ar.UnitError as e:  # dimensionally inconsistent
    print(e)

ds = ar.open_dataset("air_temperature.nc")  # lazy, memory-mapped
point = ds["air"].sel(lat=40.0, lon=260.0, method="nearest")
print(point.max("time").to("degC"))
```

xarray is optional. When present, annoray registers an xarray backend
(`engine="annoray"`) and provides `to_xarray()` / `from_xarray()` conversion (units map to the
`units` attribute) and `ar.to_netcdf(obj, path)` for writing xarray objects with annoray's
writers.

Further examples: [`examples/`](https://github.com/sunipkm/annoray/tree/main/annoray-python/examples)
(each paired with a Rust twin of identical output).

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at the licensee's
option.
