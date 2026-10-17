---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Overview

annoray's containers are xarray's: a {class}`~annoray.DataArray` is an N-dimensional array
with named dimensions, coordinates and attributes, and a {class}`~annoray.Dataset` is a
dict-like collection of them sharing dimensions. What annoray adds is that the data carries a
physical unit, checked by every operation. The code on this page runs when the documentation
is built.

## Arrays with units

```{code-cell} python
import numpy as np
import annoray as ar

temp = ar.DataArray(
    np.array([[281.2, 283.9, 279.5], [285.0, 287.3, 284.1]]),
    dims=("time", "station"),
    coords={"time": ar.date_range("2026-07-01", 2, "D"), "station": ["a", "b", "c"]},
    name="temperature",
    units="K",
)
temp
```

Units propagate through arithmetic and convert exactly:

```{code-cell} python
print(temp.to("degC").values)
print((temp.mean("time") - temp.min("time")).units)  # a kelvin-sized step
```

Dimensionally inconsistent operations raise instead of producing a number:

```{code-cell} python
try:
    temp + ar.Quantity(3.0, "m")
except ar.UnitError as e:
    print("UnitError:", e)
```

A bare number beside data with a unit is accepted only as 0 or NaN; other values are given
as quantities, so a threshold cannot silently be in the wrong unit:

```{code-cell} python
warm = temp.where(temp > ar.Quantity(10.0, "degC"))
print(warm.values)
```

## Selecting

Positional and label-based selection follow xarray; label slices include both ends.

```{code-cell} python
print(temp.isel(time=0).values)
print(temp.sel(station=["a", "c"], time="2026-07-02").values)
```

## Updating values

Values are immutable from Python: an object's data never changes underneath a reference to
it. {meth}`~annoray.DataArray.put` and {meth}`~annoray.DataArray.put_sel` return an updated
copy, converting the value into the array's unit:

```{code-cell} python
fixed = temp.put_sel(ar.Quantity(9.0, "degC"), station="b", time="2026-07-01")
print(fixed.values)
print(temp.values[0, 1])  # unchanged
```

## Grouping and windows

```{code-cell} python
hourly = ar.DataArray(
    288.0 + 0.1 * np.arange(72) + 5.0 * np.sin(np.arange(72) * 2 * np.pi / 24),
    dims="time",
    coords={"time": ar.date_range("2026-07-01", 72, "h")},
    units="K",
)
print(hourly.resample(time="1D").mean().values)
print(hourly.rolling(time=24).mean().dropna("time").values[:3])
```

## Files

netCDF and Zarr are read and written by annoray's own backends. Files open lazily:
selections and reductions read only what they need, in slabs within a memory budget.

```{code-cell} python
import tempfile, os

path = os.path.join(tempfile.mkdtemp(), "temperature.nc")
temp.to_dataset().to_netcdf(path)

ds = ar.open_dataset(path)  # lazy, memory-mapped
print(ds["temperature"].is_lazy, ds["temperature"].units)
print(ds["temperature"].mean("time").to("degC").values)
```

## In Rust

The same model, with the dimensional check available at compile time as well:

```rust
use annoray::io::open_dataset;
use annoray::sel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ds = open_dataset("temperature.nc")?; // lazy, memory-mapped
    let t = ds.get("temperature")?;
    let a = sel!(t[station = "a"])?;
    println!("{}", a.mean("time")?.to_unit(&"degC".parse()?)?);
    Ok(())
}
```

Paired examples with identical output in both languages: [Examples](examples.md).
