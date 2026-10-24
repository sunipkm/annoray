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

Values are immutable from Python: an array's data never changes underneath a reference to
it. Containers are not: a {class}`~annoray.Dataset` is a mutable collection of immutable
variables, and a {class}`~annoray.DataArray`'s coordinates, attributes and name can be
changed, but its data and unit are fixed for its lifetime.

| | Values | Replaceable in place |
|---|---|---|
| `DataArray` | immutable (`da[...] = v` and `.values[...] = v` raise) | coordinates, `attrs`, `name` |
| `Dataset` | immutable (each variable's) | variables (`ds["v"] = ...`, `del ds["v"]`, `update`), coordinates, `attrs` |

{meth}`~annoray.DataArray.put` (by position) and {meth}`~annoray.DataArray.put_sel` (by
label) return an updated copy, converting the value into the array's unit:

```{code-cell} python
fixed = temp.put_sel(ar.Quantity(9.0, "degC"), station="b", time="2026-07-01")
print(fixed.values)
print(temp.values[0, 1])  # unchanged
```

A dataset has no `put` of its own: a variable is edited with `put` and put back, which
replaces it in the dataset (or `assign` gives a new dataset and leaves this one as it was).
Arrays taken from the dataset before keep their values:

```{code-cell} python
ds = temp.to_dataset()
before = ds["temperature"]
ds["temperature"] = ds["temperature"].put_sel(ar.Quantity(9.0, "degC"), station="b")
print(ds["temperature"].values[:, 1])
print(before.values[:, 1])  # unchanged
```

## Adding values

An array grows along a dimension with {func}`~annoray.concat`; there is no `append`
method. A new value is a one-element array with its coordinate, in any compatible unit
(it is converted into the first array's):

```{code-cell} python
point = ar.DataArray(
    [[9.0, 11.0, 7.5]],
    dims=("time", "station"),
    coords={"time": [np.datetime64("2026-07-03", "ns")], "station": ["a", "b", "c"]},
    units="degC",
)
longer = ar.concat([temp, point], dim="time")
print(longer.sizes, longer.values[-1])
```

Since values are immutable, each `concat` copies everything it joins, so growing an array
one value at a time in a loop takes time quadratic in its length. Values that arrive one
by one are collected first and joined in a single `concat`:

```{code-cell} python
pieces = [temp]
for day in range(3, 6):
    t = np.datetime64("2026-07-01", "ns") + np.timedelta64(day - 1, "D")
    pieces.append(point.assign_coords(time=[t]))
print(ar.concat(pieces, dim="time").sizes)
```

Values streamed to a file go through {class}`~annoray.NetcdfWriter`, whose `append` grows the
file in place without rewriting what is already written.

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

## Times and time zones

Date-times are `datetime64[ns]` instants in UTC. A variable of them can carry a time zone,
which sets the wall-clock times its components, resampling bins and display use; times that
know their zone (whenever's, timezone-aware `datetime`s, pandas') keep it when they become an
array. The [time chapter](../user-guide/time.md) covers the details.

```{code-cell} python
local = hourly.assign_coords(time=hourly.time.dt.tz_convert("America/New_York"))
print(local.time.dt.hour.values[:3])  # 20:00, 21:00, 22:00 on the evening before
print(local.resample(time="1D").mean().sizes)  # local days: one more than in UTC
```

## Complex numbers

`complex64` and `complex128` data work as in numpy, with units: arithmetic, the numpy functions,
`sum`, `mean` and `var` (real, in the unit squared), conversion (both parts scale), and storage
in netCDF-4/HDF5 and Zarr. A single complex value with a unit is a complex
{class}`~annoray.Quantity`, so `.item()` keeps the unit. Complex numbers have no order, so `max` and `<` are errors, and a
cast to a real type refuses to drop the imaginary part silently (see the
[differences from xarray](../xarray-differences.md#element-types)). Interpolation, rolling
windows and cumulative reductions do not support them yet.

```{code-cell} python
z = ar.DataArray([3 + 4j, 1 - 1j], dims="x", units="V")
print(abs(z).values, abs(z).units)
print(z.to("mV").values)
print(z.var().values, z.var().units)
print(repr(z.sum().item()))
try:
    z.max()
except TypeError as e:
    print("TypeError:", e)
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
