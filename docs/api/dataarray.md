---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# DataArray

A labelled, units-aware N-dimensional array.

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   DataArray
```

Each section has a short example; the examples share this setup:

```{code-cell} python
import numpy as np
import annoray as ar

temp = ar.DataArray(
    280.0 + np.arange(12.0).reshape(4, 3),
    dims=("time", "station"),
    coords={"time": ar.date_range("2026-07-01", 4, "D"), "station": ["a", "b", "c"]},
    name="temperature",
    units="K",
)
temp
```

## Attributes

```{code-cell} python
print(temp.dims, temp.sizes, temp.dtype)
print(temp.coords["station"].values)
temp.attrs["source"] = "synthetic"  # attributes, coordinates and the name can change
print(temp.attrs)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.dims
   DataArray.sizes
   DataArray.shape
   DataArray.size
   DataArray.ndim
   DataArray.dtype
   DataArray.values
   DataArray.data
   DataArray.name
   DataArray.attrs
   DataArray.encoding
   DataArray.coords
   DataArray.indexes
   DataArray.xindexes
   DataArray.nbytes
   DataArray.chunks
   DataArray.T
   DataArray.is_lazy
   DataArray.loc
```

## Units

```{code-cell} python
print(temp.units, temp.dimension)
print(temp.to("degC").values[0])  # exact conversion
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.units
   DataArray.unit
   DataArray.dimension
   DataArray.to
```

## Memory and copying

```{code-cell} python
print(temp.astype("float32").dtype)
print(temp.isel(time=0, station=0).item())
print(temp.to_numpy().flags.writeable)  # values are read-only views
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.copy
   DataArray.astype
   DataArray.load
   DataArray.compute
   DataArray.persist
   DataArray.chunk
   DataArray.close
   DataArray.item
   DataArray.to_numpy
```

## Indexing

```{code-cell} python
print(temp.isel(time=0).values)
print(temp.sel(station="b", time=slice("2026-07-02", "2026-07-03")).values)
# Values are immutable: put / put_sel return an updated copy, converting the unit.
print(temp.put_sel(ar.Quantity(0.0, "degC"), station="a", time="2026-07-01").values[0])
print(temp.sortby("station", ascending=False).station.values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.isel
   DataArray.sel
   DataArray.head
   DataArray.tail
   DataArray.thin
   DataArray.drop_isel
   DataArray.drop_sel
   DataArray.query
   DataArray.put
   DataArray.put_sel
   DataArray.reindex
   DataArray.reindex_like
   DataArray.interp
   DataArray.interp_like
   DataArray.sortby
   DataArray.drop_duplicates
   DataArray.get_index
   DataArray.set_index
   DataArray.reset_index
   DataArray.set_xindex
   DataArray.swap_dims
   DataArray.broadcast_like
```

## Missing values

```{code-cell} python
gappy = temp.where(temp > ar.Quantity(283.0, "K"))
print(int(gappy.isnull().sum()))
print(gappy.ffill("time").isel(station=0).values)
print(gappy.fillna(ar.Quantity(0.0, "degC")).isel(time=0).values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.isnull
   DataArray.notnull
   DataArray.fillna
   DataArray.dropna
   DataArray.ffill
   DataArray.bfill
   DataArray.interpolate_na
   DataArray.where
   DataArray.combine_first
```

## Comparisons

```{code-cell} python
print(temp.equals(temp.copy()))
print(temp.equals(temp.to("degC")))  # the same quantities in another unit
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.equals
   DataArray.identical
   DataArray.broadcast_equals
   DataArray.isin
```

## Aggregation

```{code-cell} python
print(temp.mean("time").values, temp.mean("time").units)
print(temp.max().item())
print(temp.idxmax("time").values)
print(temp.quantile(0.5, dim="time").values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.all
   DataArray.any
   DataArray.argmax
   DataArray.argmin
   DataArray.count
   DataArray.max
   DataArray.mean
   DataArray.median
   DataArray.min
   DataArray.prod
   DataArray.std
   DataArray.sum
   DataArray.var
   DataArray.quantile
   DataArray.idxmax
   DataArray.idxmin
```

## Cumulative

```{code-cell} python
print(temp.isel(station=0).cumsum("time").values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.cumsum
   DataArray.cumprod
   DataArray.cumulative
   DataArray.cumulative_integrate
```

## Computation

```{code-cell} python
print(temp.diff("time").isel(station=0).values)
print(temp.differentiate("time").units)  # a rate: kelvin per second
print(temp.shift(time=1).isel(station=0).values)
print(temp.clip(max=ar.Quantity(10.0, "degC")).isel(station=0).values)
# Complex data: the parts, the conjugate, and the magnitude in the data's unit.
z = ar.DataArray([3 + 4j, 1 - 1j], dims="x", units="V")
print(z.real.values, z.imag.values, z.conj().values, abs(z).values, abs(z).units)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.clip
   DataArray.round
   DataArray.conj
   DataArray.conjugate
   DataArray.real
   DataArray.imag
   DataArray.diff
   DataArray.differentiate
   DataArray.integrate
   DataArray.shift
   DataArray.roll
   DataArray.rank
   DataArray.dot
   DataArray.polyfit
   DataArray.argsort
   DataArray.pipe
   DataArray.reduce
   DataArray.map_blocks
```

## Grouped and windowed

```{code-cell} python
print(temp.resample(time="2D").mean().values)
print(temp.rolling(time=2).mean().isel(station=0).values)
print(temp.coarsen(time=2).max().sizes)
w = ar.DataArray([1.0, 1.0, 2.0], dims="station")
print(temp.weighted(w).mean("station").values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.groupby
   DataArray.groupby_bins
   DataArray.resample
   DataArray.rolling
   DataArray.rolling_exp
   DataArray.coarsen
   DataArray.weighted
```

## Reshaping and renaming

```{code-cell} python
print(temp.transpose("station", "time").dims)
print(temp.expand_dims("run").sizes)
print(temp.stack(point=("time", "station")).sizes)
print(temp.assign_coords(elevation=("station", [10.0, 250.0, 1200.0])).coords["elevation"].values)
print(temp.rename("t").name)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.transpose
   DataArray.expand_dims
   DataArray.squeeze
   DataArray.stack
   DataArray.unstack
   DataArray.pad
   DataArray.rename
   DataArray.drop_vars
   DataArray.assign_coords
   DataArray.assign_attrs
   DataArray.reset_coords
```

## Time and text

```{code-cell} python
print(temp.time.dt.dayofweek.values)
print(temp.time.dt.tz_convert("America/New_York").dt.hour.values)
print(temp.station.str.upper().values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.convert_calendar
   DataArray.dt
   DataArray.str
```

## Plotting

```{code-cell} python
temp.isel(station=0).plot();
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.plot
```

## Input and output

```{code-cell} python
import os, tempfile

path = os.path.join(tempfile.mkdtemp(), "temperature.nc")
temp.to_netcdf(path)
print(ar.open_dataarray(path).units)
print(list(temp.to_dataset().data_vars))
print(sorted(temp.to_dict()))
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.to_netcdf
   DataArray.to_zarr
   DataArray.to_dataset
   DataArray.to_dataframe
   DataArray.to_series
   DataArray.to_pandas
   DataArray.to_dict
   DataArray.to_xarray
   DataArray.to_astropy
   DataArray.from_dict
   DataArray.from_series
```
