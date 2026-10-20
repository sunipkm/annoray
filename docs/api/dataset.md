---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Dataset

A dict-like collection of data arrays sharing dimensions and coordinates.

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Dataset
```

Each section has a short example; the examples share this setup:

```{code-cell} python
import numpy as np
import annoray as ar

ds = ar.Dataset(
    {
        "temperature": (("time", "station"), 280.0 + np.arange(12.0).reshape(4, 3), {"units": "K"}),
        "pressure": ("time", [1012.0, 1009.5, 1004.2, 1007.8], {"units": "hPa"}),
    },
    coords={"time": ar.date_range("2026-07-01", 4, "D"), "station": ["a", "b", "c"]},
)
ds
```

## Attributes

```{code-cell} python
print(ds.sizes)
print(list(ds.data_vars), list(ds.coords))
print(ds["pressure"].units)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.dims
   Dataset.sizes
   Dataset.values
   Dataset.attrs
   Dataset.encoding
   Dataset.coords
   Dataset.data_vars
   Dataset.variables
   Dataset.indexes
   Dataset.xindexes
   Dataset.nbytes
   Dataset.chunks
   Dataset.loc
```

## Contents

```{code-cell} python
# Variables are replaced, never written into: arrays taken earlier keep their values.
edited = ds.copy()
before = edited["pressure"]
edited["pressure"] = edited["pressure"].to("Pa")
print(before.units, edited["pressure"].units)
wind = ar.DataArray([3.0, 5.5, 2.1, 4.0], dims="time", units="m/s")
print(list(ds.assign(wind=wind).data_vars))  # a new dataset; ds is unchanged
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.keys
   Dataset.items
   Dataset.get
   Dataset.update
   Dataset.assign
   Dataset.set_coords
```

## Memory and copying

```{code-cell} python
print(ds.astype("float32")["temperature"].dtype)
print(ds.copy().identical(ds))
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.copy
   Dataset.astype
   Dataset.load
   Dataset.compute
   Dataset.persist
   Dataset.chunk
   Dataset.close
```

## Indexing

```{code-cell} python
print(ds.isel(time=0)["pressure"].item())
print(ds.sel(station="b", time="2026-07-02")["temperature"].item())
print(ds.sortby("station", ascending=False)["station"].values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.isel
   Dataset.sel
   Dataset.head
   Dataset.tail
   Dataset.thin
   Dataset.drop_isel
   Dataset.drop_sel
   Dataset.query
   Dataset.reindex
   Dataset.reindex_like
   Dataset.interp
   Dataset.interp_like
   Dataset.sortby
   Dataset.drop_duplicates
   Dataset.get_index
   Dataset.set_index
   Dataset.reset_index
   Dataset.set_xindex
   Dataset.swap_dims
```

## Missing values

```{code-cell} python
low = ds.where(ds["pressure"] < ar.Quantity(1010.0, "hPa"))
print(low["pressure"].values)
print(low.dropna("time", how="all").sizes)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.isnull
   Dataset.notnull
   Dataset.fillna
   Dataset.dropna
   Dataset.ffill
   Dataset.bfill
   Dataset.interpolate_na
   Dataset.where
   Dataset.combine_first
```

## Comparisons

```{code-cell} python
print(ds.equals(ds.copy()))
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.equals
   Dataset.identical
   Dataset.isin
```

## Aggregation

```{code-cell} python
means = ds.mean("time")
print(means["temperature"].values, means["pressure"].item())
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.all
   Dataset.any
   Dataset.argmax
   Dataset.argmin
   Dataset.count
   Dataset.max
   Dataset.mean
   Dataset.median
   Dataset.min
   Dataset.prod
   Dataset.std
   Dataset.sum
   Dataset.var
   Dataset.quantile
   Dataset.idxmax
   Dataset.idxmin
```

## Cumulative

```{code-cell} python
print(ds.cumsum("time")["pressure"].values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.cumsum
   Dataset.cumprod
   Dataset.cumulative_integrate
```

## Computation

```{code-cell} python
print(ds.diff("time")["pressure"].values)
print(ds.map(lambda v: v.max())["pressure"].item())
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.clip
   Dataset.round
   Dataset.conj
   Dataset.conjugate
   Dataset.real
   Dataset.imag
   Dataset.diff
   Dataset.differentiate
   Dataset.integrate
   Dataset.shift
   Dataset.roll
   Dataset.rank
   Dataset.polyfit
   Dataset.pipe
   Dataset.reduce
   Dataset.map
   Dataset.map_blocks
```

## Grouped and windowed

```{code-cell} python
print(ds.resample(time="2D").mean()["pressure"].values)
print(ds.rolling(time=2).mean()["pressure"].values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.groupby
   Dataset.groupby_bins
   Dataset.resample
   Dataset.rolling
   Dataset.rolling_exp
   Dataset.coarsen
   Dataset.weighted
```

## Reshaping and renaming

```{code-cell} python
print(ds.rename_vars(pressure="p").data_vars.keys())
print(ds.drop_vars("pressure").data_vars.keys())
print(ds.expand_dims("run").sizes)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.transpose
   Dataset.expand_dims
   Dataset.squeeze
   Dataset.stack
   Dataset.unstack
   Dataset.to_stacked_array
   Dataset.pad
   Dataset.rename
   Dataset.rename_dims
   Dataset.rename_vars
   Dataset.drop_vars
   Dataset.drop_dims
   Dataset.assign_coords
   Dataset.assign_attrs
   Dataset.reset_coords
   Dataset.merge
   Dataset.filter_by_attrs
   Dataset.info
```

## Time and text

```{code-cell} python
noleap = ds.convert_calendar("noleap")
print(noleap["time"].values[0])
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.convert_calendar
```

## Plotting

```{code-cell} python
ds.isel(station=0).plot.scatter(x="pressure", y="temperature");
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.plot
```

## Input and output

```{code-cell} python
import os, tempfile

path = os.path.join(tempfile.mkdtemp(), "obs.nc")
ds.to_netcdf(path)
back = ar.open_dataset(path)
print(back["pressure"].units, back["temperature"].is_lazy)
try:
    ds.to_dataarray(dim="variable")  # one array needs one unit
except ar.UnitError as e:
    print("UnitError:", e)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.to_netcdf
   Dataset.to_zarr
   Dataset.to_dataarray
   Dataset.to_array
   Dataset.to_dataframe
   Dataset.to_pandas
   Dataset.to_dict
   Dataset.to_xarray
   Dataset.from_dict
   Dataset.from_dataframe
```
