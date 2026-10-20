---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Top-level functions

Each section has a short example; the examples share this setup:

```{code-cell} python
import os
import tempfile

import numpy as np
import annoray as ar

tmp = tempfile.mkdtemp()
temp = ar.DataArray(
    280.0 + np.arange(6.0),
    dims="time",
    coords={"time": ar.date_range("2026-07-01", 6, "D")},
    name="temperature",
    units="K",
)
```

## Opening and saving

```{code-cell} python
temp.to_netcdf(os.path.join(tmp, "t.nc"))
ds = ar.open_dataset(os.path.join(tmp, "t.nc"))  # lazy: values are read when used
print(ds["temperature"].is_lazy, ds["temperature"].units)
print(ar.load_dataset(os.path.join(tmp, "t.nc"))["temperature"].is_lazy)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   open_dataset
   open_dataarray
   open_datatree
   open_groups
   open_mfdataset
   open_zarr
   load_dataset
   load_dataarray
   decode_cf
   to_netcdf
   to_zarr
   save_mfdataset
```

## Writing in place

```{code-cell} python
template = temp.isel(time=slice(0, 0)).to_dataset()
with ar.NetcdfWriter.create(os.path.join(tmp, "run.nc"), template, unlimited_dims=["time"]) as w:
    for i in range(3):
        w.append(temp.isel(time=slice(i, i + 1)).to_dataset(), "time")
    print(w.dim_len("time"))
print(ar.open_dataset(os.path.join(tmp, "run.nc"))["temperature"].values)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   NetcdfWriter
   NetcdfWriter.create
   NetcdfWriter.write
   NetcdfWriter.append
   NetcdfWriter.dim_len
   NetcdfWriter.dataset
   NetcdfWriter.flush
   NetcdfWriter.close
   NetcdfWriter.closed
```

## Combining

```{code-cell} python
first, rest = temp.isel(time=slice(0, 2)), temp.isel(time=slice(2, None))
print(ar.concat([first, rest], dim="time").sizes)
pressure = ar.DataArray([1012.0] * 6, dims="time", coords={"time": temp.time}, name="p", units="hPa")
print(list(ar.merge([temp, pressure]).data_vars))
a, b = ar.align(first, temp, join="inner")
print(a.sizes, b.sizes)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   concat
   merge
   align
   broadcast
   combine_by_coords
   combine_nested
```

## Computation

```{code-cell} python
print(ar.where(temp > ar.Quantity(282.0, "K"), temp, ar.Quantity(0.0, "K")).values)
print(ar.zeros_like(temp).units)
print(ar.corr(temp, temp * 2.0).item())
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   where
   apply_ufunc
   dot
   cross
   corr
   cov
   polyval
   full_like
   zeros_like
   ones_like
   map_blocks
   map_over_datasets
   group_subtrees
   unify_chunks
```

## Time

```{code-cell} python
print(ar.date_range("2026-01-01", periods=3, freq="MS"))
print(ar.date_range("2026-01-01", periods=2, freq="D", calendar="360_day"))
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   date_range
   cftime_range
   infer_freq
```

## Streaming and options

Lazy data (opened from files, or computed from it) is reduced and written in slabs within a
memory budget.

```{code-cell} python
print(ar.stream_bytes())
ar.set_stream_bytes(16 * 2**20)  # reductions of lazy data read 16 MiB at a time
ar.set_stream_bytes(64 * 2**20)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   set_stream_bytes
   stream_bytes
   set_options
   get_options
   show_versions
```

## Interoperability

```{code-cell} python
x = temp.to_xarray()
print(type(x).__name__, x.attrs["units"])
print(ar.from_xarray(x).units)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   from_xarray
   to_astropy
   from_astropy
```

## Extending

```{code-cell} python
@ar.register_dataarray_accessor("climate")
class Climate:
    def __init__(self, da):
        self._da = da

    def anomaly(self):
        return self._da - self._da.mean()

print(temp.climate.anomaly().values)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   register_dataarray_accessor
   register_dataset_accessor
```

