"""rarray: xarray-rs for Python.

Native, units-aware labelled arrays, plus pure-Rust netCDF-3 / netCDF-4 / HDF5 I/O:

    import rarray

    ds = rarray.open_dataset("data.nc")            # a native rarray.Dataset (lazy)
    t = ds["temp"].sel(time="2020-01").mean("time")
    t.to("degF")                                   # unit conversion; units are checked
    ds.to_netcdf("out.nc")

Interoperates with xarray both ways:

    xr.open_dataset("data.nc", engine="rarray")    # xarray objects, rarray's readers
    rarray.to_netcdf(xr_ds, "out.nc")               # xarray objects, rarray's writer
    rarray.from_xarray(xr_ds) / ds.to_xarray()
"""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from rarray._rarray import (
    DataArray,
    Dataset,
    GroupBy,
    Rolling,
    UnitError,
    __version__,
    align,
    concat,
    from_xarray,
    merge,
    open_dataset,
    open_mfdataset,
    set_stream_bytes,
    stream_bytes,
)
from rarray.backend import RarrayBackendEntrypoint
from rarray.writer import to_netcdf as _xr_to_netcdf

__all__ = [
    "DataArray",
    "Dataset",
    "GroupBy",
    "Quantity",
    "RarrayBackendEntrypoint",
    "Rolling",
    "UnitError",
    "__version__",
    "align",
    "concat",
    "from_xarray",
    "merge",
    "open_dataset",
    "open_datatree",
    "open_mfdataset",
    "set_stream_bytes",
    "stream_bytes",
    "to_netcdf",
]


@dataclass(frozen=True)
class Quantity:
    """A number with a unit, for arithmetic and labels: ``da + Quantity(2, "km")``,
    ``da.sel(x=Quantity(500, "m"))``. pint quantities work the same way."""

    magnitude: float
    units: str


def to_netcdf(ds, path, **kwargs) -> None:
    """Write an ``rarray.Dataset`` or an ``xarray.Dataset`` with rarray's writers."""
    if isinstance(ds, Dataset):
        ds.to_netcdf(path, **kwargs)
    else:
        _xr_to_netcdf(ds, path, **kwargs)


def open_datatree(path, **kwargs) -> xr.DataTree:
    """``xr.open_datatree(path, engine="rarray", **kwargs)``."""
    return xr.open_datatree(path, engine="rarray", **kwargs)
