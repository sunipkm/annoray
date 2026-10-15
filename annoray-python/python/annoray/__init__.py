"""annoray: annotated arrays: labelled, units-aware N-dimensional arrays, in Rust.

    import annoray as ar

    ds = ar.open_dataset("data.nc")               # lazy: values stay on disk until used
    t = ds["temp"].sel(time="2020-01").mean("time")
    t.to("degF")                                   # units are tracked and checked
    ds.to_netcdf("out.nc")                         # pure-Rust netCDF/HDF5 writers

annoray does not need xarray. With xarray installed (``pip install annoray[xarray]``), the
two interoperate: ``xr.open_dataset(path, engine="annoray")`` reads with annoray's readers,
``ds.to_xarray()`` / ``ar.from_xarray(obj)`` convert (units become ``"units"`` attributes),
and ``ar.to_netcdf(xr_dataset, path)`` writes an xarray dataset with annoray's writers.
"""

from __future__ import annotations

from importlib.metadata import version as _version

from annoray import dims
from annoray._annoray import (
    Coarsen,
    DataArray,
    Dataset,
    DataTree,
    Dimension,
    GroupBy,
    Quantity,
    Rolling,
    Unit,
    UnitError,
    Weighted,
    align,
    concat,
    date_range,
    from_xarray,
    merge,
    open_dataset,
    open_datatree,
    open_mfdataset,
    set_stream_bytes,
    stream_bytes,
)
from annoray._astropy import from_astropy, to_astropy
from annoray.dims import dimension

# As pip reports it (PEP 440: 0.2.1.dev7, not Cargo's 0.2.1-dev.7); see tools/version.py.
__version__ = _version("annoray")

__all__ = [
    "Coarsen",
    "DataArray",
    "DataTree",
    "Dataset",
    "Dimension",
    "GroupBy",
    "Quantity",
    "Rolling",
    "Unit",
    "UnitError",
    "Weighted",
    "__version__",
    "align",
    "concat",
    "date_range",
    "dimension",
    "dims",
    "from_astropy",
    "from_xarray",
    "merge",
    "open_dataset",
    "open_datatree",
    "open_mfdataset",
    "set_stream_bytes",
    "stream_bytes",
    "to_astropy",
    "to_netcdf",
]


def to_netcdf(obj, path, **kwargs) -> None:
    """Write an annoray ``Dataset`` or ``DataTree`` (or, with xarray installed, an
    ``xarray.Dataset``) with annoray's pure-Rust writers."""
    if isinstance(obj, (Dataset, DataTree)):
        obj.to_netcdf(path, **kwargs)
        return
    # xarray objects: the interop writer imports xarray only now.
    from annoray.writer import to_netcdf as xr_to_netcdf

    xr_to_netcdf(obj, path, **kwargs)
