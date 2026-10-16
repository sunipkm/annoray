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
from typing import Any

from annoray import dims, testing
from annoray._accessors import register_dataarray_accessor, register_dataset_accessor
from annoray._annoray import (
    Coarsen,
    DataArray,
    Dataset,
    DataTree,
    Dimension,
    GroupBy,
    Quantity,
    Rolling,
    RollingExp,
    Unit,
    UnitError,
    Weighted,
    broadcast,
    combine_by_coords,
    concat,
    corr,
    cov,
    decode_cf,
    dot,
    from_xarray,
    full_like,
    merge,
    open_datatree,
    polyval,
    set_stream_bytes,
    stream_bytes,
    where,
)
from annoray._astropy import from_astropy, to_astropy
from annoray._functions import (
    align,
    cftime_range,
    combine_nested,
    cross,
    date_range,
    get_options,
    infer_freq,
    map_blocks,
    open_dataarray,
    save_mfdataset,
    set_options,
    show_versions,
    unify_chunks,
)
from annoray._io import load_dataarray, load_dataset, open_dataset, open_mfdataset
from annoray._ufunc import apply_ufunc
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
    "RollingExp",
    "Unit",
    "UnitError",
    "Weighted",
    "__version__",
    "align",
    "apply_ufunc",
    "broadcast",
    "cftime_range",
    "combine_by_coords",
    "combine_nested",
    "concat",
    "corr",
    "cov",
    "cross",
    "date_range",
    "decode_cf",
    "dimension",
    "dims",
    "dot",
    "from_astropy",
    "from_xarray",
    "full_like",
    "get_options",
    "infer_freq",
    "load_dataarray",
    "load_dataset",
    "map_blocks",
    "merge",
    "ones_like",
    "open_dataarray",
    "open_dataset",
    "open_datatree",
    "open_mfdataset",
    "open_zarr",
    "polyval",
    "register_dataarray_accessor",
    "register_dataset_accessor",
    "save_mfdataset",
    "set_options",
    "set_stream_bytes",
    "show_versions",
    "stream_bytes",
    "testing",
    "to_astropy",
    "to_netcdf",
    "to_zarr",
    "unify_chunks",
    "where",
    "zeros_like",
]


def zeros_like(other: Any, dtype: Any = None) -> Any:
    """A DataArray or Dataset shaped like ``other``, filled with 0 (in ``other``'s unit)."""
    return full_like(other, 0, dtype=dtype)


def ones_like(other: Any, dtype: Any = None) -> Any:
    """A DataArray or Dataset shaped like ``other``, filled with 1 (in ``other``'s unit)."""
    return full_like(other, 1, dtype=dtype)


def open_zarr(store, group=None, **kwargs):
    """Open a Zarr store (v2 or v3 directory) lazily and CF-decode it; `open_dataset`
    recognises stores too."""
    return open_dataset(store, group=group, **kwargs)


def to_zarr(obj, store, **kwargs) -> None:
    """Write an annoray ``Dataset`` (or ``DataArray``, as a one-variable dataset) as a Zarr
    store; see ``Dataset.to_zarr``."""
    if isinstance(obj, DataArray):
        obj = obj.to_dataset()
    obj.to_zarr(store, **kwargs)


def to_netcdf(obj, path, **kwargs) -> None:
    """Write an annoray ``Dataset`` or ``DataTree`` (or, with xarray installed, an
    ``xarray.Dataset``) with annoray's pure-Rust writers."""
    if isinstance(obj, (Dataset, DataTree)):
        obj.to_netcdf(path, **kwargs)
        return
    # xarray objects: the interop writer imports xarray only now.
    from annoray.writer import to_netcdf as xr_to_netcdf

    xr_to_netcdf(obj, path, **kwargs)
