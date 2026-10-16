"""Module functions of xarray written on top of annoray's: ``combine_nested``, ``cross``,
``infer_freq``, ``set_options``, ``open_dataarray``, ``save_mfdataset``, ``map_blocks``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

import numpy as np

from annoray._annoray import DataArray, Dataset, concat, merge, open_dataset


def combine_nested(
    datasets: Any,
    concat_dim: Any,
    *,
    join: Literal["inner", "outer", "left", "right", "exact"] = "outer",
    **kwargs: Any,
) -> Any:
    """Combine a nested list of datasets (or arrays): the outer level along the first of
    ``concat_dim``, the next along the second, …; ``None`` merges that level."""
    dims = [concat_dim] if isinstance(concat_dim, (str, type(None))) else list(concat_dim)

    def go(objs: Any, depth: int) -> Any:
        if depth == len(dims):
            return objs
        if not isinstance(objs, (list, tuple)):
            raise TypeError(f"combine_nested: expected {len(dims)} levels of lists")
        parts = [go(o, depth + 1) for o in objs]
        dim = dims[depth]
        if dim is None:
            return merge(parts, join=join)
        return concat(parts, dim)

    return go(datasets, 0)


def cross(a: DataArray, b: DataArray, *, dim: str) -> DataArray:
    """The cross product of two 3-vectors along ``dim`` (units multiply)."""
    for v in (a, b):
        if v.sizes.get(dim) != 3:
            raise ValueError(f"cross needs {dim!r} of length 3 in both arrays")

    def c(v: DataArray, i: int) -> DataArray:
        return v.isel({dim: i}).drop_vars(dim, errors="ignore")

    parts = [
        c(a, 1) * c(b, 2) - c(a, 2) * c(b, 1),
        c(a, 2) * c(b, 0) - c(a, 0) * c(b, 2),
        c(a, 0) * c(b, 1) - c(a, 1) * c(b, 0),
    ]
    out = concat([p.expand_dims(dim) for p in parts], dim)
    if dim in a.coords:
        out = out.assign_coords({dim: a.coords[dim]})
    return out


def infer_freq(index: Any) -> str | None:
    """The frequency string of regularly spaced times (pandas' ``infer_freq``), from a 1-D
    DataArray of times or a sequence of them; ``None`` if irregular."""
    import pandas as pd

    values = np.asarray(index.values if isinstance(index, DataArray) else index)
    if values.dtype.kind != "M":
        raise TypeError("infer_freq needs datetime64 values")
    return pd.infer_freq(pd.DatetimeIndex(values))


# Options that only change speed or display: accepted, with no effect.
_FREE = {
    "display_width",
    "display_max_rows",
    "display_values_threshold",
    "display_style",
    "display_expand_attrs",
    "display_expand_coords",
    "display_expand_data_vars",
    "display_expand_data",
    "display_expand_indexes",
    "display_default_indexes",
    "display_max_children",
    "display_max_html_elements",
    "cmap_divergent",
    "cmap_sequential",
    "file_cache_maxsize",
    "warn_for_unclosed_files",
    "use_bottleneck",
    "use_numbagg",
    "use_flox",
    "use_opt_einsum",
    "chunk_manager",
    "netcdf_engine_order",
    "facetgrid_figsize",
}
# Options whose values annoray always behaves by: these are accepted, others refused.
_FIXED: dict[str, tuple[Any, ...]] = {
    "keep_attrs": (True, "default"),
    "arithmetic_join": ("inner",),
    "arithmetic_broadcast": (True,),
    "enable_cftimeindex": (True,),
}


class set_options:
    """xarray's ``set_options``, as a call or a ``with`` block. annoray has no global
    options: display and performance options are accepted with no effect; options that
    change results are accepted only at the value annoray always uses (``keep_attrs=True``,
    ``arithmetic_join="inner"``, …) and refused otherwise."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in _FREE:
                continue
            if k in _FIXED:
                if v not in _FIXED[k]:
                    raise ValueError(
                        f"annoray always uses {k}={_FIXED[k][0]!r}; {v!r} is not supported"
                    )
                continue
            raise ValueError(f"unknown option {k!r}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def open_dataarray(filename_or_obj: Any, **kwargs: Any) -> DataArray:
    """Open a file holding one data variable, as a DataArray."""
    ds = open_dataset(filename_or_obj, **kwargs)
    names = list(ds.data_vars)
    if len(names) != 1:
        raise ValueError(
            f"open_dataarray needs a file with exactly one data variable, found {names}; "
            "use open_dataset"
        )
    da = ds[names[0]]
    if da.name == "__xarray_dataarray_variable__":
        da = da.rename(None)
    return da


def save_mfdataset(
    datasets: Sequence[Dataset], paths: Sequence[Any], mode: str = "w", **kwargs: Any
) -> None:
    """Write each dataset to its path."""
    if mode != "w":
        raise ValueError("save_mfdataset writes new files (mode='w')")
    if len(datasets) != len(paths):
        raise ValueError("save_mfdataset needs one path per dataset")
    if kwargs.pop("groups", None) is not None:
        raise NotImplementedError("groups")
    for ds, path in zip(datasets, paths, strict=True):
        ds.to_netcdf(path, **kwargs)


def map_blocks(
    func: Callable[..., Any],
    obj: Any,
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
    template: Any = None,
) -> Any:
    """``func(obj, *args, **kwargs)``: without dask the object is a single block, which is
    what xarray does for unchunked data."""
    return func(obj, *args, **(kwargs or {}))
