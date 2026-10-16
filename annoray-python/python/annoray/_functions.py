"""Module functions of xarray written on top of annoray's: ``combine_nested``, ``cross``,
``infer_freq``, ``set_options``, ``open_dataarray``, ``save_mfdataset``, ``map_blocks``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Self, cast

import numpy as np

from annoray._annoray import DataArray, Dataset, concat, merge


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
    from annoray._io import open_dataset

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


def date_range(
    start: Any = None,
    end: Any = None,
    periods: Any = None,
    freq: str | None = "D",
    *,
    calendar: str = "standard",
    use_cftime: bool | None = None,
    inclusive: str = "both",
    normalize: bool = False,
    name: str | None = None,
    tz: Any = None,
    unit: str = "ns",
) -> Any:
    """Regular dates (xarray's ``date_range``; annoray's ``date_range(start, periods, freq)``
    positional form also works): numpy ``datetime64`` for the standard
    calendars (through pandas, as xarray; ``periods`` without pandas), CF dates for other
    calendars or with ``use_cftime=True`` (fixed steps and period starts ``MS``/``QS``/``YS``,
    from ``start`` with ``periods`` or up to ``end``)."""
    from annoray._annoray import _date_range_in
    from annoray._annoray import date_range as _native

    # annoray's earlier positional form, date_range(start, periods, freq): an integer
    # second argument cannot be an end date.
    if isinstance(end, int) and not isinstance(end, bool):
        if isinstance(periods, str):
            freq = periods
        end, periods = None, end
    freq = "D" if freq is None else freq
    standard = calendar in ("standard", "gregorian", "proleptic_gregorian")
    if standard and not use_cftime:
        try:
            import pandas as pd
        except ImportError:
            if start is None or periods is None or end is not None:
                raise ImportError(
                    "date_range needs pandas for this form; give start and periods"
                ) from None
            return _native(str(start), periods, freq)
        return np.asarray(
            pd.date_range(
                start,
                end,
                periods,
                freq,
                tz=tz,
                normalize=normalize,
                name=name,
                inclusive=cast(Any, inclusive),
                unit=cast(Any, unit),
            ).values
        )
    if use_cftime is False:
        raise ValueError(f"use_cftime=False needs a standard calendar, not {calendar!r}")
    if start is None:
        raise ValueError("date_range in a CF calendar needs start")
    dates = _date_range_in(str(start), None if end is None else str(end), periods, freq, calendar)
    if (
        inclusive in ("left", "neither")
        and end is not None
        and len(dates)
        and str(dates[-1])[:10] == str(end)[:10]
    ):
        dates = dates[:-1]
    if inclusive in ("right", "neither") and len(dates):
        dates = dates[1:]
    return dates


def cftime_range(*args: Any, calendar: str = "standard", **kwargs: Any) -> Any:
    """``date_range`` as CF dates (xarray's deprecated ``cftime_range``)."""
    return date_range(*args, calendar=calendar, use_cftime=True, **kwargs)


def align(
    *objects: Any,
    join: Literal["inner", "outer", "left", "right", "exact"] = "inner",
    exclude: Any = (),
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Align objects on their shared dimension coordinates; dimensions in ``exclude`` are
    left as they are (not aligned)."""
    from annoray._annoray import align as _align

    copy = kwargs.pop("copy", True)
    del copy
    if kwargs:
        raise TypeError(f"align() got unexpected keyword arguments {sorted(kwargs)}")
    exclude = [exclude] if isinstance(exclude, str) else list(exclude)
    if not exclude:
        return _align(*objects, join=join)
    # Excluded dimensions get a distinct name per object, so nothing aligns them.
    renamed = []
    for i, o in enumerate(objects):
        names = {d: f"__annoray_excluded_{i}_{d}" for d in exclude if d in o.dims}
        renamed.append((o.rename(names) if names else o, names))
    out = _align(*[o for o, _ in renamed], join=join)
    return tuple(
        o.rename({v: k for k, v in names.items()}) if names else o
        for o, (_, names) in zip(out, renamed, strict=True)
    )


def unify_chunks(*objects: Any) -> tuple[Any, ...]:
    """No-op without dask: the objects as they are."""
    return objects


def get_options() -> dict[str, Any]:
    """The options annoray always uses (see ``set_options``)."""
    return {k: v[0] for k, v in _FIXED.items()}


def show_versions(file: Any = None) -> None:
    """The versions of annoray, Python, numpy and the optional dependencies."""
    import importlib.metadata
    import platform
    import sys

    lines = [f"python: {sys.version.split()[0]} ({platform.platform()})"]
    for pkg in ("annoray", "numpy", "pandas", "xarray", "cftime", "matplotlib", "astropy", "pint"):
        try:
            lines.append(f"{pkg}: {importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{pkg}: None")
    print("\n".join(lines), file=file or sys.stdout)
