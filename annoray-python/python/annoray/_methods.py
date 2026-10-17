"""DataArray and Dataset methods written on top of the native operations (xarray's
``head``/``tail``/``thin``, ``drop_isel``, ``drop_dims``, ``reset_coords``,
``filter_by_attrs``, ``to_dataarray``, ``to_dataset(dim=)``, ``interp_like``,
``reindex_like``, ``combine_first``, ``fillna`` with arrays, ``pad``, ``rank``,
``cumulative``, ``reduce``, ``nbytes``, ``indexes``/``get_index``, ``put``/``put_sel``, and
the no-op ``chunk`` of an engine without dask).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import Any, cast

import numpy as np

from annoray._annoray import DataArray, Dataset, Quantity, UnitError


def _indexers(obj: Any, indexers: Any, kwargs: dict[str, Any], default: int) -> dict[str, int]:
    if indexers is None and not kwargs:
        return {d: default for d in obj.dims}
    if isinstance(indexers, int):
        return {d: indexers for d in obj.dims}
    return {**(indexers or {}), **kwargs}


def head(obj: Any, indexers: Any = None, **kwargs: int) -> Any:
    """The first ``n`` positions along each dimension (5 by default)."""
    return obj.isel({d: slice(None, n) for d, n in _indexers(obj, indexers, kwargs, 5).items()})


def tail(obj: Any, indexers: Any = None, **kwargs: int) -> Any:
    """The last ``n`` positions along each dimension (5 by default)."""
    sel = {}
    for d, n in _indexers(obj, indexers, kwargs, 5).items():
        sel[d] = slice(-n, None) if n else slice(0, 0)
    return obj.isel(sel)


def thin(obj: Any, indexers: Any = None, **kwargs: int) -> Any:
    """Every ``n``-th position along each dimension."""
    sel = {}
    for d, n in _indexers(obj, indexers, kwargs, 5).items():
        if n <= 0:
            raise ValueError(f"thin needs a positive step, got {n} for {d!r}")
        sel[d] = slice(None, None, n)
    return obj.isel(sel)


def drop_isel(obj: Any, indexers: Any = None, **kwargs: Any) -> Any:
    """Drop positions along dimensions: ``da.drop_isel(x=[0, 2])``."""
    sel = {}
    for d, drop in {**(indexers or {}), **kwargs}.items():
        n = obj.sizes[d]
        gone = {int(i) % n for i in np.atleast_1d(drop)}
        sel[d] = [i for i in range(n) if i not in gone]
    return obj.isel(sel)


def _put_at(
    da: DataArray, indexers: Mapping[Any, Any], pick: Callable[[DataArray, Any], Any]
) -> list[tuple[str, Any]]:
    """Each indexer resolved to positions by selecting from an array of positions labelled
    like ``da`` along that dimension: keys mean exactly what they mean to ``isel``/``sel``."""
    at: list[tuple[str, Any]] = []
    for d, key in indexers.items():
        if d not in da.dims:
            raise ValueError(f"{d!r} is not a dimension of the array (dimensions {da.dims})")
        coords = {d: da.coords[d]} if d in da.coords and da.coords[d].dims == (d,) else {}
        positions = DataArray(np.arange(da.sizes[d]), dims=(d,), coords=coords)
        got = pick(positions, {d: key})
        if got.dims not in ((), (d,)):
            raise ValueError(f"the indexer for {d!r} must select along {d!r} only")
        values = np.asarray(got.values)
        at.append((str(d), int(values) if values.ndim == 0 else [int(i) for i in values]))
    return at


def da_put(
    da: DataArray, value: Any, indexers: Mapping[Any, Any] | None = None, **indexers_kwargs: Any
) -> DataArray:
    """A copy with the values at positions (``isel`` keys) replaced by ``value``; the
    original is unchanged. See ``DataArray.put``."""
    ix = {**(indexers or {}), **indexers_kwargs}
    return da._put(_put_at(da, ix, lambda p, k: p.isel(k)), value)


def da_put_sel(
    da: DataArray,
    value: Any,
    indexers: Mapping[Any, Any] | None = None,
    method: str | None = None,
    tolerance: Any = None,
    **indexers_kwargs: Any,
) -> DataArray:
    """A copy with the values at labels (``sel`` keys, ``method``, ``tolerance``) replaced
    by ``value``; the original is unchanged. See ``DataArray.put_sel``."""
    ix = {**(indexers or {}), **indexers_kwargs}
    opts: dict[str, Any] = {}
    if method is not None:
        opts["method"] = method
    if tolerance is not None:
        opts["tolerance"] = tolerance
    return da._put(_put_at(da, ix, lambda p, k: p.sel(k, **opts)), value)


def drop_dims(ds: Dataset, drop_dims: Any, *, errors: str = "raise") -> Dataset:
    """The dataset without every variable that has one of ``drop_dims``."""
    dims = [drop_dims] if isinstance(drop_dims, str) else list(drop_dims)
    missing = [d for d in dims if d not in ds.dims]
    if missing and errors == "raise":
        raise ValueError(f"dimensions {missing} not found in the dataset")
    gone = [
        k for k, v in [*ds.data_vars.items(), *ds.coords.items()] if any(d in v.dims for d in dims)
    ]
    return ds.drop_vars(gone)


def _coord_names(obj: Any, names: Any) -> list[str]:
    if names is None:
        return [k for k, c in obj.coords.items() if c.dims != (k,)]
    names = [names] if isinstance(names, str) else list(names)
    for n in names:
        if n not in obj.coords:
            raise ValueError(f"{n!r} is not a coordinate")
        if obj.coords[n].dims == (n,):
            raise ValueError(f"cannot reset the index coordinate {n!r}")
    return names


def reset_coords(obj: Any, names: Any = None, drop: bool = False) -> Any:
    """Non-index coordinates turned into data variables (or dropped with ``drop=True``); a
    DataArray becomes a Dataset unless ``drop=True``."""
    names = _coord_names(obj, names)
    if drop:
        return obj.drop_vars(names)
    if isinstance(obj, DataArray):
        if obj.name is None:
            raise ValueError("cannot reset the coordinates of an unnamed DataArray")
        ds = obj.drop_vars(names).to_dataset()
        for n in names:
            ds[n] = obj.coords[n]
        return ds
    out = obj.copy()
    out._reset_coords(names)
    return out


def filter_by_attrs(ds: Dataset, **kwargs: Any) -> Dataset:
    """The data variables whose attributes match every keyword: equal to a value, or a
    callable returning true for the attribute (``None`` if absent)."""
    keep = []
    for name, v in ds.data_vars.items():
        attrs = dict(v.attrs)
        if v.units is not None:
            attrs.setdefault("units", v.units)
        ok = True
        for key, pattern in kwargs.items():
            value = attrs.get(key)
            if callable(pattern):
                ok = ok and bool(pattern(value))
            else:
                ok = ok and key in attrs and value == pattern
        if ok:
            keep.append(name)
    return ds[keep]


def to_dataarray(ds: Dataset, dim: str = "variable", name: Hashable | None = None) -> DataArray:
    """The data variables stacked along a new first dimension ``dim`` (labelled by the
    variable names), broadcast against each other; they must share a unit kind."""
    from annoray import broadcast, concat

    arrays = broadcast(*ds.data_vars.values())
    stacked = concat([a.expand_dims(dim) for a in arrays], dim)
    out = stacked.assign_coords({dim: (dim, list(ds.data_vars))})
    out = out.rename(None) if name is None else out.rename(str(name))
    return out.assign_attrs(dict(ds.attrs)) if ds.attrs else out


def da_to_dataset(
    da: DataArray, dim: str | None = None, *, name: str | None = None, promote_attrs: bool = False
) -> Dataset:
    """A dataset of one variable (``name``, or the array's), or of one variable per label of
    ``dim`` (split along it; variables are named by the labels as strings, where xarray
    keeps them as they are, e.g. the float ``10.0``)."""
    if dim is not None:
        if name is not None:
            raise TypeError("give dim or name, not both")
        labels = da.coords[dim].values if dim in da.coords else range(da.sizes[dim])
        out = Dataset(
            {
                _key(label): da.isel({dim: i}).drop_vars(dim, errors="ignore")
                for i, label in enumerate(labels)
            }
        )
    else:
        out = da._to_dataset_named(name)
    if promote_attrs:
        out = out.assign_attrs(dict(da.attrs))
    return out


def _key(label: Any) -> str:
    item = getattr(label, "item", None)
    return str(item() if callable(item) else label)


def _index_coords(obj: Any, other: Any) -> dict[str, Any]:
    return {
        d: other.coords[d]
        for d in other.dims
        if d in obj.dims and d in other.coords and other.coords[d].dims == (d,)
    }


def interp_like(obj: Any, other: Any, method: str = "linear", **kwargs: Any) -> Any:
    """``obj`` interpolated onto ``other``'s dimension coordinates (for shared dimensions)."""
    return obj.interp(_index_coords(obj, other), method=method, **kwargs)


def reindex_like(obj: Any, other: Any, **kwargs: Any) -> Any:
    """``obj`` reindexed onto ``other``'s dimension coordinates (for shared dimensions);
    positions with no label are missing."""
    return obj.reindex(_index_coords(obj, other), **kwargs)


def combine_first(obj: Any, other: Any) -> Any:
    """``obj``'s values, with gaps (and labels it lacks) filled from ``other`` (outer
    join; ``other`` converted into ``obj``'s unit)."""
    from annoray import align, where

    if isinstance(obj, Dataset):
        names = list(dict.fromkeys([*obj.data_vars, *other.data_vars]))
        a, b = align(obj, other, join="outer")
        out = Dataset(coords={k: v for k, v in a.coords.items()})
        for n in names:
            if n in a.data_vars and n in b.data_vars:
                out[n] = where(a[n].notnull(), a[n], b[n])
            else:
                out[n] = a[n] if n in a.data_vars else b[n]
        return out
    a, b = align(obj, other, join="outer")
    return where(a.notnull(), a, b)


def fillna(obj: Any, value: Any) -> Any:
    """Missing values replaced by ``value``: a number (in the data's unit; beside data with a
    unit, only 0), a quantity (converted), a DataArray (aligned, left join), or for a
    Dataset a Dataset or a mapping of variable names to values."""
    from annoray import align, where

    if isinstance(obj, Dataset):
        if isinstance(value, (Dataset, Mapping)):
            out = obj.copy()
            for k in obj.data_vars:
                if k in value:
                    out[k] = fillna(obj[k], value[k])
            return out
        return obj.map(lambda v: fillna(v, value))
    if isinstance(value, DataArray):
        a, b = align(obj, value, join="left")
        return where(a.notnull(), a, b)
    if isinstance(value, Quantity) or hasattr(value, "magnitude"):
        if obj.units is None:
            return obj._fillna_number(float(value))
        return obj._fillna_number(
            float(Quantity(value.magnitude, str(value.units)).to(obj.units).magnitude)
        )
    number = float(value)
    if obj.units is not None and number != 0 and not np.isnan(number):
        raise UnitError(f"a number without a unit cannot fill data in {obj.units}; give a Quantity")
    return obj._fillna_number(number)


_COORD_MODE = {"edge": "edge", "reflect": "reflect", "symmetric": "symmetric", "wrap": "wrap"}


def _widths(obj: Any, pad_width: Any, kwargs: dict[str, Any]) -> dict[str, tuple[int, int]]:
    out = {}
    for d, w in {**(pad_width or {}), **kwargs}.items():
        if d not in obj.dims:
            raise ValueError(f"cannot pad along {d!r}: not a dimension")
        out[d] = (w, w) if isinstance(w, int) else (int(w[0]), int(w[1]))
    return out


def _pad_values(values: Any, axes: list[tuple[int, int]], mode: str, fill: Any) -> Any:
    values = np.asarray(values)
    if not any(a or b for a, b in axes):
        return values  # nothing to pad (0-d, or no padded dimension)
    if mode == "constant":
        if fill is None:
            fill = np.nan
            if values.dtype.kind in "iub":
                values = values.astype(float)
            elif values.dtype.kind in "Mm":
                # NaT in the values' own unit (a unitless NaT is deprecated in numpy).
                fill = values.dtype.type("NaT", np.datetime_data(values.dtype)[0])
            elif values.dtype.kind in "OU":
                # Text has no missing value: pad with "" (xarray pads with NaN objects).
                fill = ""
        return np.pad(values, axes, mode="constant", constant_values=fill)
    return np.pad(values, axes, mode=cast(Any, mode))


def pad(
    obj: Any,
    pad_width: Mapping[str, Any] | None = None,
    mode: str = "constant",
    constant_values: Any = None,
    keep_attrs: bool | None = None,
    **pad_width_kwargs: Any,
) -> Any:
    """Pad along dimensions: ``da.pad(y=(1, 2))``. ``mode`` is numpy's (``"constant"``,
    ``"edge"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``); constant padding is missing
    values (integers become floats) unless ``constant_values`` is given (a number in the
    data's unit, or a quantity). Coordinates are padded alike (with missing values for
    ``"constant"``; text with ``""``, where xarray uses NaN objects). Data is read."""
    if isinstance(obj, Dataset):
        widths = _widths(obj, pad_width, pad_width_kwargs)
        out = Dataset(attrs=dict(obj.attrs) if keep_attrs is not False else None)
        for k, c in obj.coords.items():
            out.coords[k] = _pad_array(c, widths, mode, None)
        for k, v in obj.data_vars.items():
            out[k] = _pad_array(v, widths, mode, constant_values, keep_attrs, coords=False)
        return out
    return _pad_array(
        obj, _widths(obj, pad_width, pad_width_kwargs), mode, constant_values, keep_attrs
    )


def _pad_array(
    da: DataArray,
    widths: dict[str, tuple[int, int]],
    mode: str,
    constant_values: Any,
    keep_attrs: bool | None = None,
    coords: bool = True,
) -> DataArray:
    if mode not in ("constant", *_COORD_MODE):
        raise ValueError(f"unsupported pad mode {mode!r}")
    fill = constant_values
    if fill is not None:
        if isinstance(fill, Quantity) or hasattr(fill, "magnitude"):
            q = Quantity(fill.magnitude, str(fill.units))
            fill = q.to(da.units).magnitude if da.units else float(q)
        elif da.units is not None and fill != 0 and not np.isnan(fill):
            raise UnitError(
                f"a number without a unit cannot pad data in {da.units}; give a Quantity"
            )
    axes = [widths.get(d, (0, 0)) for d in da.dims]
    values = _pad_values(da.values, axes, mode, fill)
    new_coords = {}
    if coords:
        for k, c in da.coords.items():
            caxes = [widths.get(d, (0, 0)) for d in c.dims]
            new_coords[k] = DataArray(
                _pad_values(c.values, caxes, mode, None),
                dims=c.dims,
                attrs=dict(c.attrs),
                units=c.units,
            )
    return DataArray(
        values,
        dims=da.dims,
        coords=new_coords,
        name=da.name,
        attrs=dict(da.attrs) if keep_attrs is not False else None,
        units=da.units,
    )


def _rank_1d(x: np.ndarray, pct: bool) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    ok = ~np.isnan(x)
    v = x[ok]
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    sorted_v = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    out[ok] = ranks / len(v) if pct and len(v) else ranks
    return out


def rank(obj: Any, dim: str, *, pct: bool = False, keep_attrs: bool | None = None) -> Any:
    """Ranks along ``dim`` (1 for the smallest; ties share the average rank; missing values
    stay missing); ``pct=True`` divides by the number of valid values. Unitless."""
    if isinstance(obj, Dataset):
        return obj.map(
            lambda v: rank(v, dim, pct=pct, keep_attrs=keep_attrs) if dim in v.dims else v
        )
    axis = obj.dims.index(dim)
    values = np.asarray(obj.values, dtype=float)
    ranked = np.apply_along_axis(_rank_1d, axis, values, pct)
    return obj.copy(data=ranked).pipe(_strip_unit, keep_attrs)


def _strip_unit(da: DataArray, keep_attrs: bool | None) -> DataArray:
    out = DataArray(
        da.values,
        dims=da.dims,
        coords=dict(da.coords),
        name=da.name,
        attrs=dict(da.attrs) if keep_attrs is not False else None,
    )
    return out


def cumulative(obj: Any, dim: Any, min_periods: int = 1) -> Any:
    """Cumulative reductions along ``dim``: ``da.cumulative("t").mean()`` (a rolling window
    as long as the dimension, as in xarray)."""
    dims = [dim] if isinstance(dim, str) else list(dim)
    if len(dims) != 1:
        raise NotImplementedError("cumulative along several dimensions")
    return obj.rolling({dims[0]: obj.sizes[dims[0]]}, min_periods=min_periods)


def reduce(
    obj: Any,
    func: Callable[..., Any],
    dim: Any = None,
    *,
    axis: Any = None,
    keep_attrs: bool | None = None,
    keepdims: bool = False,
    output_units: Any = None,
    **kwargs: Any,
) -> Any:
    """``func(values, axis=…, **kwargs)`` over ``dim`` (all dimensions by default), through
    ``apply_ufunc``: the unit is kept unless ``output_units`` gives another (``func`` sees
    plain numbers)."""
    from annoray import apply_ufunc

    if isinstance(obj, Dataset):
        wanted = None if dim is None else ([dim] if isinstance(dim, str) else list(dim))

        def one(v: DataArray) -> DataArray:
            if wanted is not None and not all(d in v.dims for d in wanted):
                return v
            return reduce(
                v,
                func,
                dim,
                keep_attrs=keep_attrs,
                keepdims=keepdims,
                output_units=output_units,
                **kwargs,
            )

        return obj.map(one, keep_attrs=keep_attrs)
    if dim is not None and axis is not None:
        raise ValueError("give dim or axis, not both")
    if axis is not None:
        axes = [axis] if isinstance(axis, int) else list(axis)
        dims = [obj.dims[a] for a in axes]
    elif dim is None or dim is ...:
        dims = list(obj.dims)
    else:
        dims = [dim] if isinstance(dim, str) else list(dim)
    if keepdims:
        raise NotImplementedError("keepdims=True")
    n = len(dims)
    return apply_ufunc(
        func,
        obj,
        input_core_dims=[dims],
        kwargs={"axis": tuple(range(-n, 0)), **kwargs},
        keep_attrs=True if keep_attrs is None else keep_attrs,
        output_units=[output_units if output_units is not None else obj.units],
    )


def nbytes(obj: Any) -> int:
    """Bytes of the data (all variables, for a Dataset), without reading lazy data. Text
    counts as numpy object arrays (8 bytes per element)."""
    if isinstance(obj, Dataset):
        return sum(nbytes(v) for v in [*obj.data_vars.values(), *obj.coords.values()])
    return int(obj.size * obj.dtype.itemsize)


def get_index(obj: Any, key: str) -> Any:
    """The pandas index of dimension ``key`` (a RangeIndex without a coordinate)."""
    import pandas as pd

    if key not in obj.dims:
        raise KeyError(key)
    c = obj.coords.get(key)
    if c is not None and c.dims == (key,):
        return pd.Index(np.asarray(c.values), name=key)
    return pd.RangeIndex(obj.sizes[key], name=key)


def indexes(obj: Any) -> dict[str, Any]:
    """Dimension name → pandas index, for dimensions with a coordinate."""
    return {
        d: get_index(obj, d) for d in obj.dims if d in obj.coords and obj.coords[d].dims == (d,)
    }


def chunk(obj: Any, *args: Any, **kwargs: Any) -> Any:
    """No-op: annoray reads files lazily without dask; returns ``obj``."""
    return obj


def persist(obj: Any, **kwargs: Any) -> Any:
    """No-op without dask; returns ``obj``."""
    return obj


_UNNAMED = "__xarray_dataarray_variable__"


def da_to_netcdf(da: DataArray, *args: Any, **kwargs: Any) -> None:
    """Write as a one-variable netCDF file (an unnamed array under xarray's placeholder
    name, which ``open_dataarray`` reads back as unnamed)."""
    da.to_dataset(name=da.name if da.name is not None else _UNNAMED).to_netcdf(*args, **kwargs)


def da_to_zarr(da: DataArray, *args: Any, **kwargs: Any) -> None:
    """Write as a one-variable Zarr store (see ``to_netcdf``)."""
    da.to_dataset(name=da.name if da.name is not None else _UNNAMED).to_zarr(*args, **kwargs)


def map_blocks(
    obj: Any, func: Callable[..., Any], args: Any = (), kwargs: Any = None, template: Any = None
) -> Any:
    """``func(obj, *args, **kwargs)``: a single block without dask, as xarray for unchunked
    data."""
    return func(obj, *args, **(kwargs or {}))


def argsort(obj: DataArray, axis: int = -1, kind: Any = None, order: Any = None) -> DataArray:
    """Positions that sort the values along numpy axis ``axis`` (NaN last), as numpy's
    ``argsort``; dimensions, coordinates, name and attributes kept, no unit. Data is
    read."""
    if order is not None:
        raise ValueError("argsort: order= applies to structured arrays, which annoray lacks")
    idx = np.argsort(np.asarray(obj.values), axis=axis, kind=kind)
    return DataArray(
        idx, dims=obj.dims, coords=dict(obj.coords), name=obj.name, attrs=dict(obj.attrs)
    )


def reduce_pickle(obj: Any) -> Any:
    """``__reduce__``: rebuilt from a plain-data state (lazy data is read)."""
    from annoray import _state

    if isinstance(obj, Dataset):
        return (_state.ds_from_state, (_state.ds_state(obj),))
    return (_state.da_from_state, (_state.da_state(obj),))


def deepcopy(obj: Any, memo: Any = None) -> Any:
    """``copy.deepcopy``: a copy (data is never shared between annoray objects)."""
    return obj.copy()


def shallow_copy(obj: Any) -> Any:
    """``copy.copy``."""
    return obj.copy()


def close(obj: Any) -> None:
    """No-op: a file is released when the data read from it is dropped."""


def enter(obj: Any) -> Any:
    return obj


def exit_(obj: Any, *exc: Any) -> None:
    return None


def variables(obj: Dataset) -> dict[str, DataArray]:
    """Every variable (data variables and coordinates), by name."""
    return {**dict(obj.coords), **dict(obj.data_vars)}


def real(obj: Any) -> Any:
    """The real part (annoray holds real numbers: the data itself)."""
    return obj.copy()


def imag(obj: Any) -> Any:
    """The imaginary part (annoray holds real numbers: zeros, in the data's unit)."""
    from annoray import zeros_like

    return zeros_like(obj)


def conj(obj: Any) -> Any:
    """The complex conjugate (of real numbers: the data itself)."""
    return obj.copy()


def repr_html(obj: Any) -> str:
    """The text repr, escaped, for notebooks."""
    import html

    return f"<pre>{html.escape(repr(obj))}</pre>"


def _cdl_value(v: Any) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    a = np.atleast_1d(np.asarray(v))
    return ", ".join(str(x) for x in a.tolist())


def info(ds: Dataset, buf: Any = None) -> None:
    """A netCDF CDL-like summary (``ncdump -h``), as xarray's ``Dataset.info``."""
    import sys

    lines = ["annoray.Dataset {", "dimensions:"]
    lines += [f"\t{d} = {n} ;" for d, n in ds.sizes.items()]
    lines.append("\nvariables:")
    for name, v in variables(ds).items():
        lines.append(f"\t{v.dtype} {name}({', '.join(v.dims)}) ;")
        attrs = dict(v.attrs)
        if v.units is not None:
            attrs = {"units": v.units, **attrs}
        lines += [f"\t\t{name}:{k} = {_cdl_value(a)} ;" for k, a in attrs.items()]
    lines.append("\n// global attributes:")
    lines += [f"\t:{k} = {_cdl_value(a)} ;" for k, a in ds.attrs.items()]
    lines.append("}")
    (buf or sys.stdout).write("\n".join(lines) + "\n")


def groupby_reduce(gb: Any, func: Callable[..., Any], dim: Any = None, **kwargs: Any) -> Any:
    """``func(values, axis=…)`` over each group (over the grouped dimension by default)."""
    d = gb._dim if dim is None else dim
    return gb.map(lambda g: g.reduce(func, d, **kwargs))


def groupby_quantile(gb: Any, q: Any, dim: Any = None, **kwargs: Any) -> Any:
    """Quantiles of each group (over the grouped dimension by default)."""
    d = gb._dim if dim is None else dim
    return gb.map(lambda g: g.quantile(q, dim=d, **kwargs))


def drop_duplicates(obj: Any, dim: Any, *, keep: Any = "first") -> Any:
    """Positions whose label along ``dim`` (a name, names, or ``...`` for all) repeats an
    earlier one dropped (``keep="last"``: a later one; ``keep=False``: every repeated
    label)."""
    dims = list(obj.dims) if dim is ... else ([dim] if isinstance(dim, str) else list(dim))
    sel = {}
    for d in dims:
        if d not in obj.coords:
            raise ValueError(f"drop_duplicates: {d!r} has no coordinate")
        labels = np.asarray(obj.coords[d].values)
        if keep not in ("first", "last", False):
            raise ValueError("keep must be 'first', 'last' or False")
        seen: dict[Any, list[int]] = {}
        for i, v in enumerate(labels.tolist()):
            seen.setdefault(v, []).append(i)
        if keep == "first":
            kept = sorted(p[0] for p in seen.values())
        elif keep == "last":
            kept = sorted(p[-1] for p in seen.values())
        else:
            kept = sorted(p[0] for p in seen.values() if len(p) == 1)
        sel[d] = kept
    return obj.isel(sel)


def reindex_method(
    obj: Any,
    indexers: Any = None,
    method: str | None = None,
    tolerance: Any = None,
    fill_value: Any = None,
    **kwargs: Any,
) -> Any:
    """``reindex`` with ``method`` (``"nearest"``, ``"pad"``/``"ffill"``,
    ``"backfill"``/``"bfill"``), ``tolerance`` or ``fill_value``: positions from pandas'
    ``Index.get_indexer`` (as xarray); labels with no match are missing (``fill_value``
    where given; text coordinates get ``""``)."""
    import pandas as pd

    from annoray import DataArray

    out = obj
    for dim, labels in {**(indexers or {}), **kwargs}.items():
        target = np.asarray(labels.values if isinstance(labels, DataArray) else labels)
        source = pd.Index(np.asarray(out.coords[dim].values))
        pos = source.get_indexer(pd.Index(target), method=cast(Any, method), tolerance=tolerance)
        found = pos >= 0
        taken = out.isel({dim: np.where(found, pos, 0)})
        mask = DataArray(found, dims=(dim,))
        coords = {}
        for k, c in taken.coords.items():
            if dim in c.dims and k != dim:
                v = np.asarray(c.values)
                if v.dtype.kind in "OU":
                    coords[k] = (c.dims, np.where(np.asarray(mask.broadcast_like(c).values), v, ""))
                else:
                    coords[k] = c.where(mask.broadcast_like(c))
        coords[dim] = (dim, target)
        taken = taken.assign_coords(coords)
        mask = mask.assign_coords({dim: (dim, target)})
        out = taken.where(mask) if fill_value is None else taken.where(mask, fill_value)
    return out


_ONE_INDEX = (
    "annoray indexes a dimension by the coordinate of the same name; MultiIndexes are not "
    "supported (stack() combines dimensions)"
)


def set_index(obj: Any, indexes: Any = None, append: bool = False, **indexes_kwargs: Any) -> Any:
    """Make a coordinate the index of a dimension: ``ds.set_index(x="lab")`` (as xarray: the
    old ``x`` coordinate is dropped and ``lab``'s values become the coordinate ``x``). One
    coordinate per dimension; MultiIndexes are refused."""
    if append:
        raise NotImplementedError(f"set_index(append=True) builds a MultiIndex: {_ONE_INDEX}")
    out = obj
    for dim, var in {**(indexes or {}), **indexes_kwargs}.items():
        if isinstance(var, (list, tuple)):
            if len(var) != 1:
                raise NotImplementedError(f"set_index with several coordinates: {_ONE_INDEX}")
            var = var[0]
        if dim not in out.dims:
            raise ValueError(f"set_index: {dim!r} is not a dimension")
        if var == dim:
            continue
        if var not in out.coords or out.coords[var].dims != (dim,):
            raise ValueError(f"set_index: {var!r} is not a coordinate along {dim!r} alone")
        if dim in out.coords:
            out = out.drop_vars(dim)
        out = out.rename({var: dim}) if isinstance(out, DataArray) else out.rename_vars({var: dim})
    return out


def reset_index(obj: Any, dims_or_levels: Any, *, drop: bool = False) -> Any:
    """Remove dimension indexes: with ``drop=True``, their coordinates go too (the
    dimensions stay, unlabelled). Keeping such a coordinate without its index is not
    representable in annoray (a coordinate named after its dimension is its index); rename
    it first (``rename_vars``) to keep its values as a plain coordinate."""
    names = [dims_or_levels] if isinstance(dims_or_levels, str) else list(dims_or_levels)
    for n in names:
        if n not in obj.coords or obj.coords[n].dims != (n,):
            raise ValueError(f"reset_index: {n!r} is not an index (a dimension coordinate)")
    if not drop:
        raise NotImplementedError(
            "reset_index without drop=True keeps the coordinate without its index, which "
            f"annoray cannot represent ({_ONE_INDEX}); use drop=True, or rename_vars to keep "
            "the values as a plain coordinate"
        )
    return obj.drop_vars(names)


def set_xindex(obj: Any, coord_names: Any, index_cls: Any = None, **options: Any) -> Any:
    """Not supported: annoray selects by dimension coordinates only. Use
    ``set_index(dim=name)`` or ``swap_dims`` to make the coordinate a dimension's index."""
    raise NotImplementedError(
        "set_xindex: annoray selects by dimension coordinates only; use set_index(dim=coord) "
        "or swap_dims({dim: coord})"
    )


def to_stacked_array(obj: Any, *args: Any, **kwargs: Any) -> Any:
    """Not supported: the result is indexed by a MultiIndex."""
    raise NotImplementedError(f"to_stacked_array: {_ONE_INDEX}")
