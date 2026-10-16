"""pandas and dict conversions (xarray's ``to_pandas``, ``to_series``, ``to_dataframe``,
``from_series``, ``from_dataframe``, ``to_dict``, ``from_dict``).

Units travel with the data: in pandas as ``attrs["units"]`` (a string on a Series, a
column → unit mapping on a DataFrame), in dicts as the ``"units"`` attribute (the
convention of ``to_xarray`` and netCDF), and are read back from there.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from annoray._annoray import DataArray, Dataset


def _pd() -> Any:
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas is needed for this conversion: pip install pandas") from e
    return pd


def _index(obj: Any, dim: str) -> Any:
    pd = _pd()
    coords = obj.coords
    if dim in coords and coords[dim].dims == (dim,):
        return pd.Index(np.asarray(coords[dim].values), name=dim)
    return pd.RangeIndex(obj.sizes[dim], name=dim)


def _full_index(obj: Any, dims: list[str]) -> Any:
    pd = _pd()
    indexes = [_index(obj, d) for d in dims]
    if len(indexes) == 1:
        return indexes[0]
    return pd.MultiIndex.from_product(indexes, names=dims)


def _spread(da: DataArray, dims: list[str], sizes: dict[str, int]) -> np.ndarray:
    """``da``'s values broadcast over ``dims`` (in that order), flattened."""
    own = [d for d in dims if d in da.dims]
    values = np.asarray(da.transpose(*own).values) if own else np.asarray(da.values)
    shape = [sizes[d] if d in da.dims else 1 for d in dims]
    return np.broadcast_to(values.reshape(shape), [sizes[d] for d in dims]).ravel()


def da_to_pandas(da: DataArray) -> Any:
    pd = _pd()
    if da.ndim == 0:
        return np.asarray(da.values)
    if da.ndim == 1:
        out = pd.Series(np.asarray(da.values), index=_index(da, da.dims[0]), name=da.name)
    elif da.ndim == 2:
        out = pd.DataFrame(
            np.asarray(da.values), index=_index(da, da.dims[0]), columns=_index(da, da.dims[1])
        )
    else:
        raise ValueError(
            f"cannot convert arrays with {da.ndim} dimensions into pandas objects; "
            "use to_series or to_dataframe"
        )
    if da.units is not None:
        out.attrs["units"] = da.units
    return out


def da_to_series(da: DataArray, name: Any = None) -> Any:
    pd = _pd()
    dims = list(da.dims)
    index = _full_index(da, dims) if dims else None
    values = np.asarray(da.values).ravel()
    out = pd.Series(values, index=index, name=da.name if name is None else name)
    if da.units is not None:
        out.attrs["units"] = da.units
    return out


def ds_to_dataframe(ds: Dataset, dim_order: Any = None) -> Any:
    pd = _pd()
    dims = list(ds.dims) if dim_order is None else list(dim_order)
    if set(dims) != set(ds.dims):
        raise ValueError(f"dim_order {dims} must name every dimension of {tuple(ds.dims)}")
    sizes = dict(ds.sizes)
    index = _full_index(ds, dims)
    columns: dict[str, Any] = {}
    units: dict[str, str] = {}
    variables = [*ds.data_vars.items(), *ds.coords.items()]
    for name, v in variables:
        if name in dims and v.dims == (name,):
            continue
        columns[name] = _spread(v, dims, sizes)
        if v.units is not None:
            units[name] = v.units
    out = pd.DataFrame(columns, index=index)
    if units:
        out.attrs["units"] = units
    return out


def da_to_dataframe(da: DataArray, name: Any = None, dim_order: Any = None) -> Any:
    name = da.name if name is None else name
    if name is None:
        raise ValueError("cannot convert an unnamed DataArray to a DataFrame: give name=")
    return ds_to_dataframe(da.to_dataset(name=name), dim_order)


def ds_to_pandas(ds: Dataset) -> Any:
    if len(ds.dims) == 0:
        return _pd().Series({k: np.asarray(v.values).item() for k, v in ds.data_vars.items()})
    if len(ds.dims) == 1:
        return ds_to_dataframe(ds)
    raise ValueError(
        f"cannot convert a Dataset with {len(ds.dims)} dimensions into pandas objects "
        "without a MultiIndex; use to_dataframe"
    )


def _dims_and_values(index: Any, values: Any) -> tuple[list[str], dict[str, Any], Any]:
    """The dimensions, their labels and the reshaped values of a (multi-)indexed column."""
    pd = _pd()
    if isinstance(index, pd.MultiIndex):
        index = index.remove_unused_levels()
        dims = [n if n is not None else f"level_{i}" for i, n in enumerate(index.names)]
        full = pd.MultiIndex.from_product(index.levels, names=dims)
        if index.has_duplicates:
            raise ValueError("cannot convert an index with duplicate entries")
        reindexed = pd.Series(values, index=index.set_names(dims)).reindex(full)
        shape = [len(level) for level in index.levels]
        labels = {d: np.asarray(level) for d, level in zip(dims, index.levels)}
        return dims, labels, np.asarray(reindexed.values).reshape(shape)
    dim = index.name if index.name is not None else "index"
    return [dim], {dim: np.asarray(index)}, np.asarray(values)


def da_from_series(series: Any, sparse: bool = False) -> DataArray:
    if sparse:
        raise NotImplementedError("sparse arrays are not supported")
    dims, labels, values = _dims_and_values(series.index, series.values)
    return DataArray(
        values,
        dims=dims,
        coords=labels,
        name=series.name,
        units=series.attrs.get("units"),
    )


def ds_from_dataframe(df: Any, sparse: bool = False) -> Dataset:
    if sparse:
        raise NotImplementedError("sparse arrays are not supported")
    units = df.attrs.get("units", {})
    data_vars = {}
    labels: dict[str, Any] = {}
    for column in df.columns:
        dims, labels, values = _dims_and_values(df.index, df[column].values)
        data_vars[str(column)] = DataArray(
            values, dims=dims, coords=labels, units=units.get(column)
        )
    if not data_vars:
        dims, labels, _ = _dims_and_values(df.index, np.zeros(len(df)))
    return Dataset(data_vars, coords=labels)


def _var_dict(v: DataArray, data: Any) -> dict[str, Any]:
    attrs = dict(v.attrs)
    if v.units is not None:
        attrs["units"] = v.units
    out: dict[str, Any] = {"dims": tuple(v.dims), "attrs": attrs}
    values = np.asarray(v.values)
    if data is False:
        out["dtype"] = str(values.dtype)
        out["shape"] = values.shape
    elif data == "array":
        out["data"] = values
    else:
        out["data"] = values.tolist()
    return out


def da_to_dict(da: DataArray, data: Any = "list", encoding: bool = False) -> dict[str, Any]:
    out = _var_dict(da, data)
    out["coords"] = {k: _var_dict(c, data) for k, c in da.coords.items()}
    out["name"] = da.name
    if encoding:
        out["encoding"] = dict(getattr(da, "encoding", {}))
    return out


def ds_to_dict(ds: Dataset, data: Any = "list", encoding: bool = False) -> dict[str, Any]:
    return {
        "coords": {k: _var_dict(c, data) for k, c in ds.coords.items()},
        "attrs": dict(ds.attrs),
        "dims": dict(ds.sizes),
        "data_vars": {k: _var_dict(v, data) for k, v in ds.data_vars.items()},
    }


def _spec(d: dict[str, Any], what: str) -> tuple[Any, Any, dict[str, Any], Any]:
    if "data" not in d:
        raise ValueError(f"cannot convert dict without the key 'data' ({what})")
    attrs = dict(d.get("attrs", {}))
    units = attrs.pop("units", None)
    return d.get("dims", ()), d["data"], attrs, units


def da_from_dict(d: dict[str, Any]) -> DataArray:
    dims, data, attrs, units = _spec(d, "the array")
    coords = {}
    for k, c in d.get("coords", {}).items():
        cdims, cdata, cattrs, cunits = _spec(c, f"coordinate {k!r}")
        coords[k] = DataArray(cdata, dims=cdims, attrs=cattrs, units=cunits)
    return DataArray(data, dims=dims, coords=coords, name=d.get("name"), attrs=attrs, units=units)


def ds_from_dict(d: dict[str, Any]) -> Dataset:
    def arrays(section: str) -> dict[str, DataArray]:
        out = {}
        for k, v in d.get(section, {}).items():
            dims, data, attrs, units = _spec(v, f"variable {k!r}")
            out[k] = DataArray(data, dims=dims, attrs=attrs, units=units)
        return out

    return Dataset(arrays("data_vars"), coords=arrays("coords"), attrs=d.get("attrs", {}))
