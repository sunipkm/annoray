"""Pickling and copying of DataArray and Dataset: a plain-data state (values, units,
attributes and encoding of every variable) and its inverse. Lazy data is read."""

from __future__ import annotations

from typing import Any

import numpy as np

from annoray._annoray import DataArray, Dataset


def _var(v: DataArray) -> dict[str, Any]:
    return {
        "dims": tuple(v.dims),
        "data": np.asarray(v.values),
        "units": v.units,
        "attrs": dict(v.attrs),
        "encoding": dict(v.encoding),
    }


def _array(s: dict[str, Any], coords: Any = None, name: Any = None) -> DataArray:
    da = DataArray(
        s["data"], dims=s["dims"], coords=coords, name=name, attrs=s["attrs"], units=s["units"]
    )
    if s["encoding"]:
        da.encoding = s["encoding"]
    return da


def da_state(da: DataArray) -> dict[str, Any]:
    return {
        "var": _var(da),
        "name": da.name,
        "coords": {k: _var(c) for k, c in da.coords.items()},
    }


def da_from_state(state: dict[str, Any]) -> DataArray:
    coords = {k: _array(c) for k, c in state["coords"].items()}
    return _array(state["var"], coords=coords, name=state["name"])


def ds_state(ds: Dataset) -> dict[str, Any]:
    return {
        "data_vars": {k: _var(v) for k, v in ds.data_vars.items()},
        "coords": {k: _var(c) for k, c in ds.coords.items()},
        "attrs": dict(ds.attrs),
        "encoding": dict(ds.encoding),
    }


def ds_from_state(state: dict[str, Any]) -> Dataset:
    ds = Dataset(
        {k: _array(v) for k, v in state["data_vars"].items()},
        coords={k: _array(c) for k, c in state["coords"].items()},
        attrs=state["attrs"],
    )
    if state["encoding"]:
        ds.encoding = state["encoding"]
    return ds
