"""``rarray.to_netcdf``: write an xarray Dataset with xarray-rs's CF encoder and writers."""

from __future__ import annotations

import os
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import numpy as np
import xarray as xr

from rarray._rarray import write as _write

# Encoding keys (netCDF4-python names) xarray-rs understands; the rest are dropped, as
# xarray's own backends drop keys they don't use.
_ENCODING_KEYS = {
    "dtype",
    "_FillValue",
    "missing_value",
    "scale_factor",
    "add_offset",
    "units",
    "calendar",
    "chunksizes",
    "zlib",
    "complevel",
    "shuffle",
    "compression",
}


def _values(var: xr.Variable) -> dict[str, Any]:
    """A variable's (decoded) values in a form the Rust side takes without copying twice."""
    data = np.asarray(var.values)
    kind = data.dtype.kind
    if kind == "M":
        return {"kind": "datetime", "data": data.astype("datetime64[ns]").view(np.int64)}
    if kind == "m":
        return {"kind": "timedelta", "data": data.astype("timedelta64[ns]").view(np.int64)}
    if kind in "biuf":
        return {"kind": "num", "data": np.ascontiguousarray(data)}
    if kind in "US":
        flat = [x.decode() if isinstance(x, bytes) else str(x) for x in data.ravel()]
        return {"kind": "str", "data": flat, "shape": list(data.shape)}
    if kind == "O":
        flat = data.ravel()
        first = next((x for x in flat if x is not None), None)
        if first is not None and hasattr(first, "calendar"):
            return _cftime(flat, data.shape, first.calendar)
        if all(isinstance(x, str | bytes) for x in flat):
            strs = [x.decode() if isinstance(x, bytes) else x for x in flat]
            return {"kind": "str", "data": strs, "shape": list(data.shape)}
    raise TypeError(f"cannot write values of type {data.dtype}")


def _cftime(flat, shape, calendar: str) -> dict[str, Any]:
    parts = [[], [], [], [], [], [], [], []]
    for d in flat:
        missing = d is None or (isinstance(d, float) and np.isnan(d))
        fields = (
            (0, 1, 1, 0, 0, 0, 0)
            if missing
            else (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond)
        )
        for p, f in zip(parts, (*fields, int(missing)), strict=True):
            p.append(int(f))
    return {"kind": "cftime", "data": parts, "shape": list(shape), "calendar": calendar}


def _encoding(enc: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in enc.items():
        if k not in _ENCODING_KEYS:
            continue
        if k == "dtype":
            if v is str:
                v = "str"
            else:
                v = np.dtype(v).name
        elif k == "chunksizes" and v is not None:
            v = [int(c) for c in v]
        elif k == "_FillValue" and v is False:
            continue
        out[k] = v
    return out


def to_netcdf(
    ds: xr.Dataset,
    path: str | os.PathLike,
    *,
    format: str = "NETCDF4",
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
) -> None:
    """Write ``ds`` to ``path`` as netCDF-4 (default) or netCDF-3.

    Values are CF-encoded (times, packing, fill values) by xarray-rs, not by xarray; the
    file is written by pure-Rust writers with the GIL released. Encodings follow xarray's
    (netCDF4-python) names, taken from each variable's ``.encoding`` and ``encoding``; codecs
    xarray-rs can write: zlib (deflate), zstd, lz4 and blosc_{lz4, lz4hc, zlib, zstd}.
    """
    encoding = encoding or {}
    variables = []
    for name, var in ds.variables.items():
        enc = {**var.encoding, **encoding.get(name, {})}
        variables.append(
            {
                "name": str(name),
                "dims": [str(d) for d in var.dims],
                "attrs": dict(var.attrs),
                "encoding": _encoding(enc),
                "coord": name in ds.coords,
                **_values(var),
            }
        )
    if unlimited_dims is None:
        unlimited_dims = ds.encoding.get("unlimited_dims", ())
    _write(
        os.fspath(path),
        format,
        dict(ds.attrs),
        variables,
        [str(d) for d in unlimited_dims],
    )
