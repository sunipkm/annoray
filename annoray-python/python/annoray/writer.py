"""``annoray.to_netcdf``: write an xarray Dataset with annoray's CF encoder and writers."""

from __future__ import annotations

import os
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import numpy as np
import xarray as xr

from annoray._annoray import _check_encoding
from annoray._annoray import write as _write

# Encoding keys (netCDF4-python names) annoray understands. Keys of a variable's own
# ``.encoding`` outside this set are bookkeeping (``source``, ``original_shape``, …) and are
# dropped; keys of the explicit ``encoding=`` argument are checked, as xarray checks them.
_ENCODING_KEYS = {
    "dtype",
    "_FillValue",
    "missing_value",
    "scale_factor",
    "add_offset",
    "units",
    "calendar",
    "_Unsigned",
    "char_dim_name",
    "chunksizes",
    "contiguous",
    "zlib",
    "complevel",
    "shuffle",
    "fletcher32",
    "compression",
    "blosc_shuffle",
    "szip_coding",
    "szip_pixels_per_block",
    "endian",
    "least_significant_digit",
    "significant_digits",
    "quantize_mode",
    "ragged_counts",
    "bounds",
    "grid_mapping",
    "climatology",
    "geometry",
    "node_coordinates",
    "node_count",
    "part_node_count",
    "interior_ring",
    "cell_measures",
    "formula_terms",
}


def _values(var: xr.Variable) -> dict[str, Any]:
    """A variable's (decoded) values in a form the Rust side takes without copying twice."""
    data = np.asarray(var.values)
    kind = data.dtype.kind
    if kind == "M":
        return {"kind": "datetime", "data": data.astype("datetime64[ns]").view(np.int64)}
    if kind == "m":
        return {"kind": "timedelta", "data": data.astype("timedelta64[ns]").view(np.int64)}
    if kind in "biufc":
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


def _cftime(flat: Iterable[Any], shape: tuple[int, ...], calendar: str) -> dict[str, Any]:
    parts: list[list[int]] = [[], [], [], [], [], [], [], []]
    for d in flat:
        if d is None or (isinstance(d, float) and np.isnan(d)):
            fields = (0, 1, 1, 0, 0, 0, 0, 1)  # missing
        elif hasattr(d, "calendar"):
            c: Any = d
            fields = (c.year, c.month, c.day, c.hour, c.minute, c.second, c.microsecond, 0)
        else:
            raise TypeError(f"expected cftime dates, found {d!r}")
        for p, f in zip(parts, fields, strict=True):
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
                dt = np.dtype(v)
                # An enumeration lives in the dtype's metadata (xarray's NativeEnumCoder).
                v = dt if (dt.metadata or {}).get("enum") else dt.name
        elif k == "chunksizes" and v is not None:
            v = [int(c) for c in v]
        elif k == "_FillValue" and v is False:
            v = None  # netCDF4-python's spelling of "no fill value"
        out[k] = v
    return out


def to_netcdf(
    ds: xr.Dataset,
    path: str | os.PathLike,
    *,
    format: str = "NETCDF4",
    encoding: Mapping[Hashable, Mapping[str, Any]] | None = None,
    unlimited_dims: Iterable[Hashable] | None = None,
    compress: bool | int | Mapping[str, Any] | None = None,
) -> None:
    """Write ``ds`` to ``path`` as netCDF-4 (default) or netCDF-3.

    Values are CF-encoded (times, packing, fill values) by annoray, not by xarray; the
    file is written by pure-Rust writers with the GIL released. Encodings follow xarray's
    (netCDF4-python) names, taken from each variable's ``.encoding`` and ``encoding``; codecs
    annoray can write: zlib (deflate), zstd, lz4 and blosc_{lz4, lz4hc, zlib, zstd}.
    ``compress`` compresses every variable without compression of its own, as for
    ``annoray.Dataset.to_netcdf``.
    """
    encoding = encoding or {}
    for name, given in encoding.items():
        _check_encoding(str(name), dict(given), format)
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
    unlimited: Iterable[Hashable] = (
        unlimited_dims if unlimited_dims is not None else ds.encoding.get("unlimited_dims", ())
    )
    _write(
        os.fspath(path),
        format,
        dict(ds.attrs),
        variables,
        [str(d) for d in unlimited],
        compress,
    )
