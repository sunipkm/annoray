"""``.dt`` methods and timedelta fields beyond the native components: ``strftime``,
``floor``/``ceil``/``round`` (through pandas, for numpy times), and for durations ``days``,
``seconds`` (within the day, 0–86399), ``microseconds``, ``nanoseconds`` (as Python's
``timedelta`` and xarray) and ``total_seconds()``."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from annoray._annoray import DataArray


def _like(da: DataArray, values: Any) -> DataArray:
    return DataArray(np.asarray(values), dims=da.dims, coords=dict(da.coords), name=da.name)


def _times(da: DataArray) -> np.ndarray:
    v = np.asarray(da.values)
    if v.dtype.kind != "M":
        raise TypeError(".dt method needs numpy datetimes (CF-calendar dates: use their fields)")
    return v


def strftime(da: DataArray, date_format: str) -> DataArray:
    v = np.asarray(da.values)
    flat = v.ravel()
    if v.dtype.kind == "M":
        import pandas as pd

        out = pd.DatetimeIndex(flat).strftime(date_format).to_numpy(dtype=object)
    else:
        out = np.array([t.strftime(date_format) for t in flat], dtype=object)
    return _like(da, out.reshape(v.shape))


def _rounding(op: str, da: DataArray, freq: str) -> DataArray:
    import pandas as pd

    v = _times(da)
    idx = pd.DatetimeIndex(v.ravel())
    return _like(da, getattr(idx, op)(freq).to_numpy().reshape(v.shape))


_TD_FIELDS = {"days", "seconds", "microseconds", "nanoseconds"}


def _durations(da: DataArray) -> Any:
    import pandas as pd

    v = np.asarray(da.values)
    if v.dtype.kind != "m":
        raise AttributeError("this field needs durations (timedelta data)")
    return v, pd.TimedeltaIndex(v.ravel())


def extra(da: DataArray, name: str) -> Any:
    """The ``.dt`` attribute ``name`` not provided natively, or ``AttributeError``."""
    if name == "strftime":
        return functools.partial(strftime, da)
    if name in ("floor", "ceil", "round"):
        return functools.partial(_rounding, name, da)
    if name in _TD_FIELDS:
        v, idx = _durations(da)
        return _like(da, np.asarray(getattr(idx, name)).reshape(v.shape))
    if name == "total_seconds":
        v, idx = _durations(da)
        return lambda: _like(da, idx.total_seconds().to_numpy().reshape(v.shape))
    raise AttributeError(f"unknown date-time component {name!r}")
