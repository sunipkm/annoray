"""Times at the boundary with Python.

Values with a time zone (``whenever`` instants and zoned or offset date-times,
timezone-aware ``datetime.datetime`` and pandas timestamps, and pandas' timezone-aware
arrays) become ``datetime64[ns]`` values in UTC, exactly (whole nanoseconds, no floats), with
the zone they share; numpy is never asked to convert them (it warns, or reads whenever's
zoned date-times as dates). ``whenever.PlainDateTime`` has no zone and is refused: the zone
is the caller's to choose (``.assume_utc()``, ``.assume_tz(...)``).
"""

from __future__ import annotations

import datetime as _dt
import functools
from typing import Any

import numpy as np

_NAT = np.iinfo(np.int64).min
_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)
_MICRO = _dt.timedelta(microseconds=1)


def _offset_name(seconds: int) -> str:
    sign = "-" if seconds < 0 else "+"
    h, rest = divmod(abs(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{sign}{h:02}:{m:02}" + (f":{s:02}" if s else "")


def _seconds(delta: Any) -> int:
    """Whole seconds of a ``datetime.timedelta`` or ``whenever.TimeDelta``."""
    if isinstance(delta, _dt.timedelta):
        return int(delta.total_seconds())
    total = getattr(delta, "total", None)
    return int(total("seconds") if total is not None else delta.in_seconds())


def zone_name(tz: Any) -> str | None:
    """The name annoray gives a zone: ``None``, a name, a ``zoneinfo.ZoneInfo``, a pytz zone, a
    ``datetime.timezone`` (UTC or a fixed offset)."""
    if tz is None or isinstance(tz, str):
        return tz
    for attr in ("key", "zone"):  # zoneinfo, pytz
        key = getattr(tz, attr, None)
        if isinstance(key, str):
            return key
    if isinstance(tz, _dt.timezone):
        offset = tz.utcoffset(None)
        return "UTC" if not offset else _offset_name(_seconds(offset))
    if isinstance(tz, _dt.tzinfo) and tz.utcoffset(None) is not None:
        offset = tz.utcoffset(None)
        return "UTC" if not offset else _offset_name(_seconds(offset))
    raise TypeError(
        f"cannot tell the time zone of {tz!r}: give a name such as 'America/New_York', "
        "'UTC' or '+05:30', or a zoneinfo.ZoneInfo"
    )


def _is_whenever(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "whenever"


@functools.cache
def _whenever_011() -> bool:
    """Whether whenever has its 0.11 spellings (`timestamp(unit=...)`, `tz_id`); 0.10's are
    deprecated there."""
    import whenever

    major, minor = (int(p) for p in whenever.__version__.split(".")[:2])
    return (major, minor) >= (0, 11)


def _nanos(x: Any) -> int:
    """Nanoseconds since the epoch of a whenever instant or exact date-time."""
    return x.timestamp(unit="nanosecond") if _whenever_011() else x.timestamp_nanos()


def _from_nanos(n: int) -> Any:
    import whenever

    # Typed as Any: the installed version's stubs know only one of the two spellings.
    instant: Any = whenever.Instant
    if _whenever_011():
        return instant.from_timestamp(n, unit="nanosecond")
    return instant.from_timestamp_nanos(n)


def _is_missing(x: Any) -> bool:
    """``None``, NaN, numpy's NaT or pandas' ``NaT``."""
    if x is None:
        return True
    if isinstance(x, float | np.floating):
        return bool(np.isnan(x))
    if isinstance(x, np.datetime64 | np.timedelta64):
        return bool(np.isnat(x))
    return type(x).__name__ == "NaTType"


def _one(x: Any) -> tuple[int, str | None, bool] | None:
    """``(ns, zone, has_zone)`` for a time with a zone, or ``None`` for anything else."""
    if _is_whenever(x):
        kind = type(x).__name__
        if kind == "PlainDateTime":
            raise TypeError(
                "a whenever.PlainDateTime has no time zone; choose one first "
                "(.assume_utc(), .assume_tz('Europe/Paris'), .assume_fixed_offset(...))"
            )
        if kind == "Instant":
            return _nanos(x), None, True
        if kind == "ZonedDateTime":
            return _nanos(x), x.tz_id if _whenever_011() else x.tz, True
        if kind == "OffsetDateTime":
            return _nanos(x), _offset_name(_seconds(x.offset)), True
        return None
    if isinstance(x, _dt.datetime) and x.tzinfo is not None and x.utcoffset() is not None:
        zone = zone_name(getattr(x, "tz", None) or x.tzinfo)
        value = getattr(x, "value", None)  # pandas: nanoseconds since the epoch
        if isinstance(value, (int, np.integer)) and type(x).__name__ == "Timestamp":
            return int(value), zone, True
        return (x - _EPOCH) // _MICRO * 1000, zone, True
    return None


def _pandas_aware(obj: Any) -> tuple[np.ndarray, str | None] | None:
    """pandas' timezone-aware arrays: their UTC values without a copy through objects."""
    tz = getattr(getattr(obj, "dtype", None), "tz", None)
    if tz is None or type(obj).__module__.split(".")[0] != "pandas":
        return None
    import pandas as pd

    idx = pd.DatetimeIndex(obj).tz_convert(None)
    as_unit = getattr(idx, "as_unit", None)
    values = np.asarray(as_unit("ns") if as_unit is not None else idx).astype("datetime64[ns]")
    return values, zone_name(tz)


def coerce(obj: Any) -> tuple[Any, str | None]:
    """``(values, zone)``: times with a zone as ``datetime64[ns]`` in UTC and the zone they
    share (``None`` for UTC, ``whenever.Instant``\\ s, or zones that differ); anything else as
    ``numpy.asarray`` gives it, with no zone."""
    if isinstance(obj, np.ndarray) and obj.dtype.kind != "O":
        return obj, None
    aware = _pandas_aware(obj)
    if aware is not None:
        return aware
    single = _one(obj)
    if single is not None:
        ns, zone, _ = single
        return np.array(ns, dtype="int64").view("datetime64[ns]"), zone
    arr = np.asarray(obj)
    if arr.dtype.kind != "O" or arr.size == 0:
        return arr, None
    flat = arr.ravel()
    first = next((x for x in flat if not _is_missing(x)), None)
    if first is None or _one(first) is None:
        return arr, None
    ns = np.empty(flat.size, dtype="int64")
    zones: set[str | None] = set()
    for i, x in enumerate(flat):
        if _is_missing(x):
            ns[i] = _NAT
            continue
        one = _one(x)
        if one is None:
            raise TypeError(
                f"{x!r} has no time zone, but other times here do; give every time a zone (or none)"
            )
        ns[i], zone, _ = one
        zones.add(zone)
    zone = zones.pop() if len(zones) == 1 else None
    return ns.view("datetime64[ns]").reshape(arr.shape), zone


def to_whenever(values: np.ndarray, zone: str | None) -> np.ndarray:
    """``datetime64[ns]`` values (UTC) as whenever objects: ``ZonedDateTime`` in an IANA zone,
    ``OffsetDateTime`` at a fixed offset, ``Instant`` without a zone; ``None`` for NaT."""
    import whenever

    ns = np.asarray(values).astype("datetime64[ns]").view("int64")
    out = np.empty(ns.shape, dtype=object)
    offset = None
    if zone is not None and zone != "UTC" and zone[:1] in "+-":
        sign = -1 if zone[0] == "-" else 1
        parts = [int(p) for p in zone[1:].split(":")]
        parts += [0] * (3 - len(parts))
        offset = whenever.TimeDelta(seconds=sign * (parts[0] * 3600 + parts[1] * 60 + parts[2]))
    for i, n in np.ndenumerate(ns):
        if n == _NAT:
            out[i] = None
            continue
        t = _from_nanos(int(n))
        if offset is not None:
            out[i] = t.to_fixed_offset(offset)
        elif zone is not None and zone != "UTC":
            out[i] = t.to_tz(zone)
        else:
            out[i] = t
    return out
