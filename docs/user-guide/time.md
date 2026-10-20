---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Dates, times and time zones

A date-time in annoray is an instant: a point on the UTC time line, stored as a 64-bit count of
nanoseconds since 1970-01-01T00:00:00Z, which is numpy's `datetime64[ns]`. A variable of
date-times can also carry a time zone, which sets the wall-clock time each instant shows: its
components (`.dt.hour`), its resampling bins, how date strings select it, and how it is
displayed. The zone never changes the instants themselves, so comparisons, differences and
alignment are those of the instants.

```{code-cell} python
import datetime as dt
import zoneinfo

import numpy as np
import whenever
import annoray as ar
```

## Storage

Date-times are `datetime64[ns]` and durations `timedelta64[ns]`: both are 64-bit nanosecond
counts in memory, so `.values` is a read-only view of the variable's data, not a copy. The
range of an instant is that of numpy's nanoseconds, 1677-09-21 to 2262-04-11. Dates outside it,
and dates in the other CF calendars (`noleap`, `360_day`, …), are CF dates (`cftime` objects
in Python). For data read lazily from a file the choice is made when the file is opened, from
the first and last stored values, as xarray makes it; a value in between outside the range is
an error when the data are read.

```{code-cell} python
times = ar.DataArray(np.array(["2024-05-11T07:00", "2024-05-11T08:00"], "M8[ns]"), dims="t")
a, b = times.values, times.values
print(a.dtype, np.shares_memory(a, b), a.flags.writeable)
```

## Times with a zone as input

Inputs that know their zone are converted to UTC instants exactly, in whole nanoseconds,
without numpy's warning (`no explicit representation of timezones available for
np.datetime64`), and the zone they share becomes the variable's:

| Input | Zone |
|---|---|
| `whenever.ZonedDateTime` | its IANA zone |
| `whenever.OffsetDateTime` | its offset, such as `-04:00` |
| `whenever.Instant` | none (UTC) |
| `datetime.datetime` with a `zoneinfo.ZoneInfo` or pytz zone | the zone's name |
| `datetime.datetime` with a `datetime.timezone` | `UTC`, or the offset |
| pandas `Timestamp`, `DatetimeIndex` or `Series` with a zone | the zone's name |

```{code-cell} python
zoned = [whenever.ZonedDateTime(2024, 5, 11, h, tz="America/New_York") for h in (3, 4)]
da = ar.DataArray(zoned, dims="t")
print(da.values, da.dt.tz)

aware = dt.datetime(2024, 5, 11, 3, tzinfo=zoneinfo.ZoneInfo("Europe/Paris"))
print(ar.DataArray([aware], dims="t").dt.tz)
```

Times in different zones become UTC instants with no zone. A `whenever.PlainDateTime` is
refused, since it names no instant until a zone is chosen for it (`.assume_utc()`,
`.assume_tz(...)`); so is a mix of times with and without a zone. Times without a zone
(`datetime64`, naive `datetime.datetime`) are read as UTC, as in xarray.

```{code-cell} python
try:
    ar.DataArray([whenever.PlainDateTime(2024, 5, 11, 3)], dims="t")
except TypeError as e:
    print(e)
```

## The zone of a variable

`.dt.tz` is the zone's name: `UTC`, an offset such as `+05:30`, or an IANA name such as
`America/New_York` (from the system's time zone database, or the copy built into annoray
where the system has none). `None` means UTC.

`.dt.tz_convert(tz)` keeps the instants and changes the zone: only the wall-clock times
change.

```{code-cell} python
utc = ar.DataArray(np.array(["2024-07-01T02:00"], "M8[ns]"), dims="t")
ny = utc.dt.tz_convert("America/New_York")
print(ny.dt.hour.values, ny.dt.day.values, (ny.values == utc.values).all())
print(ny)
```

`.dt.tz_localize(tz)` keeps the wall-clock times and changes the instants: times stored as
UTC that were in fact local times become the instants those clocks showed. A wall-clock time
that the zone skips (clocks going forward) or repeats (clocks going back) is resolved by
`disambiguate`, with whenever's names: `"compatible"` (the default: the earlier of two
repeated times, a skipped time moved forward by the gap), `"earlier"`, `"later"` or
`"raise"`.

```{code-cell} python
local = ar.DataArray(np.array(["2024-11-03T01:30"], "M8[ns]"), dims="t")
for how in ("earlier", "later"):
    print(how, local.dt.tz_localize("America/New_York", disambiguate=how).values)
```

What the zone changes:

- **Components** (`.dt.hour`, `.dt.dayofweek`, `groupby("time.day")`, …) are those of the
  wall-clock times.
- **Resampling** bins fall on wall-clock times: daily bins start at local midnight, so a day
  across a daylight-saving change lasts 23 or 25 hours. The bins' labels keep the zone.
- **Date strings** in `sel` are wall-clock times: `sel(time="2024-07-01")` selects that local
  day. Labels with a zone (whenever's, aware `datetime`s) are instants.
- **Display** shows wall-clock times with their offset.
- **Results** computed from date-times keep their zone (selection, reductions, a date-time
  plus a duration); a difference of date-times is a duration, without one.

```{code-cell} python
hours = np.datetime64("2024-07-01T04:00", "ns") + np.arange(48) * np.timedelta64(1, "h")
series = ar.DataArray(np.arange(48.0), dims="time", coords={"time": hours})
local = series.assign_coords(time=series.time.dt.tz_convert("America/New_York"))
print(local.resample(time="D").mean().values)
print(series.resample(time="D").mean().values)
print(local.sel(time="2024-07-02").sizes)
```

Date-times in different zones are not combined: `concat`, `merge` and alignment of
coordinates whose zones differ raise an error asking for one zone (`tz_convert`), as they do
for different units.

## Output

`.values` stays `datetime64[ns]` in UTC, a view of the data. `.dt.to_whenever()` gives
whenever objects: `ZonedDateTime` in an IANA zone, `OffsetDateTime` at a fixed offset,
`Instant` without a zone, and `None` for NaT.

```{code-cell} python
print(ny.dt.to_whenever())
print(utc.dt.to_whenever())
```

`date_range(..., tz=...)` returns pandas' index with its zone (as xarray does); a variable
made from it takes the zone.

```{code-cell} python
days = ar.date_range("2024-03-09", periods=3, freq="D", tz="America/New_York")
print(ar.DataArray(days, dims="time").values)
```

## Files

The zone is written as a `time_zone` attribute of the variable (CF has no attribute for it;
other readers see correct UTC instants) and becomes the variable's zone again when the file is
opened. `to_xarray` writes the same attribute, and `from_xarray` (or a `time_zone` entry in
`attrs`) reads it back.

```{code-cell} python
import tempfile
from pathlib import Path

path = Path(tempfile.mkdtemp()) / "zoned.nc"
ar.Dataset({"v": ("time", [1.0, 2.0])}, coords={"time": zoned}).to_netcdf(path)
back = ar.open_dataset(path)
print(back.time.dt.tz, back.time.attrs)
```
