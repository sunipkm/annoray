"""Golden file with missing times (NaT) for annoray/tests/nat.rs.

Run from the repository root:  python3 fixtures/gen_nat.py

Writes fixtures/nc/nat.nc with xarray's own encodings of NaT:
- `t`: datetimes, default encoding (int64 days; NaT stored as the int64 minimum);
- `t_ns`: int64 nanoseconds too precise for float64, with a _FillValue (written raw: xarray's
  own encoder writes these values as all fill, losing them);
- `t_float`: datetimes stored as float64 (NaT stored as NaN);
- `t_noleap`: a noleap calendar with a _FillValue (written raw through netCDF4);
- `d`: durations (`dtype = "timedelta64[…]"`), with NaT.
Then prints how xarray decodes each, and the raw stored values.
"""

import os

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

OUT = os.path.join(os.path.dirname(__file__), "nc", "nat.nc")

t = pd.to_datetime(["2020-01-01", None, "2020-01-03"])
d = pd.to_timedelta([90, None, -5], unit="min")
ds = xr.Dataset(
    {
        "t": ("x", t),
        "t_float": ("x", t),
        "d": ("x", d),
    },
    coords={"x": [1, 2, 3]},
)
ds.to_netcdf(
    OUT,
    format="NETCDF4",
    encoding={
        "t_float": {"dtype": "float64", "units": "hours since 2020-01-01"},
    },
)
with netCDF4.Dataset(OUT, "a") as nc:
    v = nc.createVariable("t_noleap", "i4", ("x",), fill_value=-999)
    v.units = "days since 2000-01-01"
    v.calendar = "noleap"
    v[:] = np.ma.masked_array([59, 0, 365], mask=[False, True, False])
    v = nc.createVariable("t_ns", "i8", ("x",), fill_value=-1)
    v.units = "nanoseconds since 1970-01-01"
    v[:] = np.ma.masked_array([1_700_000_000_123_456_789, 0, 1], mask=[False, True, False])

# How xarray decodes them. The masked `t_noleap` raises an OverflowError in xarray (a bug
# annoray does not copy: it decodes to a NaT).
back = xr.open_dataset(OUT)
for name in ["t", "t_ns", "t_float", "t_noleap", "d"]:
    try:
        print(name, [str(x) for x in back[name].values])
    except Exception as e:  # noqa: BLE001
        print(name, "error:", type(e).__name__)
with netCDF4.Dataset(OUT) as nc:
    nc.set_auto_maskandscale(False)
    for v in nc.variables.values():
        print("raw", v.name, v.dtype, v[:].tolist(), {k: str(v.getncattr(k)) for k in v.ncattrs()})
