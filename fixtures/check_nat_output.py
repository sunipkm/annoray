"""Check that Python xarray decodes NaT in a file written by annoray like fixtures/nc/nat.nc.

Usage: python3 fixtures/check_nat_output.py <written.nc>   (prints OK or raises)

`t_noleap` is skipped: xarray cannot decode a masked non-standard calendar (an OverflowError
in cftime), in the golden file or ours.
"""

import os
import sys
import warnings

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")
golden = xr.open_dataset(os.path.join(os.path.dirname(__file__), "nc", "nat.nc"), drop_variables=["t_noleap"])
ours = xr.open_dataset(sys.argv[1], drop_variables=["t_noleap"])
for name in ["t", "t_ns", "t_float", "d"]:
    a, b = golden[name].values, ours[name].values
    assert a.dtype.kind == b.dtype.kind, (name, a.dtype, b.dtype)
    np.testing.assert_array_equal(a, b, err_msg=name)
print("OK")
