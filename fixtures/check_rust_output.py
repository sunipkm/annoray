"""Check that Python xarray decodes a file written by annoray like the golden file it came from.

Usage: python3 fixtures/check_rust_output.py <written.nc>   (prints OK or raises)
"""

import os
import sys
import warnings

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")
golden = xr.open_dataset(os.path.join(os.path.dirname(__file__), "nc", "xr_nc4.nc"))
ours = xr.open_dataset(sys.argv[1])
assert sorted(ours.data_vars) == sorted(golden.data_vars), (sorted(ours.data_vars), sorted(golden.data_vars))
assert sorted(ours.coords) == sorted(golden.coords), (sorted(ours.coords), sorted(golden.coords))
for name in golden.variables:
    a, b = golden[name], ours[name]
    assert a.dims == b.dims, (name, a.dims, b.dims)
    if a.dtype.kind in "fc":
        np.testing.assert_allclose(a.values, b.values, equal_nan=True, err_msg=name)
    elif a.dtype.kind == "O" and name == "model_time":
        assert [str(x) for x in a.values] == [str(x) for x in b.values], name
    else:
        assert (a.values == b.values).all(), (name, a.values, b.values)
    assert a.attrs.get("units") == b.attrs.get("units"), (name, a.attrs, b.attrs)
assert golden.attrs["title"] == ours.attrs["title"]
print("OK")
