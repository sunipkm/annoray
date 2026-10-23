"""A netCDF-4 file of complex numbers written by netCDF4-python (nc-complex), for the complex
I/O test.

Run from the repository root:  python3 fixtures/gen_complex.py
Writes fixtures/nc/complex.nc. The values are a closed form the Rust test recomputes:
z[y, x] = (y - 1) + 0.25 x i (complex128, unit "V/m", z[1, 2] = NaN). Only complex128:
netCDF4-python 1.7.4 writes a complex64 variable it cannot read back itself.
"""

import os

import netCDF4
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "nc", "complex.nc")

y, x = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
z = (y - 1.0) + 0.25j * x
z[1, 2] = complex(np.nan, np.nan)

with netCDF4.Dataset(OUT, "w", auto_complex=True) as ds:
    ds.createDimension("y", 3)
    ds.createDimension("x", 4)
    zv = ds.createVariable("field", np.complex128, ("y", "x"))
    zv.units = "V/m"
    zv[:] = z

with netCDF4.Dataset(OUT, auto_complex=True) as ds:
    for name, v in ds.variables.items():
        print(name, v.dimensions, v.dtype, np.asarray(v[:]).ravel())
