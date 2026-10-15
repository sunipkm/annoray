"""annoray quick start: build, inspect, select, compute with units, reduce.

After xarray's "Quick overview". The Rust twin is `annoray/examples/quickstart.rs`; both
print the same lines.

    python examples/quickstart.py
"""

import numpy as np

import annoray as ar


def fmt(v):
    return f"{float(v):.3f}"


def show(label, values):
    print(f"{label}: [" + ", ".join(fmt(v) for v in np.ravel(values)) + "]")


# A DataArray: values, dimension names, coordinates and a unit.
da = ar.DataArray(
    np.arange(6.0).reshape(2, 3),
    dims=("x", "y"),
    coords={"x": [10.0, 20.0], "y": [1.0, 2.0, 3.0]},
    name="height",
    units="m",
)
print(f"dims: {', '.join(da.dims)}")
print("sizes: " + ", ".join(f"{k}={n}" for k, n in da.sizes.items()))
print(f"units: {da.units}")

# Selection by position (isel) and by label (sel); label slices include both ends.
show("isel x=0", da.isel(x=0).values)
show("sel x=20", da.sel(x=20.0).values)
show("sel y=1.5:3", da.sel(y=slice(1.5, 3.0)).values)
show("sel y~2.4 (nearest)", da.sel(y=2.4, method="nearest").values)

# Arithmetic carries units, and refuses what does not make sense.
seconds = ar.DataArray(2.0, units="s")
print(f"da * 2 -> {(da * 2.0).units}")
print(f"da * da -> {(da * da).units}")
print(f"da / 2 s -> {(da / seconds).units}")
try:
    da + seconds
except ar.UnitError as e:
    print(f"da + 2 s -> error: {e}")

# Reductions, by dimension name.
show("mean over x", da.mean("x").values)
show("std over y (ddof=1)", da.std("y", ddof=1).values)
show("sum", da.sum().values)

# Unit conversion.
show("in cm", da.to("cm").values)

# A Dataset: variables sharing dimensions; arithmetic broadcasts by name.
ds = ar.Dataset(
    {"height": da, "width": ("y", np.array([1.0, 2.0, 4.0]), {"units": "m"})},
    attrs={"title": "quick start"},
)
area = ds["height"] * ds["width"]
print(f"area units: {area.units}, dims: {', '.join(area.dims)}")
show("area", area.values)
print(ds)
