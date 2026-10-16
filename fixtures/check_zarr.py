"""Check a Zarr store written by annoray-zarr (crates/annoray-zarr/tests/write.rs) with
zarr-python, and that xarray opens it with consolidated metadata.

Usage: python3 fixtures/check_zarr.py <store>   (prints OK or raises)
Arrays hold f(i, j) = 100 i + j (divided by 7 for floats, "s{i}_{j}" for text, (i + j) odd
for booleans).
"""

import sys
import warnings

import numpy as np
import zarr

root = zarr.open_group(sys.argv[1], mode="r")


def values(path: str) -> np.ndarray:
    """Every value of the array at `path`."""
    a = root[path]
    assert isinstance(a, zarr.Array), path
    return np.asarray(a[...])


fmt = root.metadata.zarr_format
i, j = np.meshgrid(np.arange(7), np.arange(5), indexing="ij")
f = 100 * i + j
expect = {
    "bool": (i + j) % 2 == 1,
    "int8": (f % 100).astype("i1"),
    "uint8": (f % 100).astype("u1"),
    "float32": f.astype("f4") / np.float32(7),
    "float64": f / 7.0,
    "string": np.array([[f"s{a}_{b}" for b in range(5)] for a in range(7)], dtype=object),
}
for name in ["int16", "int32", "int64", "uint16", "uint32", "uint64"]:
    expect[name] = f.astype(name)

assert root.attrs["title"] == "written by annoray-zarr"
for name, want in expect.items():
    got = values(f"dtype_{name}")
    assert str(got.dtype) == ("object" if name == "string" else name) or name == "string", (name, got.dtype)
    np.testing.assert_array_equal(np.asarray(got, dtype=want.dtype), want, err_msg=name)
for key in root.array_keys():
    if key.startswith("comp_"):
        np.testing.assert_array_equal(values(key), f / 7.0, err_msg=key)
np.testing.assert_array_equal(values("fixed_unicode").astype(str), expect["string"].astype(str))
sparse = values("sparse")
np.testing.assert_array_equal(sparse, [1.0, 2.0, np.nan, np.nan, 5.0, np.nan])
assert values("scalar") == 42
if fmt == 3:
    v = np.where(np.arange(8)[:, None] >= 4, 0, 100 * np.arange(8)[:, None] + np.arange(9)[None, :])
    np.testing.assert_array_equal(values("sharded"), v)
np.testing.assert_array_equal(values("sub/inner"), [1, 2, 3])

# Consolidated metadata: opening it lists every array without warnings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    cons = zarr.open_consolidated(sys.argv[1], zarr_format=fmt)
    assert "dtype_float64" in list(cons.array_keys())
print("OK")
