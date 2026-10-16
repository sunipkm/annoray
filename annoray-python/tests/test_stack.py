"""stack and unstack, checked against xarray."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")


def pair():
    v = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    x = xr.DataArray(
        v,
        dims=("w", "x", "y"),
        coords={"x": [30.0, 10.0, 20.0], "y": ["a", "b", "c", "d"], "t": ("x", [1.0, 2.0, 3.0])},
        name="v",
    )
    return ar.from_xarray(x), x


def same(a, x):
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(x.values))
    assert tuple(a.dims) == tuple(x.dims)


def test_stack_matches_xarray():
    a, x = pair()
    s, xs = a.stack(z=("x", "y")), x.stack(z=("x", "y"))
    same(s, xs)
    for k in ["x", "y", "t"]:
        np.testing.assert_array_equal(s.coords[k].values, xs.coords[k].values)
        assert tuple(s.coords[k].dims) == tuple(xs.coords[k].dims)
    same(a.stack(z=("y", "w")), x.stack(z=("y", "w")))
    # `...` is the remaining dimensions in the array's order (xarray orders them through a
    # set, so its order depends on the hash seed).
    same(a.stack({"z": ["x", ...]}), x.stack(z=("x", "w", "y")))
    # A dimension without a coordinate gets 0..n as its level.
    np.testing.assert_array_equal(a.stack(z=("w", "x")).coords["w"].values, [0, 0, 0, 1, 1, 1])


def test_round_trip_and_order():
    a, x = pair()
    for dims in [("x", "y"), ("y", "x"), ("w", "y")]:
        got = a.stack(z=dims).unstack("z")
        want = x.stack(z=dims).unstack("z")
        same(got, want)
        for k in dims:
            np.testing.assert_array_equal(got.coords[k].values, want.coords[k].values)
    same(a.stack(z=("x", "y")).unstack(), x.stack(z=("x", "y")).unstack())


def test_absent_combinations():
    a, x = pair()
    # Dropping whole-level-preserving positions keeps the level order the same as xarray's.
    keep = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11]
    got = a.stack(z=("x", "y")).isel(z=keep).unstack("z")
    want = x.stack(z=("x", "y")).isel(z=keep).unstack("z")
    same(got, want)
    got = a.stack(z=("x", "y")).isel(z=keep).unstack("z", fill_value=-1.0)
    want = x.stack(z=("x", "y")).isel(z=keep).unstack("z", fill_value=-1.0)
    same(got, want)
    ints = xr.DataArray(np.arange(4, dtype="int32").reshape(2, 2), dims=("p", "q"))
    ai = ar.from_xarray(ints)
    got = ai.stack(z=("p", "q")).isel(z=[0, 1, 3]).unstack("z")
    want = ints.stack(z=("p", "q")).isel(z=[0, 1, 3]).unstack("z")
    same(got, want)
    assert got.values.dtype == want.values.dtype


def test_dataset():
    a, x = pair()
    ds = ar.Dataset({"v": a, "u": a * 2.0})
    xs = xr.Dataset({"v": x, "u": x * 2.0})
    s, xst = ds.stack(z=("x", "y")), xs.stack(z=("x", "y"))
    same(s["u"], xst["u"])
    same(s.unstack("z")["v"], xst.unstack("z")["v"])


@pytest.mark.parametrize("fmt", ["NETCDF4", "NETCDF3_CLASSIC"])
def test_stacked_files_round_trip(tmp_path, fmt):
    a, _ = pair()
    ds = ar.Dataset({"v": a}).stack(z=("x", "y"))
    path = tmp_path / "s.nc"
    ar.to_netcdf(ds, path, format=fmt)
    back = ar.open_dataset(path)
    same(back.unstack("z")["v"], ds.unstack("z")["v"])
    assert "z" not in back.coords
    # On disk it is CF compression by gathering, which xarray reads as plain variables.
    raw = xr.open_dataset(path)
    assert raw["z"].attrs["compress"] == "x y"
    np.testing.assert_array_equal(raw["z"].values, np.arange(12))
    np.testing.assert_array_equal(raw["x"].values, [30.0, 10.0, 20.0])
    raw.close()


def test_cf_xarray_interop(tmp_path):
    cfx = pytest.importorskip("cf_xarray")
    _, x = pair()
    xs = xr.Dataset({"v": x}).stack(z=("x", "y"))
    path = tmp_path / "cfx.nc"
    cfx.encode_multi_index_as_compress(xs, "z").to_netcdf(path)
    back = ar.open_dataset(path)
    same(back.unstack("z")["v"], xs.unstack("z")["v"])
    ours = tmp_path / "ours.nc"
    ar.to_netcdf(back, ours)
    decoded = cfx.decode_compress_to_multi_index(xr.open_dataset(ours), "z")
    same(back["v"], decoded["v"])
