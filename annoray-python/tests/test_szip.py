"""szip (HDF5 filter 4) chunks, as netCDF-C writes them through libaec: decoded bit for bit,
for every integer width and both floats, with and without the nearest-neighbour
preprocessor, for several block sizes and scanlines that do not fill whole blocks."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

nc = pytest.importorskip("netCDF4")
h5py = pytest.importorskip("h5py")
if not getattr(nc, "__has_szip_support__", False):
    pytest.skip("netCDF4 without szip", allow_module_level=True)

DTYPES = ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"]


def values(dt, rng, shape):
    if dt[0] == "f":
        # Steps compress; noise would be stored raw (HDF5 skips a filter that does not help).
        return np.repeat(np.arange(shape[0] // 10), 10)[:, None].repeat(shape[1], 1).astype(dt)
    info = np.iinfo(dt)
    walk = np.cumsum(rng.integers(-3, 4, size=shape), axis=0) + (50 if dt[0] == "u" else 0)
    data = np.clip(walk, info.min, info.max).astype(dt)
    data[shape[0] // 4 : shape[0] // 2] = 0  # zero blocks
    return data


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("coding", ["nn", "ec"])
@pytest.mark.parametrize("ppb", [8, 32])
def test_szip_matches_netcdf_c(tmp_path, dt, coding, ppb):
    rng = np.random.default_rng(0)
    data = values(dt, rng, (300, 37))
    path = tmp_path / "s.nc"
    with nc.Dataset(path, "w") as f:
        f.createDimension("x", 300)
        f.createDimension("y", 37)
        v = f.createVariable(
            "v",
            dt,
            ("x", "y"),
            compression="szip",
            szip_coding=coding,
            szip_pixels_per_block=ppb,
            chunksizes=(100, 37),
        )
        v[:] = data
    with h5py.File(path) as f:
        compressed = f["v"].id.get_storage_size() < data.nbytes
    for lazy in (True, False):
        got = ar.open_dataset(path, mask_and_scale=False, lazy=lazy)["v"].values
        np.testing.assert_array_equal(got, data)
    if not compressed:
        pytest.skip("libaec stored every chunk raw")


def test_szip_constant_and_noise(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "s.nc"
    noise = rng.integers(0, 65535, size=(300, 37)).astype("u2")
    with nc.Dataset(path, "w") as f:
        f.createDimension("x", 300)
        f.createDimension("y", 37)
        for name, dt, data in [("zeros", "i4", np.zeros((300, 37), "i4")), ("noise", "u2", noise)]:
            v = f.createVariable(name, dt, ("x", "y"), compression="szip", szip_pixels_per_block=32)
            v[:] = data
    ds = ar.open_dataset(path, mask_and_scale=False)
    assert (ds["zeros"].values == 0).all()
    np.testing.assert_array_equal(ds["noise"].values, noise)
