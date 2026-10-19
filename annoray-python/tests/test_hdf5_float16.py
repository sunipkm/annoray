"""float16 (HDF5 only; netCDF has no such type): read exactly as float32, written back as
halves rounded like numpy's astype(np.float16)."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

h5py = pytest.importorskip("h5py")

RNG = np.random.default_rng(3)
SRC = np.concatenate(
    [
        RNG.standard_normal(2000) * 10.0 ** RNG.integers(-9, 6, 2000),
        [np.nan, np.inf, -np.inf, 65504, 65519.9, 65520, 1e-8, -0.0, 6e-8, 2.0**-25],
    ]
)


def halves(x):
    with np.errstate(over="ignore"):
        return np.asarray(x, dtype="f8").astype("<f2")


def same_bits(a, b):
    a, b = np.asarray(a).astype("<f2"), np.asarray(b).astype("<f2")
    m = ~np.isnan(b)
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(
        a.view("u2")[m], b.view("u2")[m]
    )


def test_read_as_exact_float32(tmp_path):
    p = tmp_path / "h.h5"
    with h5py.File(p, "w") as f:
        f["v"] = halves(SRC)
    v = ar.open_dataset(p)["v"]
    assert v.dtype == np.float32 and v.encoding["dtype"] == "float16"
    np.testing.assert_array_equal(v.values, halves(SRC).astype("f4"))


@pytest.mark.parametrize(
    "enc",
    [
        {"dtype": "float16"},
        {"dtype": "float16", "_FillValue": -1.0},
        {"dtype": "float16", "endian": "big", "zlib": True, "fletcher32": True},
    ],
)
def test_written_like_numpy(tmp_path, enc):
    p = tmp_path / "w.h5"
    ar.Dataset({"v": ar.DataArray(SRC, dims="x")}).to_netcdf(p, encoding={"v": enc})
    with h5py.File(p) as f:
        got, fill = f["v"][()], f["v"].attrs.get("_FillValue")
        assert f["v"].dtype.kind == "f" and f["v"].dtype.itemsize == 2
    want = halves(SRC)
    if "_FillValue" in enc:
        want = np.where(np.isnan(want), np.float16(enc["_FillValue"]), want)
    assert same_bits(got, want)
    assert fill is not None and fill.dtype.itemsize == 2  # the fill value is a half too


def test_round_trip_keeps_half_precision(tmp_path):
    p, q = tmp_path / "a.h5", tmp_path / "b.h5"
    with h5py.File(p, "w") as f:
        f["v"] = halves(SRC)
    ar.open_dataset(p).to_netcdf(q)
    with h5py.File(q) as f:
        assert f["v"].dtype == np.float16
        assert same_bits(f["v"][()], halves(SRC))
