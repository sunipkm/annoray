"""Plain Python numbers take the data's type (numpy's NEP 50 "weak" scalars), as in numpy and
xarray: `float32 * 2.0` stays float32, `int32 * 2` stays int32."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar


@pytest.mark.parametrize("dtype", ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"])
def test_python_ints_keep_integer_types(dtype):
    x = np.array([1, 2, 3], dtype)
    a = ar.DataArray(x, dims="t")
    for got, want in [(a * 2, x * 2), (2 * a, 2 * x), (a + 1, x + 1), (a - 1, x - 1)]:
        assert got.dtype == want.dtype, (dtype, got.dtype, want.dtype)
        np.testing.assert_array_equal(got.values, want)
    assert (a / 2).dtype == np.float64  # true division
    assert (a * 2.5).dtype == np.float64  # a Python float is float64 beside integers


def test_with_units_too():
    a = ar.DataArray(np.array([1, 2], "i2"), dims="t", attrs={"units": "m"})
    assert (a * 3).dtype == np.int16 and str((a * 3).units) == "m"


def test_out_of_range_is_an_error_not_wrapped():
    a = ar.DataArray(np.array([1, 2], "u1"), dims="t")
    with pytest.raises(ValueError, match="out of range"):
        _ = a * 300


def test_float32_stays_float32():
    a = ar.DataArray(np.array([1.0, 2.0], "f4"), dims="t")
    assert (a * 2.0).dtype == np.float32 and (a + 1).dtype == np.float32
