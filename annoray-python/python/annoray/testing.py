"""Assertions for tests (xarray's ``xarray.testing``): ``assert_equal``,
``assert_identical``, ``assert_allclose``.

Units count: ``assert_equal`` and ``assert_allclose`` convert between units of a kind
(1 km equals 1000 m) and fail for different kinds; ``assert_identical`` needs the same
units, names and attributes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from annoray._annoray import DataArray, Dataset

__all__ = ["assert_allclose", "assert_equal", "assert_identical"]


def _check_types(a: Any, b: Any) -> None:
    __tracebackhide__ = True
    if type(a) is not type(b) or not isinstance(a, (DataArray, Dataset)):
        raise AssertionError(
            f"expected two DataArrays or two Datasets, got {type(a).__name__} and "
            f"{type(b).__name__}"
        )


def _ordered(a: Any, b: Any, check_dim_order: bool) -> Any:
    if not check_dim_order and isinstance(b, DataArray) and set(a.dims) == set(b.dims):
        return b.transpose(*a.dims)
    return b


def _fail(what: str, a: Any, b: Any) -> None:
    __tracebackhide__ = True
    raise AssertionError(f"{what}\n\nLeft:\n{a!r}\n\nRight:\n{b!r}")


def assert_equal(a: Any, b: Any, check_dim_order: bool = True) -> None:
    """Fail unless ``a.equals(b)``: same dimensions, coordinates and values (units
    converted; NaN equals NaN)."""
    __tracebackhide__ = True
    _check_types(a, b)
    if not a.equals(_ordered(a, b, check_dim_order)):
        _fail(f"{type(a).__name__} objects are not equal", a, b)


def assert_identical(a: Any, b: Any) -> None:
    """Fail unless ``a.identical(b)``: equal, with the same units, names and attributes."""
    __tracebackhide__ = True
    _check_types(a, b)
    if not a.identical(b):
        _fail(f"{type(a).__name__} objects are not identical", a, b)


def _close_arrays(a: DataArray, b: DataArray, rtol: float, atol: float, what: str) -> None:
    __tracebackhide__ = True
    if a.dims != b.dims or a.shape != b.shape:
        raise AssertionError(f"{what}: dimensions differ: {a.sizes} and {b.sizes}")
    if (a.units is None) != (b.units is None):
        raise AssertionError(f"{what}: units differ: {a.units} and {b.units}")
    if a.units is not None and b.units != a.units:
        b = b.to(a.units)  # raises UnitError for another kind
    x, y = np.asarray(a.values), np.asarray(b.values)
    if x.dtype.kind in "fciub" and y.dtype.kind in "fciub":
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, equal_nan=True, err_msg=what)
    else:
        np.testing.assert_array_equal(x, y, err_msg=what)


def assert_allclose(
    a: Any, b: Any, rtol: float = 1e-05, atol: float = 1e-08, check_dim_order: bool = True
) -> None:
    """Fail unless ``a`` and ``b`` have the same dimensions and coordinates, and values
    within ``atol + rtol * |b|`` (numpy's rule; units converted first; NaN matches NaN)."""
    __tracebackhide__ = True
    _check_types(a, b)
    b = _ordered(a, b, check_dim_order)
    if isinstance(a, DataArray):
        _close_arrays(a, b, rtol, atol, "values")
        pairs = [(a.coords, b.coords, "coordinate")]
    else:
        pairs = [(a.data_vars, b.data_vars, "variable"), (a.coords, b.coords, "coordinate")]
    for left, right, kind in pairs:
        if set(left) != set(right):
            raise AssertionError(f"{kind}s differ: {sorted(left)} and {sorted(right)}")
        for name in left:
            _close_arrays(left[name], right[name], rtol, atol, f"{kind} {name!r}")
