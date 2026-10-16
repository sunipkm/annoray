"""Rolling windows over several dimensions at once: ``da.rolling(x=2, y=3).mean()``.

Built from the one-dimensional windows (``construct``), as xarray does: each position's
window is the product of the per-dimension windows; a result needs at least
``min_periods`` valid values (the full window size by default), else it is missing."""

from __future__ import annotations

import math
from typing import Any


class MultiRolling:
    __slots__ = ("_center", "_min_periods", "_obj", "_windows")

    def __init__(self, obj: Any, windows: dict[str, int], center: bool, min_periods: int | None):
        self._obj = obj
        self._windows = windows
        self._center = center
        self._min_periods = math.prod(windows.values()) if min_periods is None else min_periods

    def __repr__(self) -> str:
        w = ", ".join(f"{d}={n}" for d, n in self._windows.items())
        return f"<annoray.Rolling ({w}), center={self._center}>"

    def construct(self, window_dim: Any = None, **window_dim_kwargs: str) -> Any:
        """The windows as new trailing dimensions, named by ``{dim: name}``."""
        names = {**(window_dim or {}), **window_dim_kwargs}
        if set(names) != set(self._windows):
            raise ValueError(
                f"construct needs a window dimension name for each of {list(self._windows)}"
            )
        out = self._obj
        for d, n in self._windows.items():
            out = out.rolling({d: n}, center=self._center).construct(names[d])
        return out

    def _reduce(self, op: str) -> Any:
        names = {d: f"_window_{d}" for d in self._windows}
        windows = self.construct(names)
        dims = list(names.values())
        count = windows.notnull().sum(dims)
        result = getattr(windows, op)(dims)
        return result.where(count >= self._min_periods)

    def sum(self) -> Any:
        return self._reduce("sum")

    def mean(self) -> Any:
        return self._reduce("mean")

    def min(self) -> Any:
        return self._reduce("min")

    def max(self) -> Any:
        return self._reduce("max")

    def std(self) -> Any:
        return self._reduce("std")

    def var(self) -> Any:
        return self._reduce("var")

    def median(self) -> Any:
        return self._reduce("median")

    def prod(self) -> Any:
        return self._reduce("prod")

    def count(self) -> Any:
        return self._reduce("count")
