"""Test configuration.

Comparisons against xarray use xarray's numpy and bottleneck code paths, which annoray
matches to the bit. Both options are pinned, as xarray's defaults change between versions:

- ``use_numbagg=False``: with numbagg, xarray computes some reductions (quantiles) through
  numbagg, whose rounding differs by an ulp. ``rolling_exp`` imports numbagg directly and is
  unaffected by this option.
- ``use_bottleneck=True``: xarray's ``ffill``/``bfill`` (and ``interpolate_na`` with
  ``max_gap``) need bottleneck or numbagg; xarray 2026.9 turned bottleneck off by default.
"""

from __future__ import annotations

import contextlib


def pytest_configure(config: object) -> None:
    with contextlib.suppress(ImportError):
        import xarray as xr

        xr.set_options(use_numbagg=False, use_bottleneck=True)
