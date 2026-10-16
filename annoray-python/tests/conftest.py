"""Test configuration.

Comparisons against xarray use xarray's numpy code paths, which annoray matches to the
bit. With numbagg installed xarray computes some reductions (quantiles) through numbagg,
whose rounding differs by an ulp; ``rolling_exp`` imports numbagg directly and is
unaffected by this option.
"""

from __future__ import annotations

import contextlib


def pytest_configure(config: object) -> None:
    with contextlib.suppress(ImportError):
        import xarray as xr

        xr.set_options(use_numbagg=False)
