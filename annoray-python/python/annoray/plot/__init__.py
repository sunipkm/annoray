"""Plotting with matplotlib, following ``xarray.plot``: ``da.plot()`` and ``da.plot.<kind>()``
(``line``, ``step``, ``hist``, ``scatter``, ``pcolormesh``, ``imshow``, ``contour``,
``contourf``, ``surface``), ``ds.plot.<kind>()`` (``scatter``, ``quiver``, ``streamplot``),
and :class:`FacetGrid` small multiples through ``row``/``col``.

Axis and colorbar labels come from ``long_name`` (else ``standard_name``, else the name)
and the array's unit. matplotlib is imported on first use.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from annoray.plot._dataarray import (
    contour,
    contourf,
    hist,
    imshow,
    line,
    pcolormesh,
    plot,
    scatter,
    step,
    surface,
)
from annoray.plot._dataset import quiver, streamplot
from annoray.plot._dataset import scatter as dataset_scatter
from annoray.plot._facetgrid import FacetGrid
from annoray.plot._utils import OPTIONS

if TYPE_CHECKING:
    from annoray._annoray import DataArray, Dataset


def _require_matplotlib() -> None:
    """Plotting needs matplotlib, which is an optional extra."""
    if find_spec("matplotlib") is None:
        raise ImportError("plotting needs matplotlib: install it with `pip install annoray[plot]`")


__all__ = [
    "OPTIONS",
    "DataArrayPlotAccessor",
    "DatasetPlotAccessor",
    "FacetGrid",
    "contour",
    "contourf",
    "dataset_scatter",
    "hist",
    "imshow",
    "line",
    "pcolormesh",
    "plot",
    "quiver",
    "scatter",
    "step",
    "streamplot",
    "surface",
]


class DataArrayPlotAccessor:
    """``da.plot``: call it for the default plot by rank, or pick a kind."""

    __slots__ = ("_da",)

    def __init__(self, darray: DataArray) -> None:
        _require_matplotlib()
        self._da = darray

    def __call__(self, **kwargs: Any) -> Any:
        return plot(self._da, **kwargs)

    def line(self, *args: Any, **kwargs: Any) -> Any:
        return line(self._da, *args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return step(self._da, *args, **kwargs)

    def hist(self, **kwargs: Any) -> Any:
        return hist(self._da, **kwargs)

    def scatter(self, *args: Any, **kwargs: Any) -> Any:
        return scatter(self._da, *args, **kwargs)

    def pcolormesh(self, **kwargs: Any) -> Any:
        return pcolormesh(self._da, **kwargs)

    def imshow(self, **kwargs: Any) -> Any:
        return imshow(self._da, **kwargs)

    def contour(self, **kwargs: Any) -> Any:
        return contour(self._da, **kwargs)

    def contourf(self, **kwargs: Any) -> Any:
        return contourf(self._da, **kwargs)

    def surface(self, **kwargs: Any) -> Any:
        return surface(self._da, **kwargs)


class DatasetPlotAccessor:
    """``ds.plot``: plots of variables against each other."""

    __slots__ = ("_ds",)

    def __init__(self, dataset: Dataset) -> None:
        _require_matplotlib()
        self._ds = dataset

    def __call__(self, **kwargs: Any) -> Any:
        raise ValueError(
            "Dataset.plot cannot be called directly. Use an explicit plot method, "
            "e.g. ds.plot.scatter(...)"
        )

    def scatter(self, *args: Any, **kwargs: Any) -> Any:
        return dataset_scatter(self._ds, *args, **kwargs)

    def quiver(self, **kwargs: Any) -> Any:
        return quiver(self._ds, **kwargs)

    def streamplot(self, **kwargs: Any) -> Any:
        return streamplot(self._ds, **kwargs)


# The accessor methods carry the plotting functions' documentation.
for _name in (
    "line",
    "step",
    "hist",
    "scatter",
    "pcolormesh",
    "imshow",
    "contour",
    "contourf",
    "surface",
):
    getattr(DataArrayPlotAccessor, _name).__doc__ = globals()[_name].__doc__
for _name, _func in (("scatter", dataset_scatter), ("quiver", quiver), ("streamplot", streamplot)):
    getattr(DatasetPlotAccessor, _name).__doc__ = _func.__doc__
DataArrayPlotAccessor.__call__.__doc__ = plot.__doc__
