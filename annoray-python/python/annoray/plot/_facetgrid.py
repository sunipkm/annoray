"""Small multiples: one subplot per value of a ``row`` and/or ``col`` dimension."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from annoray.plot import _utils as u

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from annoray._annoray import DataArray, Dataset

#: Arguments that set the shared colour scale, consumed by the grid.
_CMAP_ARGS = ("vmin", "vmax", "cmap", "center", "robust", "extend", "levels", "norm")


class FacetGrid:
    """A grid of axes with one subplot per value of ``row`` and/or ``col``, sharing axes
    and colour scale (as ``xarray.plot.FacetGrid``).

    Attributes: ``fig`` (also ``figure``), ``axs`` (also ``axes``; a 2-D array),
    ``name_dicts`` (per subplot, ``{dim: value}``, ``None`` for unused cells),
    ``row_names``/``col_names``, ``cbar``, ``figlegend``.
    """

    def __init__(
        self,
        data: DataArray | Dataset,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | str | None = None,
        sharex: bool = True,
        sharey: bool = True,
        figsize: Sequence[float] | None = None,
        aspect: float = 1,
        size: float = 3,
        subplot_kws: Mapping[str, Any] | None = None,
    ) -> None:
        plt = u.import_pyplot()
        for name in (row, col):
            if name is not None and name not in data.dims:
                raise ValueError(f"facet dimension {name!r} is not one of {tuple(data.dims)}")
        if row is not None and col is not None and col_wrap is not None:
            raise ValueError("Can't use col_wrap with both row and col")
        nrow_values = data.sizes[row] if row is not None else 1
        ncol_values = data.sizes[col] if col is not None else 1
        if col is not None and row is None and col_wrap is not None:
            n = ncol_values
            if col_wrap == "auto":
                ncol = max(1, math.ceil(math.sqrt(n)))
            else:
                ncol = min(int(col_wrap), n)
            nrow = math.ceil(n / ncol)
        else:
            nrow, ncol = nrow_values, ncol_values
        if figsize is None:
            figsize = (ncol * size * aspect, nrow * size)
        fig, axs = plt.subplots(
            nrow,
            ncol,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
            figsize=figsize,
            subplot_kw=dict(subplot_kws or {}),
        )
        self.data = data
        self.fig: Figure = fig
        self.axs: np.ndarray = axs
        self._row_var, self._col_var = row, col
        self.row_names = list(np.asarray(u.coord(data, row).values)) if row else [None]
        self.col_names = list(np.asarray(u.coord(data, col).values)) if col else [None]
        # Per subplot: positional indexers (for isel) and the facet values (for titles).
        self._indexers = np.full((nrow, ncol), None, dtype=object)
        self.name_dicts = np.full((nrow, ncol), None, dtype=object)
        cells: list[dict[str, int]] = []
        for i in range(nrow_values):
            for j in range(ncol_values):
                ix: dict[str, int] = {}
                if row is not None:
                    ix[row] = i
                if col is not None:
                    ix[col] = j
                cells.append(ix)
        for k, ix in enumerate(cells):
            r, c = (
                (k // ncol, k % ncol)
                if (col_wrap is not None and row is None)
                else (
                    ix.get(row, 0) if row else 0,
                    ix.get(col, 0) if col else 0,
                )
            )
            self._indexers[r, c] = ix
            names = {}
            if row is not None:
                names[row] = self.row_names[ix[row]]
            if col is not None:
                names[col] = self.col_names[ix[col]]
            self.name_dicts[r, c] = names
        for ax, ix in zip(self.axs.flat, self._indexers.flat, strict=True):
            if ix is None:
                ax.set_visible(False)
        self.cbar: Any = None
        self.figlegend: Any = None
        self._mappables: list[Any] = []
        self._cmap_extend: str | None = None
        self._x_var: str | None = None
        self._y_var: str | None = None
        self._hue_var: str | None = None
        self._finalized = False

    # ----- access -----

    @property
    def figure(self) -> Figure:
        return self.fig

    @property
    def axes(self) -> np.ndarray:
        return self.axs

    def __iter__(self) -> Iterator[Axes]:
        return iter(self.axs.flat)

    def _cells(self) -> Iterator[tuple[Axes, DataArray | Dataset]]:
        for ax, ix in zip(self.axs.flat, self._indexers.flat, strict=True):
            if ix is not None:
                yield ax, self.data.isel(ix)

    @property
    def _left_axes(self) -> list[Axes]:
        return [self.axs[i, 0] for i in range(self.axs.shape[0]) if self._indexers[i, 0]]

    @property
    def _bottom_axes(self) -> list[Axes]:
        out = []
        nrow, ncol = self.axs.shape
        for j in range(ncol):
            rows = [i for i in range(nrow) if self._indexers[i, j] is not None]
            if rows:
                out.append(self.axs[max(rows), j])
        return out

    # ----- mapping -----

    def map_dataarray(
        self, func: Callable[..., Any], x: str | None = None, y: str | None = None, **kwargs: Any
    ) -> FacetGrid:
        """Apply a 2-D plotting function (``pcolormesh``, ``contourf``, …) to each facet,
        with one colour scale for all of them and one colorbar."""
        name = getattr(func, "__name__", "")
        cbar_kwargs = dict(kwargs.pop("cbar_kwargs", None) or {})
        kwargs.pop("cbar_ax", None)
        add_colorbar = kwargs.pop("add_colorbar", True)
        add_labels = kwargs.pop("add_labels", True)
        cmap_kw = {k: kwargs.pop(k) for k in _CMAP_ARGS if k in kwargs}
        colors = kwargs.get("colors")
        values = np.asarray(self.data.values)
        rgb = name == "imshow" and self.data.ndim == 3 + sum(
            v is not None for v in (self._row_var, self._col_var)
        )
        if name == "surface" or rgb:
            params = {
                k: cmap_kw.get(k) for k in ("vmin", "vmax", "cmap", "extend", "levels", "norm")
            }
            if rgb:
                kwargs.update({k: cmap_kw[k] for k in ("vmin", "vmax", "robust") if k in cmap_kw})
        else:
            if cmap_kw.get("cmap") and colors:
                raise ValueError("Can't specify both cmap and colors.")
            params = u.cmap_params(
                values,
                filled=name != "contour",
                **{**cmap_kw, "cmap": colors or cmap_kw.get("cmap")},
            )
        self._cmap_extend = params["extend"]
        from annoray.plot._dataarray import _plot2d

        for ax, sub in self._cells():
            mappable = _plot2d(
                name,
                sub,  # type: ignore[arg-type]
                x=x,
                y=y,
                ax=ax,
                add_colorbar=False,
                add_labels=False,
                _facet_params=params,
                **kwargs,
            )
            self._mappables.append(mappable)
        first = next(self._cells())[1]
        xname, yname = (x, y) if rgb else u.infer_xy(first, x, y)  # type: ignore[arg-type]
        if xname is None or yname is None:
            from annoray.plot._dataarray import _infer_xy_rgb

            xname, yname, _ = _infer_xy_rgb(first, x, y, kwargs.get("rgb"))  # type: ignore[arg-type]
        self._finalize_grid(xname, yname, add_labels)
        if add_colorbar:
            if add_labels and "label" not in cbar_kwargs:
                cbar_kwargs["label"] = u.label(self.data)  # type: ignore[arg-type]
            self.add_colorbar(**cbar_kwargs)
        return self

    def map_dataarray_line(
        self,
        func: Callable[..., Any],
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        add_legend: bool = True,
        **kwargs: Any,
    ) -> FacetGrid:
        """Apply :func:`~annoray.plot.line` (or ``step``) to each facet, with one legend."""
        from annoray.plot._dataarray import _line_data

        handles = None
        for ax, sub in self._cells():
            handles = func(sub, x=x, y=y, ax=ax, hue=hue, add_legend=False, _labels=False, **kwargs)
            self._mappables.append(handles)
        first = next(self._cells())[1]
        xplt, yplt, _, _, hplt, hue_label = _line_data(first, x, y, hue)  # type: ignore[arg-type]
        self._hue_var = hue
        for ax in self._bottom_axes:
            ax.set_xlabel(u.label(xplt))
        for ax in self._left_axes:
            ax.set_ylabel(u.label(yplt))
        self.set_titles()
        self._finalized = True
        if add_legend and hplt is not None and handles is not None:
            self.add_legend(handles=handles, labels=list(np.asarray(hplt.values)), title=hue_label)
        return self

    def map_plot1d(self, func: Callable[..., Any], **kwargs: Any) -> FacetGrid:
        """Apply :func:`~annoray.plot.scatter` to each facet with a shared colour scale."""
        hue = kwargs.get("hue")
        add_colorbar = kwargs.pop("add_colorbar", None)
        add_legend = kwargs.pop("add_legend", None)
        numeric = False
        if hue is not None:
            hv = np.asarray(u.coord(self.data, hue).values)
            numeric = hv.dtype.kind in "iufcmM"
            if numeric:
                p = u.cmap_params(
                    hv,
                    vmin=kwargs.pop("vmin", None),
                    vmax=kwargs.pop("vmax", None),
                    cmap=kwargs.pop("cmap", None),
                    norm=kwargs.pop("norm", None),
                )
                kwargs.update(vmin=p["vmin"], vmax=p["vmax"], cmap=p["cmap"], norm=p["norm"])
                self._cmap_extend = p["extend"]
        for ax, sub in self._cells():
            m = func(sub, ax=ax, add_labels=False, add_colorbar=False, add_legend=False, **kwargs)
            self._mappables.append(m)
        first = next(self._cells())[1]
        xname = kwargs.get("x") or first.dims[0]
        for ax in self._bottom_axes:
            ax.set_xlabel(u.label(u.coord(first, xname)))
        for ax in self._left_axes:
            ax.set_ylabel(u.label(first))  # type: ignore[arg-type]
        self.set_titles()
        self._finalized = True
        if hue is not None:
            hue_label = u.label(u.coord(self.data, hue))
            if numeric and (add_colorbar is None or add_colorbar):
                self.add_colorbar(label=hue_label)
            elif not numeric and (add_legend is None or add_legend):
                handles = self._mappables[-1]
                self.add_legend(
                    handles=handles, labels=[h.get_label() for h in handles], title=hue_label
                )
        return self

    def map_dataset(self, func: Callable[..., Any], **kwargs: Any) -> FacetGrid:
        """Apply a Dataset plot (``scatter``, ``quiver``, ``streamplot``) to each facet with a
        shared ``hue`` colour scale."""
        hue = kwargs.get("hue")
        name = getattr(func, "__name__", "")
        add_guide = kwargs.pop("add_guide", None)
        if name == "scatter":
            add_guide = kwargs.pop("add_colorbar", add_guide)
            kwargs.pop("add_legend", None)
        if hue is not None:
            hv = np.asarray(u.coord(self.data, hue).values)
            if hv.dtype.kind in "iufcmM":
                p = u.cmap_params(
                    hv,
                    vmin=kwargs.pop("vmin", None),
                    vmax=kwargs.pop("vmax", None),
                    cmap=kwargs.pop("cmap", None),
                    norm=kwargs.pop("norm", None),
                )
                kwargs.update(vmin=p["vmin"], vmax=p["vmax"], cmap=p["cmap"], norm=p["norm"])
                self._cmap_extend = p["extend"]
        for ax, sub in self._cells():
            if name == "scatter":
                m = func(
                    sub, ax=ax, add_labels=False, add_colorbar=False, add_legend=False, **kwargs
                )
            else:
                m = func(sub, ax=ax, _facet=True, **kwargs)
            self._mappables.append(m)
        self._finalize_grid(kwargs.get("x"), kwargs.get("y"), True)
        if hue is not None and (add_guide is None or add_guide):
            hue_label = u.label(u.coord(self.data, hue))
            if isinstance(self._mappables[-1], list):
                handles = self._mappables[-1]
                self.add_legend(
                    handles=handles, labels=[h.get_label() for h in handles], title=hue_label
                )
            else:
                self.add_colorbar(label=hue_label)
        if name == "quiver" and (add_guide is None or add_guide):
            from annoray.plot._dataset import _nice_magnitude

            uname, vname = kwargs["u"], kwargs["v"]
            mag = _nice_magnitude(
                np.asarray(u.coord(self.data, uname).values),
                np.asarray(u.coord(self.data, vname).values),
            )
            unit = u.units_of(u.coord(self.data, uname)) or ""
            self.axs.flat[0].quiverkey(
                self._mappables[-1],
                X=0.8,
                Y=0.9,
                U=mag,
                label=f"{mag}\n{unit}",
                labelpos="E",
                coordinates="figure",
            )
        return self

    def map(self, func: Callable[..., Any], *args: str, **kwargs: Any) -> FacetGrid:
        """Call ``func(*values, **kwargs)`` with the current axes set to each facet, where
        ``values`` are the named variables/coordinates of that facet's subset."""
        plt = u.import_pyplot()
        for ax, sub in self._cells():
            plt.sca(ax)
            inner = [np.asarray(u.coord(sub, a).values) for a in args]
            artist = func(*inner, **kwargs)
            if artist is not None and hasattr(artist, "get_array"):
                self._mappables.append(artist)
        self._finalize_grid(args[0] if args else None, args[1] if len(args) > 1 else None)
        return self

    # ----- decoration -----

    def _finalize_grid(
        self, x: str | None = None, y: str | None = None, add_labels: bool = True
    ) -> None:
        if self._finalized:
            return
        self._x_var, self._y_var = x, y
        if add_labels:
            self.set_axis_labels(x, y)
        self.set_titles()
        for ax, ix in zip(self.axs.flat, self._indexers.flat, strict=True):
            if ix is None:
                ax.set_visible(False)
        self._finalized = True

    def set_axis_labels(self, x_var: str | None = None, y_var: str | None = None) -> None:
        """Label the outer axes from the coordinates' metadata (or with the given text)."""
        for var, axes, setter in (
            (x_var, self._bottom_axes, "set_xlabel"),
            (y_var, self._left_axes, "set_ylabel"),
        ):
            if var is None:
                continue
            try:
                text = u.label(u.coord(self.data, var))
            except ValueError:  # not a name in the data: the label itself
                text = var
            for ax in axes:
                getattr(ax, setter)(text)

    def set_xlabels(self, label: str | None = None, **kwargs: Any) -> None:
        """Label the bottom x axes (default: from the x coordinate)."""
        if label is None and self._x_var is not None:
            label = u.label(u.coord(self.data, self._x_var))
        for ax in self._bottom_axes:
            ax.set_xlabel(label or "", **kwargs)

    def set_ylabels(self, label: str | None = None, **kwargs: Any) -> None:
        """Label the left y axes (default: from the y coordinate)."""
        if label is None and self._y_var is not None:
            label = u.label(u.coord(self.data, self._y_var))
        for ax in self._left_axes:
            ax.set_ylabel(label or "", **kwargs)

    def set_titles(
        self,
        template: str = "{coord} = {value}",
        maxchar: int = 30,
        size: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Title each facet with its ``row``/``col`` value; with both, column values head the
        top row and row values annotate the right column."""
        plt = u.import_pyplot()
        if size is None:
            size = plt.rcParams["axes.labelsize"]

        def fmt(coord: str, value: Any) -> str:
            t = template.format(coord=coord, value=u.format_item(value))
            return t if len(t) <= maxchar else t[: maxchar - 3] + "..."

        row, col = self._row_var, self._col_var
        if row is None or col is None:
            for ax, names in zip(self.axs.flat, self.name_dicts.flat, strict=True):
                if names:
                    ((k, v),) = names.items()
                    ax.set_title(fmt(k, v), size=size, **kwargs)
            return
        for j, value in enumerate(self.col_names):
            self.axs[0, j].set_title(fmt(col, value), size=size, **kwargs)
        for i, value in enumerate(self.row_names):
            self.axs[i, -1].annotate(
                fmt(row, value),
                xy=(1.02, 0.5),
                xycoords="axes fraction",
                rotation=270,
                ha="left",
                va="center",
                size=size,
                **kwargs,
            )

    def set_ticks(self, max_xticks: int = 4, max_yticks: int = 4, fontsize: Any = None) -> None:
        """At most ``max_xticks``/``max_yticks`` ticks per facet."""
        from matplotlib.ticker import MaxNLocator

        for ax in self.axs.flat:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=max_yticks, prune="both"))
            if fontsize is not None:
                for t in ax.get_xticklabels() + ax.get_yticklabels():
                    t.set_fontsize(fontsize)

    def add_colorbar(self, **kwargs: Any) -> None:
        """One colorbar for all facets, from the last mappable."""
        kwargs = dict(kwargs)
        if self._cmap_extend is not None:
            kwargs.setdefault("extend", self._cmap_extend)
        if hasattr(self._mappables[-1], "extend"):
            kwargs.pop("extend", None)
        from annoray._annoray import DataArray

        kwargs.setdefault("label", u.label(self.data) if isinstance(self.data, DataArray) else "")
        self.cbar = self.fig.colorbar(self._mappables[-1], ax=list(self.axs.flat), **kwargs)

    def add_legend(self, **kwargs: Any) -> None:
        """One figure legend, to the right of the facets."""
        kwargs.setdefault("loc", "center right")
        self.figlegend = self.fig.legend(**kwargs)
        self.fig.subplots_adjust(right=0.85)


__all__ = ["FacetGrid"]
