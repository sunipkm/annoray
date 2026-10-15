"""Plots of a Dataset's variables against each other: ``scatter``, ``quiver``,
``streamplot`` (as ``xarray.Dataset.plot``). Variables and coordinates are broadcast against
each other by dimension name; ``row``/``col`` facet into a :class:`FacetGrid`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from annoray.plot import _utils as u
from annoray.plot._dataarray import _broadcast, _concise_dates, _facets, _scatter_core

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from annoray._annoray import DataArray, Dataset


def _arrays(ds: Dataset, names: Sequence[str], dims: Sequence[str] | None = None) -> list:
    """The named variables/coordinates broadcast to their union of dimensions (in order of
    first appearance, or ``dims``)."""
    arrays: list[DataArray] = [u.coord(ds, n) for n in names]
    if dims is None:
        dims = list(dict.fromkeys(d for a in arrays for d in a.dims))
    shape = [ds.sizes[d] for d in dims]
    return [_broadcast(a, dims, shape) for a in arrays]


def _check(ds: Dataset, **names: str | None) -> None:
    valid = set(ds.data_vars) | set(ds.coords) | set(ds.dims)
    for what, name in names.items():
        if name is not None and name not in valid:
            raise ValueError(f"{what} must be a variable or coordinate, not {name!r}")


def _facet(ds: Dataset, func: Any, row: str | None, col: str | None, kw: dict) -> Any:
    return _facets(
        ds,
        kind="dataset",
        plotfunc=func,
        row=row,
        col=col,
        col_wrap=kw.pop("col_wrap", None),
        figsize=kw.pop("figsize", None),
        size=kw.pop("size", None),
        aspect=kw.pop("aspect", None),
        ax=kw.pop("ax", None),
        subplot_kws=kw.pop("subplot_kws", None),
        **kw,
    )


def scatter(
    ds: Dataset,
    *args: Any,
    x: str,
    y: str,
    hue: str | None = None,
    markersize: str | None = None,
    row: str | None = None,
    col: str | None = None,
    ax: Axes | None = None,
    figsize: Sequence[float] | None = None,
    size: float | None = None,
    aspect: float | None = None,
    add_legend: bool | None = None,
    add_colorbar: bool | None = None,
    add_labels: bool = True,
    cmap: Any = None,
    vmin: Any = None,
    vmax: Any = None,
    norm: Any = None,
    xincrease: bool | None = None,
    yincrease: bool | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xticks: Any = None,
    yticks: Any = None,
    xlim: Any = None,
    ylim: Any = None,
    **kwargs: Any,
) -> Any:
    """Scatter of variable ``y`` against ``x``, coloured by ``hue`` and sized by
    ``markersize`` (variables or coordinates)."""
    _check(ds, x=x, y=y, hue=hue, markersize=markersize)
    if row or col:
        kw = dict(kwargs, figsize=figsize, size=size, aspect=aspect, ax=ax)
        return _facet(
            ds,
            scatter,
            row,
            col,
            dict(
                kw,
                x=x,
                y=y,
                hue=hue,
                markersize=markersize,
                add_legend=add_legend,
                add_colorbar=add_colorbar,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
            ),
        )
    names = [x, y] + [n for n in (hue, markersize) if n is not None]
    values = [np.ravel(a) for a in _arrays(ds, names)]
    xv, yv = values[0], values[1]
    hue_values = values[2] if hue is not None else None
    sizes = values[-1] if markersize is not None else None
    ax = u.get_axis(figsize, size, aspect, ax)
    hue_label = u.label(u.coord(ds, hue)) if hue is not None else ""
    primitive = _scatter_core(
        ax,
        xv,
        yv,
        args,
        hue_values,
        hue_label,
        sizes,
        add_legend,
        add_colorbar,
        cmap,
        vmin,
        vmax,
        norm,
        kwargs,
    )
    if add_labels:
        ax.set_xlabel(u.label(u.coord(ds, x)))
        ax.set_ylabel(u.label(u.coord(ds, y)))
        ax.set_title(u.title_for_slice(ds))
    if u.is_datetime(xv):
        _concise_dates(ax, "x")
    u.update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)
    return primitive


def _nice_magnitude(uv: np.ndarray, vv: np.ndarray) -> float:
    import matplotlib.ticker as mticker

    mean = float(np.nanmean(np.hypot(uv, vv)))
    return float(mticker.MaxNLocator(3).tick_values(0, mean)[-2])


def _hue_norm(values: np.ndarray, cmap: Any, vmin: Any, vmax: Any, norm: Any) -> dict:
    import matplotlib.colors as mcolors

    params = u.cmap_params(values, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
    if params["norm"] is None:
        params["norm"] = mcolors.Normalize(params["vmin"], params["vmax"])
    return {"cmap": params["cmap"], "norm": params["norm"], "extend": params["extend"]}


def _vector_plot(
    kind: str,
    ds: Dataset,
    x: str | None,
    y: str | None,
    u_: str | None,
    v_: str | None,
    hue: str | None,
    row: str | None,
    col: str | None,
    ax: Axes | None,
    add_guide: bool | None,
    cmap: Any,
    vmin: Any,
    vmax: Any,
    norm: Any,
    cbar_kwargs: Mapping[str, Any] | None,
    cbar_ax: Axes | None,
    kwargs: dict[str, Any],
) -> Any:
    if x is None or y is None or u_ is None or v_ is None:
        raise ValueError(f"Must specify x, y, u, v for {kind} plots.")
    _check(ds, x=x, y=y, u=u_, v=v_, hue=hue)
    facet = kwargs.pop("_facet", False)
    if row or col:
        func = quiver if kind == "quiver" else streamplot
        return _facet(
            ds,
            func,
            row,
            col,
            dict(
                kwargs,
                ax=ax,
                x=x,
                y=y,
                u=u_,
                v=v_,
                hue=hue,
                add_guide=add_guide,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
            ),
        )
    dims = None
    if kind == "streamplot":
        # x may not vary along rows nor y along columns: order the grid (y, x).
        xd = u.coord(ds, x).dims
        yd = u.coord(ds, y).dims
        xdim = xd[0] if len(xd) == 1 else None
        ydim = yd[0] if len(yd) == 1 else None
        if xdim is not None and ydim is None and len(set(yd) - {xdim}) == 1:
            ydim = next(iter(set(yd) - {xdim}))
        if ydim is not None and xdim is None and len(set(xd) - {ydim}) == 1:
            xdim = next(iter(set(xd) - {ydim}))
        if xdim is not None and ydim is not None:
            dims = [ydim, xdim]
    names = [x, y, u_, v_] + ([hue] if hue is not None else [])
    arrays = _arrays(ds, names, dims)
    ax = u.get_axis(
        kwargs.pop("figsize", None), kwargs.pop("size", None), kwargs.pop("aspect", None), ax
    )
    hue_params = None
    if hue is not None:
        arrays[4] = u.as_colour_values(np.asarray(arrays[4]))
        hue_params = _hue_norm(arrays[4], cmap, vmin, vmax, norm)
        kwargs.update(cmap=hue_params["cmap"], norm=hue_params["norm"])
    if kind == "quiver":
        kwargs.setdefault("pivot", "middle")
        primitive = ax.quiver(*arrays, **kwargs)
    else:
        if hue is not None:
            kwargs["color"] = arrays[4]
        primitive = ax.streamplot(*arrays[:4], **kwargs).lines
    if facet:
        return primitive
    ax.set_xlabel(u.label(u.coord(ds, x)))
    ax.set_ylabel(u.label(u.coord(ds, y)))
    ax.set_title(u.title_for_slice(ds))
    if hue is not None and (add_guide is None or add_guide):
        assert hue_params is not None
        cbar_kw = dict(cbar_kwargs or {})
        cbar_kw.setdefault("label", u.label(u.coord(ds, hue)))
        u.add_colorbar(primitive, ax, cbar_ax, cbar_kw, hue_params)
    if kind == "quiver" and (add_guide is None or add_guide):
        from matplotlib.quiver import Quiver

        assert isinstance(primitive, Quiver)
        mag = _nice_magnitude(np.asarray(arrays[2]), np.asarray(arrays[3]))
        unit = u.units_of(u.coord(ds, u_)) or ""
        ax.quiverkey(
            primitive,
            X=0.85,
            Y=0.9,
            U=mag,
            label=f"{mag}\n{unit}",
            labelpos="E",
            coordinates="figure",
        )
    return primitive


def quiver(
    ds: Dataset,
    *,
    x: str | None = None,
    y: str | None = None,
    u: str | None = None,
    v: str | None = None,
    hue: str | None = None,
    row: str | None = None,
    col: str | None = None,
    ax: Axes | None = None,
    add_guide: bool | None = None,
    cmap: Any = None,
    vmin: Any = None,
    vmax: Any = None,
    norm: Any = None,
    cbar_kwargs: Mapping[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    **kwargs: Any,
) -> Any:
    """Arrows ``(u, v)`` at ``(x, y)`` (:meth:`matplotlib.axes.Axes.quiver`), coloured by
    ``hue``, with a quiver key in the units of ``u``."""
    return _vector_plot(
        "quiver",
        ds,
        x,
        y,
        u,
        v,
        hue,
        row,
        col,
        ax,
        add_guide,
        cmap,
        vmin,
        vmax,
        norm,
        cbar_kwargs,
        cbar_ax,
        kwargs,
    )


def streamplot(
    ds: Dataset,
    *,
    x: str | None = None,
    y: str | None = None,
    u: str | None = None,
    v: str | None = None,
    hue: str | None = None,
    row: str | None = None,
    col: str | None = None,
    ax: Axes | None = None,
    add_guide: bool | None = None,
    cmap: Any = None,
    vmin: Any = None,
    vmax: Any = None,
    norm: Any = None,
    cbar_kwargs: Mapping[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    **kwargs: Any,
) -> Any:
    """Streamlines of ``(u, v)`` over the ``(x, y)`` grid
    (:meth:`matplotlib.axes.Axes.streamplot`), coloured by ``hue``."""
    return _vector_plot(
        "streamplot",
        ds,
        x,
        y,
        u,
        v,
        hue,
        row,
        col,
        ax,
        add_guide,
        cmap,
        vmin,
        vmax,
        norm,
        cbar_kwargs,
        cbar_ax,
        kwargs,
    )


__all__ = ["quiver", "scatter", "streamplot"]
