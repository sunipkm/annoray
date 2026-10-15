"""Plots of a DataArray: ``plot`` (dispatch by rank), ``line``, ``step``, ``hist``,
``scatter``, and the 2-D ``pcolormesh``, ``imshow``, ``contour``, ``contourf``, ``surface``.

Signatures and semantics mirror ``xarray.plot``; ``row``/``col`` facet into a
:class:`~annoray.plot.FacetGrid`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from annoray.plot import _utils as u

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize

    from annoray._annoray import DataArray
    from annoray.plot._facetgrid import FacetGrid


def _concise_dates(ax: Axes, axis: str = "x") -> None:
    import matplotlib.dates as mdates

    locator = mdates.AutoDateLocator()
    target = getattr(ax, f"{axis}axis")
    target.set_major_locator(locator)
    target.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _values(da: DataArray) -> np.ndarray:
    return np.asarray(da.values)


def _broadcast(c: DataArray, dims: Sequence[str], shape: Sequence[int]) -> np.ndarray:
    """A coordinate's values broadcast to an array of ``dims``/``shape``."""
    own = [d for d in dims if d in c.dims]
    v = np.asarray(c.transpose(*own).values) if list(c.dims) != own else _values(c)
    index = tuple(slice(None) if d in c.dims else np.newaxis for d in dims)
    return np.broadcast_to(v[index], tuple(shape))


def _facets(
    data: Any,
    *,
    kind: str,
    plotfunc: Callable[..., Any],
    row: str | None,
    col: str | None,
    col_wrap: int | str | None = None,
    sharex: bool = True,
    sharey: bool = True,
    figsize: Sequence[float] | None = None,
    size: float | None = None,
    aspect: float | None = None,
    ax: Axes | None = None,
    subplot_kws: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> FacetGrid:
    from annoray.plot._facetgrid import FacetGrid

    if ax is not None:
        raise ValueError("Can't use axes when making faceted plots.")
    if aspect is None:
        aspect = 1
    if size is None:
        size = 3
    elif figsize is not None:
        raise ValueError("cannot provide both `figsize` and `size` arguments")
    g = FacetGrid(
        data,
        col=col,
        row=row,
        col_wrap=col_wrap,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        aspect=aspect,
        size=size,
        subplot_kws=subplot_kws,
    )
    if kind == "line":
        return g.map_dataarray_line(plotfunc, **kwargs)
    if kind == "dataarray":
        return g.map_dataarray(plotfunc, **kwargs)
    if kind == "plot1d":
        return g.map_plot1d(plotfunc, **kwargs)
    if kind == "dataset":
        return g.map_dataset(plotfunc, **kwargs)
    raise ValueError(f"unknown facet kind {kind!r}")


# ----- dispatch -----


def plot(
    darray: DataArray,
    *,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | str | None = None,
    ax: Axes | None = None,
    hue: str | None = None,
    subplot_kws: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Default plot by rank after dropping length-1 dimensions: 1-D :func:`line`, 2-D
    :func:`pcolormesh` (:func:`line` with ``hue``), anything else :func:`hist`. ``row`` and
    ``col`` facet over further dimensions."""
    darray = u.squeezed(darray, keep=(row, col, hue))
    plot_dims = set(darray.dims) - {row, col, hue}
    ndims = len(plot_dims)
    if ndims == 0 or darray.size == 0:
        raise TypeError("No numeric data to plot.")
    func: Callable[..., Any]
    if ndims in (1, 2):
        if row or col:
            kwargs["subplot_kws"] = subplot_kws
            kwargs["row"] = row
            kwargs["col"] = col
            kwargs["col_wrap"] = col_wrap
        if ndims == 1 or hue:
            func = line
            kwargs["hue"] = hue
        else:
            func = pcolormesh
            kwargs["subplot_kws"] = subplot_kws
    else:
        if row or col or hue:
            raise ValueError(
                "Only 1d and 2d plots are supported for facets in annoray. "
                "See the package `Seaborn` for more options."
            )
        func = hist
    kwargs["ax"] = ax
    return func(darray, **kwargs)


# ----- 1-D -----


def _line_data(
    da: DataArray, x: str | None, y: str | None, hue: str | None
) -> tuple[DataArray, DataArray, np.ndarray, np.ndarray, DataArray | None, str]:
    """``(xplt, yplt, xvalues, yvalues, hue coordinate, hue label)``; for 2-D data the
    values are columns per hue."""
    if x is not None and y is not None:
        raise ValueError("Cannot specify both x and y kwargs for line plots.")
    u.check_name(da, x, "x")
    u.check_name(da, y, "y")
    if da.ndim == 1:
        if x is not None:
            xplt, yplt = u.coord(da, x), da
        elif y is not None:
            xplt, yplt = da, u.coord(da, y)
        else:
            xplt, yplt = u.coord(da, da.dims[0]), da
        return xplt, yplt, _values(xplt), _values(yplt), None, ""

    if x is None and y is None and hue is None:
        raise ValueError("For 2D inputs, please specify either hue, x or y.")
    if hue is not None:
        u.check_name(da, hue, "hue")
    if y is None:
        xname, huename = u.infer_xy(da, x, hue)
        xplt = u.coord(da, xname)
        hplt = u.coord(da, huename)
        if xplt.ndim > 1:
            if huename not in da.dims:
                raise ValueError(f"For 2D inputs, hue must be a dimension i.e. one of {da.dims!r}")
            other = next(d for d in da.dims if d != huename)
            dims = (other, huename)
            xv = _broadcast(xplt, dims, [da.sizes[d] for d in dims])
            yv = _values(da.transpose(*dims))
        else:
            dims = (xplt.dims[0], hplt.dims[0])
            xv, yv = _values(xplt), _values(da.transpose(*dims))
        return xplt, da, xv, yv, hplt, u.label(hplt)
    yname, huename = u.infer_xy(da, y, hue)
    yplt = u.coord(da, yname)
    hplt = u.coord(da, huename)
    if yplt.ndim > 1:
        if huename not in da.dims:
            raise ValueError(f"For 2D inputs, hue must be a dimension i.e. one of {da.dims!r}")
        other = next(d for d in da.dims if d != huename)
        dims = (other, huename)
        yv = _broadcast(yplt, dims, [da.sizes[d] for d in dims])
        xv = _values(da.transpose(*dims))
    else:
        dims = (yplt.dims[0], hplt.dims[0])
        xv, yv = _values(da.transpose(*dims)), _values(yplt)
    return da, yplt, xv, yv, hplt, u.label(hplt)


def line(
    darray: DataArray,
    *args: Any,
    row: str | None = None,
    col: str | None = None,
    figsize: Sequence[float] | None = None,
    aspect: float | None = None,
    size: float | None = None,
    ax: Axes | None = None,
    hue: str | None = None,
    x: str | None = None,
    y: str | None = None,
    xincrease: bool | None = None,
    yincrease: bool | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xticks: Any = None,
    yticks: Any = None,
    xlim: Any = None,
    ylim: Any = None,
    add_legend: bool = True,
    _labels: bool = True,
    **kwargs: Any,
) -> Any:
    """Line plot of a 1-D array, or one line per ``hue`` value of a 2-D array
    (:meth:`matplotlib.axes.Axes.plot`). ``x`` or ``y`` names the coordinate on that axis.
    Returns the ``Line2D`` list, or a :class:`FacetGrid` with ``row``/``col``."""
    if row or col:
        return _facets(
            darray,
            kind="line",
            plotfunc=line,
            row=row,
            col=col,
            col_wrap=kwargs.pop("col_wrap", None),
            figsize=figsize,
            size=size,
            aspect=aspect,
            ax=ax,
            subplot_kws=kwargs.pop("subplot_kws", None),
            x=x,
            y=y,
            hue=hue,
            xincrease=xincrease,
            yincrease=yincrease,
            xscale=xscale,
            yscale=yscale,
            xticks=xticks,
            yticks=yticks,
            xlim=xlim,
            ylim=ylim,
            add_legend=add_legend,
            args=args,
            **kwargs,
        )
    kwargs.pop("subplot_kws", None)
    kwargs.pop("col_wrap", None)
    if darray.ndim == 0 or darray.size == 0:
        raise TypeError("No numeric data to plot.")
    if darray.ndim > 2:
        raise ValueError(
            "Line plots are for 1- or 2-dimensional DataArrays. "
            f"Passed DataArray has {darray.ndim} dimensions"
        )
    if not args:
        args = kwargs.pop("args", ())
    ax = u.get_axis(figsize, size, aspect, ax)
    xplt, yplt, xv, yv, hplt, hue_label = _line_data(darray, x, y, hue)
    primitive = ax.plot(xv, yv, *args, **kwargs)
    if _labels:
        ax.set_xlabel(u.label(xplt))
        ax.set_ylabel(u.label(yplt))
        ax.set_title(u.title_for_slice(darray))
    if darray.ndim == 2 and add_legend:
        assert hplt is not None
        ax.legend(handles=primitive, labels=list(_values(hplt)), title=hue_label)
    if u.is_datetime(np.asarray(xv)):
        _concise_dates(ax, "x")
    u.update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)
    return primitive


def step(
    darray: DataArray,
    *args: Any,
    where: str = "pre",
    drawstyle: str | None = None,
    ds: str | None = None,
    row: str | None = None,
    col: str | None = None,
    **kwargs: Any,
) -> Any:
    """Step plot: :func:`line` with ``drawstyle="steps-<where>"`` (``where`` is ``"pre"``,
    ``"post"`` or ``"mid"``)."""
    if where not in {"pre", "post", "mid"}:
        raise ValueError("'where' argument to step must be 'pre', 'post' or 'mid'")
    if ds is not None:
        if drawstyle is not None:
            raise TypeError("ds and drawstyle are mutually exclusive")
        drawstyle = ds
    return line(
        darray, *args, drawstyle="steps-" + where + (drawstyle or ""), row=row, col=col, **kwargs
    )


def hist(
    darray: DataArray,
    *,
    figsize: Sequence[float] | None = None,
    size: float | None = None,
    aspect: float | None = None,
    ax: Axes | None = None,
    xincrease: bool | None = None,
    yincrease: bool | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xticks: Any = None,
    yticks: Any = None,
    xlim: Any = None,
    ylim: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Histogram of all non-missing values, any rank (:meth:`matplotlib.axes.Axes.hist`)."""
    if darray.ndim == 0 or darray.size == 0:
        raise TypeError("No numeric data to plot.")
    ax = u.get_axis(figsize, size, aspect, ax)
    v = np.ravel(_values(darray))
    if v.dtype.kind in "fc":
        v = v[~np.isnan(v)]
    elif v.dtype.kind in "mM":
        v = v[~np.isnat(v)]
    n, bins, patches = ax.hist(v, **kwargs)
    ax.set_title(u.title_for_slice(darray))
    ax.set_xlabel(u.label(darray))
    u.update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)
    return n, bins, patches


def scatter(
    darray: DataArray,
    *args: Any,
    x: str | None = None,
    hue: str | None = None,
    markersize: str | None = None,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | str | None = None,
    figsize: Sequence[float] | None = None,
    size: float | None = None,
    aspect: float | None = None,
    ax: Axes | None = None,
    add_legend: bool | None = None,
    add_colorbar: bool | None = None,
    add_labels: bool = True,
    cmap: str | Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    xincrease: bool | None = None,
    yincrease: bool | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xticks: Any = None,
    yticks: Any = None,
    xlim: Any = None,
    ylim: Any = None,
    subplot_kws: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Scatter of the values against coordinate ``x`` (default: the first dimension),
    coloured by ``hue`` (colorbar for numeric, legend for categorical) and sized by
    ``markersize`` (:meth:`matplotlib.axes.Axes.scatter`)."""
    if row or col:
        return _facets(
            darray,
            kind="plot1d",
            plotfunc=scatter,
            row=row,
            col=col,
            col_wrap=col_wrap,
            figsize=figsize,
            size=size,
            aspect=aspect,
            ax=ax,
            subplot_kws=subplot_kws,
            x=x,
            hue=hue,
            markersize=markersize,
            add_legend=add_legend,
            add_colorbar=add_colorbar,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            **kwargs,
        )
    if darray.size == 0:
        raise TypeError("No numeric data to plot.")
    xname = x if x is not None else darray.dims[0]
    u.check_name(darray, xname, "x")
    u.check_name(darray, hue, "hue")
    u.check_name(darray, markersize, "markersize")
    ax = u.get_axis(figsize, size, aspect, ax, **dict(subplot_kws or {}))
    dims, shape = darray.dims, darray.shape
    xc = u.coord(darray, xname)
    xv = np.ravel(_broadcast(xc, dims, shape))
    yv = np.ravel(_values(darray))
    hue_values = None
    if hue is not None:
        hue_values = np.ravel(_broadcast(u.coord(darray, hue), dims, shape))
    sizes = None
    if markersize is not None:
        sizes = np.ravel(_broadcast(u.coord(darray, markersize), dims, shape))
    hue_label = u.label(u.coord(darray, hue)) if hue is not None else ""
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
        ax.set_xlabel(u.label(xc))
        ax.set_ylabel(u.label(darray))
        ax.set_title(u.title_for_slice(darray))
    if u.is_datetime(xv):
        _concise_dates(ax, "x")
    u.update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)
    return primitive


def _marker_sizes(s: np.ndarray) -> np.ndarray:
    """Marker areas spanning 18 to 72 points² over the range of ``s`` (as seaborn)."""
    s = s.astype(float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    return 18.0 + (s - lo) / ((hi - lo) or 1.0) * (72.0 - 18.0)


def _scatter_core(
    ax: Axes,
    xv: np.ndarray,
    yv: np.ndarray,
    args: Sequence[Any],
    hue_values: np.ndarray | None,
    hue_label: str,
    sizes: np.ndarray | None,
    add_legend: bool | None,
    add_colorbar: bool | None,
    cmap: Any,
    vmin: Any,
    vmax: Any,
    norm: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Scatter coloured by ``hue_values``: a colormap and colorbar when numeric, one labelled
    series per value (and a legend) otherwise."""
    kwargs = dict(kwargs)
    if sizes is not None:
        kwargs["s"] = _marker_sizes(sizes)
    if hue_values is None:
        return ax.scatter(xv, yv, *args, **kwargs)
    if hue_values.dtype.kind in "iufcmM":
        params = u.cmap_params(hue_values, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
        primitive = ax.scatter(
            xv,
            yv,
            *args,
            c=u.as_colour_values(hue_values),
            cmap=params["cmap"],
            vmin=params["vmin"],
            vmax=params["vmax"],
            norm=params["norm"],
            **kwargs,
        )
        if add_colorbar is None or add_colorbar:
            fig = ax.get_figure()
            assert fig is not None
            u.date_colorbar(fig.colorbar(primitive, ax=ax, label=hue_label), hue_values)
        return primitive
    primitive = []
    for value in dict.fromkeys(hue_values.tolist()):
        mask = hue_values == value
        kw = dict(kwargs)
        s = kw.get("s")
        if s is not None and np.ndim(s) == 1:
            kw["s"] = s[mask]
        primitive.append(ax.scatter(xv[mask], yv[mask], *args, label=str(value), **kw))
    if add_legend is None or add_legend:
        ax.legend(title=hue_label)
    return primitive


# ----- 2-D -----


def _rescale_rgb(z: np.ndarray, vmin: Any, vmax: Any, robust: bool) -> np.ndarray:
    if robust:
        if vmax is None:
            vmax = np.nanpercentile(z, 100 - u.ROBUST_PERCENTILE)
        if vmin is None:
            vmin = np.nanpercentile(z, u.ROBUST_PERCENTILE)
    elif vmax is None:
        vmax = 255 if np.issubdtype(z.dtype, np.integer) else 1
        if vmax < vmin:
            raise ValueError(f"vmin={vmin!r} is above the default vmax ({vmax!r}); give vmax")
    elif vmin is None:
        vmin = 0
        if vmin > vmax:
            raise ValueError(f"vmax={vmax!r} is below the default vmin (0); give vmin")
    z = ((z.astype("f8") - vmin) / (vmax - vmin)).astype("f4")
    return np.minimum(np.maximum(z, 0), 1)


def _pcolormesh(
    x: np.ndarray,
    y: np.ndarray,
    z: Any,
    ax: Axes,
    xscale: str | None = None,
    yscale: str | None = None,
    infer_intervals: bool | None = None,
    **kwargs: Any,
) -> Any:
    x = np.asarray(x)
    y = np.asarray(y)
    if infer_intervals is None:
        infer_intervals = not hasattr(ax, "projection") or x.ndim == 1
    if any(np.issubdtype(k.dtype, np.str_) for k in (x, y)):
        infer_intervals = False
    zshape = np.shape(z)
    if infer_intervals and (x.shape[0] == zshape[1] or (x.ndim > 1 and x.shape[1] == zshape[1])):
        if x.ndim == 1:
            x = u.interval_breaks(x, check_monotonic=True, scale=xscale)
        else:
            x = u.interval_breaks(u.interval_breaks(x, axis=1, scale=xscale), scale=xscale)
    if infer_intervals and y.shape[0] == zshape[0]:
        if y.ndim == 1:
            y = u.interval_breaks(y, check_monotonic=True, scale=yscale)
        else:
            y = u.interval_breaks(u.interval_breaks(y, axis=1, scale=yscale), scale=yscale)
    ax.grid(False)
    primitive = ax.pcolormesh(x, y, z, **kwargs)
    if not hasattr(ax, "projection") and x.ndim == 1 and y.ndim == 1:
        # pcolormesh rounds the limits outwards; fit them to the cells instead.
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
    return primitive


def _imshow(x: np.ndarray, y: np.ndarray, z: Any, ax: Axes, **kwargs: Any) -> Any:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("imshow requires 1D coordinates, try using pcolormesh or contour(f)")

    def centred(v: np.ndarray) -> tuple[Any, Any]:
        if np.issubdtype(v.dtype, np.str_):
            return -0.5, len(v) - 0.5
        step = 0.5 * (v[1] - v[0]) if len(v) > 1 else 0.1
        return v[0] - step, v[-1] + step

    left, right = centred(x)
    top, bottom = centred(y)
    opts: dict[str, Any] = {"origin": "upper", "interpolation": "nearest"}
    if not hasattr(ax, "projection"):
        opts["aspect"] = "auto"
    opts.update(kwargs)
    opts["extent"] = (
        [left, right, bottom, top] if opts["origin"] == "upper" else [left, right, top, bottom]
    )
    if z.ndim == 3:
        # Missing data is transparent: add an alpha channel and clear it where masked.
        if z.shape[-1] == 3:
            alpha = np.ma.ones(z.shape[:2] + (1,), dtype=np.promote_types(z.dtype, np.uint8))
            if np.issubdtype(z.dtype, np.integer):
                alpha[:] = 255
            z = np.ma.concatenate((z, alpha), axis=2)
        else:
            z = z.copy()
        z[np.any(np.ma.getmaskarray(z), axis=-1), -1] = 0
    primitive = ax.imshow(z, **opts)
    for axis, v in (("x", x), ("y", y)):
        if np.issubdtype(v.dtype, np.str_):
            getattr(ax, f"set_{axis}ticks")(np.arange(len(v)))
            getattr(ax, f"set_{axis}ticklabels")(v)
    return primitive


def _contour(x: Any, y: Any, z: Any, ax: Axes, **kwargs: Any) -> Any:
    return ax.contour(x, y, z, **kwargs)


def _contourf(x: Any, y: Any, z: Any, ax: Axes, **kwargs: Any) -> Any:
    return ax.contourf(x, y, z, **kwargs)


def _surface(x: Any, y: Any, z: Any, ax: Axes, **kwargs: Any) -> Any:
    return ax.plot_surface(x, y, z, **kwargs)  # type: ignore[attr-defined]


_KINDS: dict[str, Callable[..., Any]] = {
    "pcolormesh": _pcolormesh,
    "imshow": _imshow,
    "contour": _contour,
    "contourf": _contourf,
    "surface": _surface,
}


def _plot2d(
    kind: str,
    darray: DataArray,
    *,
    x: str | None = None,
    y: str | None = None,
    figsize: Sequence[float] | None = None,
    size: float | None = None,
    aspect: float | None = None,
    ax: Axes | None = None,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | str | None = None,
    xincrease: bool | None = True,
    yincrease: bool | None = True,
    add_colorbar: bool | None = None,
    add_labels: bool = True,
    vmin: Any = None,
    vmax: Any = None,
    cmap: str | Colormap | None = None,
    center: Any = None,
    robust: bool = False,
    extend: str | None = None,
    levels: Any = None,
    infer_intervals: bool | None = None,
    colors: Any = None,
    subplot_kws: Mapping[str, Any] | None = None,
    cbar_ax: Axes | None = None,
    cbar_kwargs: Mapping[str, Any] | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xticks: Any = None,
    yticks: Any = None,
    xlim: Any = None,
    ylim: Any = None,
    norm: Normalize | None = None,
    _facet_params: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    if add_colorbar is None:
        add_colorbar = not (kind == "contour" or (kind == "surface" and cmap is None))
    rgb_dims = 3 + (row is not None) + (col is not None)
    imshow_rgb = kind == "imshow" and darray.ndim == rgb_dims
    if imshow_rgb:
        add_colorbar = False
    subplot_kws = dict(subplot_kws or {})
    sharex = sharey = True
    if kind == "surface" and _facet_params is None:
        if ax is None:
            subplot_kws["projection"] = "3d"
        sharex = sharey = False

    if row or col:
        return _facets(
            darray,
            kind="dataarray",
            plotfunc=globals()[kind],
            row=row,
            col=col,
            col_wrap=col_wrap,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            size=size,
            aspect=aspect,
            ax=ax,
            subplot_kws=subplot_kws,
            x=x,
            y=y,
            xincrease=xincrease,
            yincrease=yincrease,
            add_colorbar=add_colorbar,
            add_labels=add_labels,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            center=center,
            robust=robust,
            extend=extend,
            levels=levels,
            infer_intervals=infer_intervals,
            colors=colors,
            cbar_ax=cbar_ax,
            cbar_kwargs=cbar_kwargs,
            xscale=xscale,
            yscale=yscale,
            xticks=xticks,
            yticks=yticks,
            xlim=xlim,
            ylim=ylim,
            norm=norm,
            **kwargs,
        )
    if darray.ndim == 0 or darray.size == 0:
        raise TypeError("No numeric data to plot.")
    if kind == "surface" and _facet_params is None and ax is not None:
        from mpl_toolkits.mplot3d import Axes3D

        if not isinstance(ax, Axes3D):
            raise ValueError(
                'If ax is passed to surface(), it must be created with projection="3d"'
            )

    rgb = kwargs.pop("rgb", None)
    if rgb is not None and kind != "imshow":
        raise ValueError('The "rgb" keyword is only valid for imshow()')
    if rgb is not None and not imshow_rgb:
        raise ValueError(
            'The "rgb" keyword is only valid for imshow() with a three-dimensional array '
            "(per facet)"
        )
    if imshow_rgb:
        xlab, ylab, rgb = _infer_xy_rgb(darray, x, y, rgb)
    else:
        xlab, ylab = u.infer_xy(darray, x, y)
    xc, yc = u.coord(darray, xlab), u.coord(darray, ylab)

    if imshow_rgb:
        assert rgb is not None
        dims: tuple[str, ...] = (ylab, xlab, rgb)
        xv, yv = _values(xc), _values(yc)
    elif xc.ndim > 1 or yc.ndim > 1 or kind == "surface":
        dims = tuple(darray.dims)
        shape = darray.shape
        xv, yv = _broadcast(xc, dims, shape), _broadcast(yc, dims, shape)
    else:
        dims = (yc.dims[0], xc.dims[0])
        xv, yv = _values(xc), _values(yc)
    z = _values(darray.transpose(*dims)) if tuple(darray.dims) != dims else _values(darray)
    if imshow_rgb and (robust or vmin is not None or vmax is not None):
        z = _rescale_rgb(z, vmin, vmax, robust)
        vmin = vmax = None
        robust = False
    zval = np.ma.masked_invalid(z) if z.dtype.kind in "fc" else np.ma.asarray(z)

    cbar_kw = dict(cbar_kwargs or {})
    if kind == "surface":
        params = {
            "vmin": vmin,
            "vmax": vmax,
            "cmap": cmap,
            "extend": extend,
            "levels": levels,
            "norm": norm,
        }
    else:
        if cmap and colors:
            raise ValueError("Can't specify both cmap and colors.")
        if colors and "contour" not in kind and levels is None:
            raise ValueError("Can only specify colors with contour or levels")
        if isinstance(cmap, list | tuple):
            raise ValueError("Give a list of colors as `colors`, not `cmap`.")
        if _facet_params is not None:
            params = dict(_facet_params)
        elif imshow_rgb:
            params = {
                "vmin": None,
                "vmax": None,
                "cmap": None,
                "extend": None,
                "levels": None,
                "norm": None,
            }
        else:
            params = u.cmap_params(
                zval.filled(np.nan) if zval.dtype.kind in "fc" else zval.data,
                vmin=vmin,
                vmax=vmax,
                cmap=colors or cmap,
                center=center,
                robust=robust,
                extend=extend,
                levels=levels,
                filled=kind != "contour",
                norm=norm,
            )

    if "contour" in kind:
        kwargs["extend"] = params["extend"]
        kwargs["levels"] = params["levels"]
        if colors is not None:  # a single colour keeps matplotlib's dashed negatives
            params["cmap"] = None
            kwargs["colors"] = colors
    if kind == "pcolormesh":
        kwargs["infer_intervals"] = infer_intervals
        kwargs["xscale"] = xscale
        kwargs["yscale"] = yscale
    if kind == "imshow" and isinstance(aspect, str):
        raise ValueError("matplotlib's string `aspect` is not available; give a number")

    if _facet_params is None:
        ax = u.get_axis(figsize, size, aspect, ax, **subplot_kws)
    assert ax is not None
    primitive = _KINDS[kind](
        xv,
        yv,
        zval,
        ax=ax,
        cmap=params["cmap"],
        vmin=params["vmin"],
        vmax=params["vmax"],
        norm=params["norm"],
        **kwargs,
    )
    if add_labels:
        ax.set_xlabel(u.label(xc))
        ax.set_ylabel(u.label(yc))
        ax.set_title(u.title_for_slice(darray))
        if kind == "surface":
            ax.set_zlabel(u.label(darray))  # type: ignore[attr-defined]
    if add_colorbar:
        if add_labels and "label" not in cbar_kw:
            cbar_kw["label"] = u.label(darray)
        u.add_colorbar(primitive, ax, cbar_ax, cbar_kw, params)
    elif cbar_ax is not None or cbar_kwargs:
        raise ValueError("cbar_ax and cbar_kwargs can't be used with add_colorbar=False.")
    if "origin" in kwargs:  # origin overrides yincrease
        yincrease = None
    u.update_axes(ax, xincrease, yincrease, xscale, yscale, xticks, yticks, xlim, ylim)
    if u.is_datetime(np.asarray(xv)):
        _concise_dates(ax, "x")
    return primitive


def _infer_xy_rgb(
    da: DataArray, x: str | None, y: str | None, rgb: str | None
) -> tuple[str, str, str]:
    """x, y and the colour dimension of a 3-D array for imshow."""
    if rgb is not None and rgb not in da.dims:
        raise ValueError(f"rgb must be one of {da.dims!r}")
    for name, value in (("x", x), ("y", y)):
        if value is not None and value not in da.dims:
            raise ValueError(f"{name} must be a dimension of a 3-D array for imshow")
    if rgb is None:
        candidates = [d for d in da.dims if d not in (x, y) and da.sizes[d] in (3, 4)]
        if not candidates:
            raise ValueError(
                "A 3-dimensional array was passed to imshow(), but there is no dimension "
                "that could be color. At least one dimension must be of size 3 (RGB) or 4 "
                "(RGBA), and not given as x or y."
            )
        rgb = candidates[-1]
    rest = [d for d in da.dims if d != rgb]
    if x is None and y is None:
        y, x = rest
    elif x is None:
        x = next(d for d in rest if d != y)
    elif y is None:
        y = next(d for d in rest if d != x)
    assert x is not None and y is not None
    return x, y, rgb


def _make2d(kind: str, doc: str) -> Callable[..., Any]:
    def f(darray: DataArray, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError(f"{kind}() takes x and y as keyword arguments")
        return _plot2d(kind, darray, **kwargs)

    f.__name__ = f.__qualname__ = kind
    f.__doc__ = doc + _DOC2D
    return f


_DOC2D = """

Keyword arguments (shared by all 2-D plots, as in ``xarray.plot``): ``x``, ``y``
(coordinate names; default: the last and first dimensions), ``row``, ``col``, ``col_wrap``
(facets), ``figsize``/``size``/``aspect``/``ax``, ``xincrease``/``yincrease``,
``add_colorbar``, ``add_labels``, ``vmin``/``vmax``, ``cmap``, ``center`` (diverging map
around this value; ``False`` disables), ``robust`` (2nd/98th percentile limits), ``extend``,
``levels`` (int: nice levels; sequence: boundaries), ``colors``, ``norm``, ``cbar_ax``,
``cbar_kwargs``, ``subplot_kws``, ``xscale``/``yscale``, ``xticks``/``yticks``,
``xlim``/``ylim``; anything else goes to matplotlib. Returns the matplotlib artist, or a
:class:`FacetGrid` with ``row``/``col``.
"""

pcolormesh = _make2d(
    "pcolormesh",
    "Pseudocolour plot (:meth:`matplotlib.axes.Axes.pcolormesh`); cell edges are inferred "
    "from the coordinates (``infer_intervals``).",
)
imshow = _make2d(
    "imshow",
    "Image plot (:meth:`matplotlib.axes.Axes.imshow`), pixels centred on evenly spaced 1-D "
    "coordinates; a 3-D array with a length-3/4 dimension (``rgb``) is an RGB(A) image.",
)
contour = _make2d("contour", "Contour lines (:meth:`matplotlib.axes.Axes.contour`).")
contourf = _make2d("contourf", "Filled contours (:meth:`matplotlib.axes.Axes.contourf`).")
surface = _make2d(
    "surface",
    "3-D surface (:meth:`mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface`).",
)

__all__ = [
    "contour",
    "contourf",
    "hist",
    "imshow",
    "line",
    "pcolormesh",
    "plot",
    "scatter",
    "step",
    "surface",
]
