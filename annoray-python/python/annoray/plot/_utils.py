"""Shared machinery of the plotting module: labels, colormap parameters, axes handling.

Semantics follow ``xarray.plot`` (labels from CF attributes, robust and divergent colormap
heuristics, discrete levels, interval inference) so that plots of the same data agree. Units
come from the array's own unit rather than from a ``units`` attribute.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize

    from annoray._annoray import DataArray, Dataset

#: Percentiles that bound the colormap with ``robust=True``.
ROBUST_PERCENTILE = 2.0

#: Default colormaps, as ``xarray.set_options(cmap_sequential=..., cmap_divergent=...)``.
OPTIONS: dict[str, Any] = {"cmap_sequential": "viridis", "cmap_divergent": "RdBu_r"}


def import_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("annoray.plot requires matplotlib (a dependency of annoray)") from e
    return plt


# ----- labels -----


def units_of(da: DataArray) -> str | None:
    """The array's unit; else a ``units``/``unit`` attribute."""
    if da.units is not None:
        return da.units
    for key in ("units", "unit"):
        if key in da.attrs:
            return str(da.attrs[key])
    return None


def label(da: DataArray | None, extra: str = "") -> str:
    """``long_name`` (else ``standard_name``, else the name) and ``[units]``, wrapped at 30
    characters unless the name is LaTeX."""
    if da is None:
        return ""
    attrs = da.attrs
    if "long_name" in attrs:
        name = str(attrs["long_name"])
    elif "standard_name" in attrs:
        name = str(attrs["standard_name"])
    elif da.name is not None:
        name = str(da.name)
    else:
        name = ""
    u = units_of(da)
    text = name + extra + (f" [{u}]" if u is not None else "")
    if name.startswith("$") and name.count("$") % 2 == 0:
        return text
    return "\n".join(textwrap.wrap(text, 30))


def format_item(x: Any) -> str:
    """A scalar coordinate value as ``xarray.plot`` titles show it."""
    a = np.asarray(x)
    if a.dtype.kind == "M":
        ns = a.astype("datetime64[ns]")
        day = ns.astype("datetime64[D]")
        if ns == day:
            return str(day)
        return str(ns.astype("datetime64[s]")).replace("T", " ")
    if a.dtype.kind == "m":
        return str(a)
    if a.dtype.kind == "f":
        return f"{float(a):.4}"
    return str(a.item() if a.ndim == 0 else a)


def title_for_slice(da: DataArray | Dataset, truncate: int = 50) -> str:
    """``"lat = 75.0 [degrees_north], time = 2013-01-01"`` from the size-1 coordinates."""
    parts = []
    for name, c in da.coords.items():
        if c.size == 1:
            unit = units_of(c)
            value = format_item(np.asarray(c.values).reshape(()))
            parts.append(f"{name} = {value}" + (f" [{unit}]" if unit is not None else ""))
    title = ", ".join(parts)
    if len(title) > truncate:
        title = title[: truncate - 3] + "..."
    return title


# ----- coordinates -----


def coord(da: DataArray | Dataset, name: str) -> DataArray:
    """A coordinate (or variable) by name; a bare dimension becomes ``0..n-1``."""
    from annoray._annoray import DataArray as _DA
    from annoray._annoray import Dataset as _DS

    if name in da.coords:
        return da.coords[name]
    if isinstance(da, _DS) and name in da.data_vars:
        return da.data_vars[name]
    if name in da.dims:
        return _DA(np.arange(da.sizes[name]), dims=(name,), name=name)
    raise ValueError(f"{name!r} is not a dimension or coordinate")


def valid_names(da: DataArray | Dataset) -> set[str]:
    return set(da.dims) | set(da.coords)


def check_name(da: DataArray | Dataset, value: str | None, what: str) -> None:
    if value is not None and value not in valid_names(da):
        valid = "', '".join(sorted(valid_names(da)))
        raise ValueError(f"{what} must be one of None, '{valid}'. Received '{value}' instead.")


def infer_xy(da: DataArray, x: str | None, y: str | None) -> tuple[str, str]:
    """The x and y names of a 2-D plot (by default y is the first dimension)."""
    if x is not None and x == y:
        raise ValueError("x and y cannot be equal.")
    if x is None and y is None:
        if da.ndim != 2:
            raise ValueError("DataArray must be 2d")
        y, x = da.dims
        return x, y
    if x is None:
        check_name(da, y, "y")
        assert y is not None
        x = da.dims[0] if y == da.dims[1] else da.dims[1]
    elif y is None:
        check_name(da, x, "x")
        y = da.dims[0] if x == da.dims[1] else da.dims[1]
    else:
        check_name(da, x, "x")
        check_name(da, y, "y")
    return x, y


def squeezed(da: DataArray, keep: Iterable[str | None] = ()) -> DataArray:
    """Drop length-1 dimensions (keeping them as scalar coordinates), except ``keep``."""
    keep = set(keep)
    drop = {d: 0 for d, n in da.sizes.items() if n == 1 and d not in keep}
    return da.isel(drop) if drop else da


def is_datetime(a: np.ndarray) -> bool:
    return a.dtype.kind == "M"


def rotate_dates(ax: Axes, values: np.ndarray, axis: str = "x") -> None:
    if is_datetime(values):
        labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
        for t in labels:
            t.set_rotation(30)
            t.set_horizontalalignment("right")


# ----- axes -----


def get_axis(
    figsize: Sequence[float] | None = None,
    size: float | None = None,
    aspect: float | str | None = None,
    ax: Axes | None = None,
    **subplot_kws: Any,
) -> Axes:
    plt = import_pyplot()
    if figsize is not None:
        if ax is not None:
            raise ValueError("cannot provide both `figsize` and `ax` arguments")
        if size is not None:
            raise ValueError("cannot provide both `figsize` and `size` arguments")
        _, new = plt.subplots(figsize=figsize, subplot_kw=subplot_kws)
        return new
    if size is not None:
        if ax is not None:
            raise ValueError("cannot provide both `size` and `ax` arguments")
        if aspect is None or aspect == "auto":
            width, height = plt.rcParams["figure.figsize"]
            faspect = width / height
        elif aspect == "equal":
            faspect = 1.0
        else:
            faspect = float(aspect)
        _, new = plt.subplots(figsize=(size * faspect, size), subplot_kw=subplot_kws)
        return new
    if aspect is not None:
        raise ValueError("cannot provide `aspect` argument without `size`")
    if subplot_kws and ax is not None:
        raise ValueError("cannot use subplot_kws with existing ax")
    if ax is not None:
        return ax
    return plt.gca() if plt.gcf().axes else plt.axes(**subplot_kws)


def update_axes(
    ax: Axes,
    xincrease: bool | None = None,
    yincrease: bool | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
    xticks: Any = None,
    yticks: Any = None,
    xlim: Any = None,
    ylim: Any = None,
) -> None:
    if xincrease is not None and bool(xincrease) == ax.xaxis_inverted():
        ax.invert_xaxis()
    if yincrease is not None and bool(yincrease) == ax.yaxis_inverted():
        ax.invert_yaxis()
    # Setting a scale resets the formatters (losing date labels): only when asked.
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def is_monotonic(c: np.ndarray, axis: int = 0) -> bool:
    if c.shape[axis] < 3:
        return True
    d = np.diff(c, axis=axis)
    zero = np.zeros((), dtype=d.dtype)
    return bool(np.all(d >= zero) or np.all(d <= zero))


def interval_breaks(
    c: Any, axis: int = 0, scale: str | None = None, check_monotonic: bool = False
) -> np.ndarray:
    """Cell edges from cell centres (half-way between neighbours, extrapolated at the ends;
    in log space for ``scale="log"``)."""
    c = np.asarray(c)
    if check_monotonic and not is_monotonic(c, axis):
        raise ValueError(
            f"The input coordinate is not sorted in increasing order along axis {axis}. "
            "Sort the array (e.g. with isel) before plotting."
        )
    if scale == "log":
        if (c <= 0).any():
            raise ValueError("Coordinates must be positive on logscale plots.")
        c = np.log10(c)
    deltas = 0.5 * np.diff(c, axis=axis)
    if deltas.size == 0:
        deltas = np.zeros((1,) * c.ndim, dtype=deltas.dtype)
    first = np.take(c, [0], axis=axis) - np.take(deltas, [0], axis=axis)
    last = np.take(c, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
    trim = tuple(slice(None, -1) if n == axis else slice(None) for n in range(c.ndim))
    breaks = np.concatenate([first, c[trim] + deltas, last], axis=axis)
    return np.power(10, breaks) if scale == "log" else breaks


# ----- colormaps -----


def as_colour_values(a: np.ndarray) -> np.ndarray:
    """Values a colormap can take: dates as matplotlib date numbers, durations in seconds."""
    if a.dtype.kind == "M":
        import matplotlib.dates as mdates

        return np.asarray(mdates.date2num(a))
    if a.dtype.kind == "m":
        return a / np.timedelta64(1, "s")
    return a


def date_colorbar(cbar: Any, values: np.ndarray) -> None:
    """Date tick labels on a colorbar of date numbers."""
    if values.dtype.kind == "M":
        import matplotlib.dates as mdates

        locator = mdates.AutoDateLocator()
        cbar.ax.yaxis.set_major_locator(locator)
        cbar.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _extend(calc: np.ndarray, vmin: Any, vmax: Any) -> str:
    lo, hi = calc.min() < vmin, calc.max() > vmax
    return "both" if lo and hi else "min" if lo else "max" if hi else "neither"


def _palette(cmap: Any, n: int) -> np.ndarray:
    import itertools

    import matplotlib as mpl
    from matplotlib.colors import ListedColormap

    at = np.linspace(0, 1.0, n)
    if isinstance(cmap, list | tuple):
        return ListedColormap(list(itertools.islice(itertools.cycle(cmap), n)))(at)
    if isinstance(cmap, str):
        try:
            return mpl.colormaps[cmap](at)
        except KeyError:
            try:
                from seaborn import color_palette  # type: ignore[import-not-found]

                return np.asarray(color_palette(cmap, n_colors=n))
            except (ValueError, ImportError):
                return ListedColormap([cmap] * n)(at)  # a single colour
    return cmap(at)


def discrete_cmap(cmap: Any, levels: Sequence[float], extend: str, filled: bool) -> tuple:
    """A colormap with one colour per level interval, and its norm."""
    import matplotlib.colors as mcolors

    if len(levels) == 1:
        levels = [levels[0], levels[0]]
    if not filled:
        extend = "max"
    ext_n = {"both": 2, "min": 1, "max": 1}.get(extend, 0)
    pal = _palette(cmap, len(levels) + ext_n - 1)
    new, norm = mcolors.from_levels_and_colors(levels, list(pal), extend=cast("Any", extend))
    new.name = getattr(cmap, "name", cmap)
    if isinstance(cmap, mcolors.Colormap):
        under = cmap(-np.inf)
        over = cmap(np.inf)
        new = new.with_extremes(
            bad=cmap(np.nan),
            under=None if under == cmap(0) else under,
            over=None if over == cmap(cmap.N - 1) else over,
        )
    return new, norm


def cmap_params(
    data: Any,
    vmin: Any = None,
    vmax: Any = None,
    cmap: str | Colormap | None = None,
    center: Any = None,
    robust: bool = False,
    extend: str | None = None,
    levels: Any = None,
    filled: bool = True,
    norm: Normalize | None = None,
) -> dict[str, Any]:
    """Colormap, limits, levels and norm for ``data``, with xarray's heuristics: robust
    limits from the 2nd and 98th percentiles; a divergent, symmetric map when the data
    straddle zero or ``center`` is given (``center=False`` disables it); integer ``levels``
    become nice tick values; ``extend`` follows from data outside the limits."""
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker

    # Times are coloured by number: matplotlib date numbers, or seconds for durations.
    # (xarray keeps them as times, where the centring arithmetic fails.)
    data = as_colour_values(np.asarray(data))
    zero = 0.0
    if isinstance(levels, Iterable):
        levels = sorted(levels)
    calc = np.ravel(data[np.isfinite(data)]) if data.dtype.kind in "fc" else np.ravel(data)
    if calc.size == 0:
        calc = np.array(zero)

    possibly_divergent = center is not False
    center_is_none = center is None
    if center is None or center is False:
        center = zero
    if vmin is not None and vmax is not None:
        possibly_divergent = False
    user_minmax = vmin is not None or vmax is not None
    vlim = None
    vmin_was_none, vmax_was_none = vmin is None, vmax is None

    if vmin is None:
        vmin = np.percentile(calc, ROBUST_PERCENTILE) if robust else calc.min()
    elif possibly_divergent:
        vlim = abs(vmin - center)
    if vmax is None:
        vmax = np.percentile(calc, 100 - ROBUST_PERCENTILE) if robust else calc.max()
    elif possibly_divergent:
        vlim = abs(vmax - center)

    if possibly_divergent:
        divergent_levels = isinstance(levels, Sequence) and levels[0] * levels[-1] < zero
        divergent = (vmin < zero < vmax) or not center_is_none or divergent_levels
    else:
        divergent = False
    if divergent:
        if vlim is None:
            vlim = max(abs(vmin - center), abs(vmax - center))
        vmin, vmax = -vlim, vlim
    vmin += center
    vmax += center

    if norm is not None:
        if norm.vmin is None:
            norm.vmin = vmin
        elif not vmin_was_none and vmin != norm.vmin:
            raise ValueError("Cannot supply vmin and a norm with a different vmin.")
        else:
            vmin = norm.vmin
        if norm.vmax is None:
            norm.vmax = vmax
        elif not vmax_was_none and vmax != norm.vmax:
            raise ValueError("Cannot supply vmax and a norm with a different vmax.")
        else:
            vmax = norm.vmax
    if isinstance(norm, mcolors.BoundaryNorm):
        levels = norm.boundaries

    if cmap is None:
        cmap = OPTIONS["cmap_divergent" if divergent else "cmap_sequential"]

    if levels is not None:
        if np.ndim(levels) == 0:
            n_levels = int(cast("int", levels))
            if user_minmax:
                levels = np.linspace(vmin, vmax, n_levels)
            elif n_levels == 1:
                levels = np.asarray([(vmin + vmax) / 2])
            else:  # MaxNLocator counts bins, not ticks
                levels = mticker.MaxNLocator(n_levels - 1).tick_values(vmin, vmax)
        vmin, vmax = levels[0], levels[-1]

    if vmin == vmax:
        vmin, vmax = mticker.LinearLocator(2).tick_values(vmin, vmax)
    if extend is None:
        extend = _extend(calc, vmin, vmax)
    if levels is not None and not isinstance(norm, mcolors.BoundaryNorm):
        cmap, discrete_norm = discrete_cmap(cmap, levels, extend, filled)
        norm = discrete_norm if norm is None else norm
    if norm is not None:  # the norm carries the limits
        vmin = vmax = None
    return {
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap,
        "extend": extend,
        "levels": levels,
        "norm": norm,
    }


def add_colorbar(
    primitive: Any,
    ax: Axes,
    cbar_ax: Axes | None,
    cbar_kwargs: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Any:
    kw = dict(cbar_kwargs)
    kw.setdefault("extend", params["extend"])
    if cbar_ax is None:
        kw.setdefault("ax", ax)
    else:
        kw.setdefault("cax", cbar_ax)
    if hasattr(primitive, "extend"):  # contour sets carry their own
        kw.pop("extend")
    fig = ax.get_figure()
    assert fig is not None
    return fig.colorbar(primitive, **kw)
