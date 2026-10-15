"""annoray.plot against xarray.plot: the same data must give the same matplotlib artists
(labels, titles, colour limits and maps, levels, meshes, lines, legends, facets)."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import annoray as ar

xr = pytest.importorskip("xarray")

T = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], dtype="datetime64[ns]")
LAT = np.array([10.0, 20.0, 30.0])
LON = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
RNG = np.random.default_rng(0)
VALUES = 280.0 + 10 * RNG.standard_normal((4, 3, 5))
ATTRS = {"long_name": "Air temperature", "units": "K"}


def pair(values=VALUES, attrs=ATTRS):
    """The same 3-D array as annoray and xarray objects."""
    coords = {"time": T, "lat": LAT, "lon": LON}
    x = xr.DataArray(values, dims=("time", "lat", "lon"), coords=coords, name="air", attrs=attrs)
    for k, a in (
        ("lat", {"long_name": "Latitude", "units": "degrees_north"}),
        ("lon", {"long_name": "Longitude", "units": "degrees_east"}),
    ):
        x[k].attrs.update(a)
    a = ar.from_xarray(x)
    return a, x


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def axes_of(f):
    """Draw with ``f`` on a fresh figure; the figure's axes."""
    fig = plt.figure()
    out = f()
    return fig, out


def texts(ax):
    return ax.get_xlabel(), ax.get_ylabel(), ax.get_title()


def cbar_label(fig):
    return [a.get_ylabel() for a in fig.axes[1:]]


def same_mappable(m1, m2):
    np.testing.assert_allclose(
        np.ma.filled(m1.get_array(), np.nan), np.ma.filled(m2.get_array(), np.nan)
    )
    assert m1.get_clim() == pytest.approx(m2.get_clim())
    assert m1.get_cmap().name == m2.get_cmap().name


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"robust": True},
        {"vmin": 270},
        {"center": 280},
        {"cmap": "magma"},
        {"levels": 5},
        {"levels": [270, 280, 290]},
        {"x": "lat", "y": "lon"},
        {"yincrease": False},
        {"center": False},
    ],
)
def test_pcolormesh_matches_xarray(kwargs):
    a, x = pair()
    f1, m1 = axes_of(lambda: a.isel({"time": 0}).plot.pcolormesh(**kwargs))
    f2, m2 = axes_of(lambda: x.isel(time=0).plot.pcolormesh(**kwargs))
    same_mappable(m1, m2)
    np.testing.assert_allclose(m1.get_coordinates(), m2.get_coordinates())
    assert texts(f1.axes[0]) == texts(f2.axes[0])
    assert cbar_label(f1) == cbar_label(f2)
    assert f1.axes[0].get_ylim() == pytest.approx(f2.axes[0].get_ylim())


def test_divergent_default_matches_xarray():
    a, x = pair()
    a2 = a.isel({"time": 0}) - ar.DataArray(280.0, units="K")
    x2 = x.isel(time=0) - 280.0
    _, m1 = axes_of(lambda: a2.plot())
    _, m2 = axes_of(lambda: x2.plot())
    same_mappable(m1, m2)
    assert m1.get_cmap().name == "RdBu_r"
    lo, hi = m1.get_clim()
    assert lo == pytest.approx(-hi)


@pytest.mark.parametrize("kind", ["contour", "contourf"])
@pytest.mark.parametrize(
    "kwargs", [{}, {"levels": 7}, {"levels": [275, 280, 285]}, {"colors": "k"}]
)
def test_contours_match_xarray(kind, kwargs):
    a, x = pair()
    f1, c1 = axes_of(lambda: getattr(a.isel({"time": 1}).plot, kind)(**kwargs))
    f2, c2 = axes_of(lambda: getattr(x.isel(time=1).plot, kind)(**kwargs))
    np.testing.assert_allclose(c1.levels, c2.levels)
    assert c1.extend == c2.extend
    assert texts(f1.axes[0]) == texts(f2.axes[0])
    assert len(f1.axes) == len(f2.axes)


def test_imshow_matches_xarray():
    a, x = pair()
    f1, m1 = axes_of(lambda: a.isel({"time": 2}).plot.imshow())
    f2, m2 = axes_of(lambda: x.isel(time=2).plot.imshow())
    same_mappable(m1, m2)
    assert m1.get_extent() == pytest.approx(m2.get_extent())
    assert f1.axes[0].get_ylim() == pytest.approx(f2.axes[0].get_ylim())


def test_imshow_rgb_matches_xarray():
    rgb = np.random.default_rng(1).random((6, 5, 3))
    x = xr.DataArray(
        rgb, dims=("y", "x", "band"), coords={"x": np.arange(5.0), "y": np.arange(6.0)}
    )
    a = ar.from_xarray(x)
    f1, m1 = axes_of(lambda: a.plot.imshow(robust=True))
    f2, m2 = axes_of(lambda: x.plot.imshow(robust=True))
    np.testing.assert_allclose(m1.get_array(), m2.get_array())
    assert len(f1.axes) == len(f2.axes) == 1


def test_surface_runs_with_labels():
    a, x = pair()
    f1, _ = axes_of(lambda: a.isel({"time": 0}).plot.surface())
    f2, _ = axes_of(lambda: x.isel(time=0).plot.surface())
    ax1, ax2 = f1.axes[0], f2.axes[0]
    assert isinstance(ax1, Axes3D) and isinstance(ax2, Axes3D)
    labels = [(ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) for ax in (ax1, ax2)]
    assert labels[0] == labels[1]


def test_line_1d_matches_xarray():
    a, x = pair()
    f1, (l1,) = axes_of(lambda: a.isel({"lat": 0, "lon": 0}).plot())
    f2, (l2,) = axes_of(lambda: x.isel(lat=0, lon=0).plot())
    np.testing.assert_array_equal(l1.get_xdata(), l2.get_xdata())
    np.testing.assert_allclose(l1.get_ydata(), l2.get_ydata())
    assert texts(f1.axes[0]) == texts(f2.axes[0])
    fmt = f1.axes[0].xaxis.get_major_formatter()
    assert type(fmt).__name__ == "ConciseDateFormatter"


@pytest.mark.parametrize(
    "kwargs", [{"hue": "lat"}, {"x": "time"}, {"y": "lon"}, {"x": "time", "add_legend": False}]
)
def test_line_2d_matches_xarray(kwargs):
    a, x = pair()
    a2, x2 = a.isel({"lon": slice(0, 3)}).isel({"lat": 1}), x.isel(lon=slice(0, 3), lat=1)
    if "hue" in kwargs:
        a2, x2 = a.isel({"lon": 0}), x.isel(lon=0)
    f1, l1 = axes_of(lambda: a2.plot.line(**kwargs))
    f2, l2 = axes_of(lambda: x2.plot.line(**kwargs))
    assert len(l1) == len(l2)
    for p, q in zip(l1, l2, strict=True):
        np.testing.assert_array_equal(p.get_xdata(), q.get_xdata())
        np.testing.assert_allclose(p.get_ydata(), q.get_ydata())
    assert texts(f1.axes[0]) == texts(f2.axes[0])
    leg1, leg2 = f1.axes[0].get_legend(), f2.axes[0].get_legend()
    assert (leg1 is None) == (leg2 is None)
    if leg1 is not None and leg2 is not None:
        assert [t.get_text() for t in leg1.get_texts()] == [t.get_text() for t in leg2.get_texts()]
        assert leg1.get_title().get_text() == leg2.get_title().get_text()


def test_step_matches_xarray():
    a, x = pair()
    _, (l1,) = axes_of(lambda: a.isel({"lat": 0, "lon": 0}).plot.step(where="mid"))
    _, (l2,) = axes_of(lambda: x.isel(lat=0, lon=0).plot.step(where="mid"))
    assert l1.get_drawstyle() == l2.get_drawstyle() == "steps-mid"
    with pytest.raises(ValueError, match="where"):
        a.isel({"lat": 0, "lon": 0}).plot.step(where="up")


def test_hist_matches_xarray():
    values = VALUES.copy()
    values[0, 0, 0] = np.nan
    a, x = pair(values)
    f1, (n1, b1, _) = axes_of(lambda: a.plot(bins=7))
    f2, (n2, b2, _) = axes_of(lambda: x.plot(bins=7))
    np.testing.assert_array_equal(n1, n2)
    np.testing.assert_allclose(b1, b2)
    assert texts(f1.axes[0]) == texts(f2.axes[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"col": "time"},
        {"col": "time", "col_wrap": 3},
        {"col": "time", "robust": True},
        {"col": "time", "levels": 4},
        {"col": "time", "kind": "contourf"},
    ],
)
def test_facets_match_xarray(kwargs):
    a, x = pair()
    kind = kwargs.pop("kind", None)
    g1 = getattr(a.plot, kind)(**kwargs) if kind else a.plot(**kwargs)
    g2 = getattr(x.plot, kind)(**kwargs) if kind else x.plot(**kwargs)
    assert g1.axs.shape == g2.axs.shape
    for p, q in zip(g1.axs.flat, g2.axs.flat, strict=True):
        assert p.get_visible() == q.get_visible()
        if p.get_visible():
            assert (p.get_ylabel(), p.get_title()) == (q.get_ylabel(), q.get_title())
            # annoray labels the lowest facet of every column (xarray only the last row,
            # leaving wrapped columns unlabelled).
            assert q.get_xlabel() in ("", p.get_xlabel())
    assert g1.cbar.ax.get_ylabel() == g2.cbar.ax.get_ylabel()
    assert g1._mappables[-1].get_clim() == pytest.approx(g2._mappables[-1].get_clim())


def test_row_col_facets_match_xarray():
    a, x = pair()
    g1 = a.plot(row="time", col="lat")
    g2 = x.plot(row="time", col="lat")
    assert g1.axs.shape == g2.axs.shape == (4, 3)
    assert [ax.get_title() for ax in g1.axs[0]] == [ax.get_title() for ax in g2.axs[0]]
    ann1 = [t.get_text() for t in g1.axs[0, -1].texts]
    ann2 = [t.get_text() for t in g2.axs[0, -1].texts]
    assert ann1 == ann2


def test_facet_lines_match_xarray():
    a, x = pair()
    g1 = a.isel({"lon": slice(0, 2)}).plot.line(x="time", hue="lon", col="lat")
    g2 = x.isel(lon=slice(0, 2)).plot.line(x="time", hue="lon", col="lat")
    assert g1.axs.shape == g2.axs.shape
    assert [ax.get_title() for ax in g1.axs.flat] == [ax.get_title() for ax in g2.axs.flat]
    assert [t.get_text() for t in g1.figlegend.get_texts()] == [
        t.get_text() for t in g2.figlegend.get_texts()
    ]
    assert g1.axs[0, 0].get_ylabel() == g2.axs[0, 0].get_ylabel()


def test_dataset_quiver_and_streamplot():
    y, xx = np.arange(4.0), np.arange(5.0)
    u = np.cos(xx)[None, :] * np.ones((4, 1))
    v = np.sin(y)[:, None] * np.ones((1, 5))
    xd = xr.Dataset(
        {
            "u": (("y", "x"), u, {"units": "m s-1"}),
            "v": (("y", "x"), v),
            "speed": (("y", "x"), np.hypot(u, v)),
        },
        coords={"x": xx, "y": y},
    )
    ad = ar.from_xarray(xd)
    _, q1 = axes_of(lambda: ad.plot.quiver(x="x", y="y", u="u", v="v", hue="speed"))
    _, q2 = axes_of(lambda: xd.plot.quiver(x="x", y="y", u="u", v="v", hue="speed"))
    np.testing.assert_allclose(q1.U, q2.U)
    np.testing.assert_allclose(q1.V, q2.V)
    # Colours follow the arrows. (xarray passes the hue in the variable's own (y, x)
    # order while the arrows are broadcast to (x, y), so its colours are transposed.)
    np.testing.assert_allclose(q1.get_array(), np.hypot(q1.U, q1.V))
    f1, s1 = axes_of(lambda: ad.plot.streamplot(x="x", y="y", u="u", v="v"))
    assert len(s1.get_segments()) > 0
    assert f1.axes[0].get_xlabel() == "x"
    xd3 = xr.concat([xd, xd * 2], dim="t")
    g = ar.from_xarray(xd3).plot.quiver(x="x", y="y", u="u", v="v", col="t", hue="speed")
    assert g.axs.shape == (1, 2)
    assert g.cbar is not None


def test_dataset_scatter():
    _, x = pair()
    xd = x.isel(time=0).to_dataset(name="air")
    xd["other"] = xd["air"] * 2
    ad = ar.from_xarray(xd)
    _, p1 = axes_of(lambda: ad.plot.scatter(x="air", y="other", hue="lat"))
    np.testing.assert_allclose(p1.get_offsets()[:, 0], np.ravel(xd["air"].values))
    np.testing.assert_allclose(p1.get_offsets()[:, 1], np.ravel(xd["other"].values))
    with pytest.raises(ValueError, match="cannot be called directly"):
        ad.plot()


def test_dataarray_scatter_hues():
    a, _ = pair()
    _, p = axes_of(lambda: a.isel({"lat": 0}).plot.scatter(x="lon", hue="time"))
    assert p.get_offsets().shape == (20, 2)  # dates coloured by number
    a2 = ar.DataArray(np.arange(4.0), dims=("x",), coords={"kind": ("x", ["a", "b", "a", "b"])})
    _, ps = axes_of(lambda: a2.plot.scatter(hue="kind"))
    assert [p.get_label() for p in ps] == ["a", "b"]


def test_units_label_from_annoray_unit():
    a = ar.DataArray(np.arange(6.0).reshape(2, 3), dims=("y", "x"), name="h", units="km")
    fig, _ = axes_of(lambda: a.to("m").plot())
    assert fig.axes[1].get_ylabel() == "h [m]"


def test_errors():
    a, _ = pair()
    with pytest.raises(ValueError, match="x and y cannot be equal"):
        a.isel({"time": 0}).plot(x="lat", y="lat")
    with pytest.raises(ValueError, match="hue, x or y"):
        a.isel({"time": 0}).plot.line()
    with pytest.raises(ValueError, match="Line plots are for 1- or 2-dimensional"):
        a.plot.line()
    with pytest.raises(ValueError, match="both cmap and colors"):
        a.isel({"time": 0}).plot.contour(cmap="viridis", colors="k")
    with pytest.raises(TypeError, match="No numeric data"):
        a.isel({"time": slice(0, 0)}).plot.hist()
