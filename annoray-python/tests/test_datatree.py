"""DataTree against xarray's: nodes as views of their tree, inherited coordinates,
node-by-node computation, editing, comparison, files."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")


def build(m):
    root = m.Dataset({"g": ("x", [1.0, 2.0])}, coords={"x": [10.0, 20.0]}, attrs={"title": "t"})
    a = m.Dataset({"v": ("x", [3.0, 4.0])})
    b = m.Dataset({"w": (("x", "y"), np.arange(6.0).reshape(2, 3))}, coords={"y": [1.0, 2.0, 3.0]})
    return m.DataTree.from_dict({"/": root, "/a": a, "/a/b": b, "/c": m.Dataset()})


def same_tree(r, x):
    """Same paths, and each node's own dataset equal (values, variables, coordinates)."""
    assert [n.path for n in r.subtree] == [n.path for n in x.subtree]
    for rn, xn in zip(r.subtree, x.subtree, strict=True):
        rd, xd = rn.to_dataset(inherit=False), xn.to_dataset(inherit=False)
        assert sorted(rd.data_vars) == sorted(xd.data_vars), rn.path
        assert sorted(rd.coords) == sorted(xd.coords), rn.path
        for k in xd.variables:
            np.testing.assert_allclose(
                np.asarray(rd[k].values, float), np.asarray(xd[k].values, float)
            )


def test_structure_and_navigation():
    t, x = build(ar), build(xr)
    for f in (
        lambda t: [n.path for n in t.subtree],
        lambda t: [n.path for n in t.leaves],
        lambda t: [n.path for n in t["a/b"].parents],
        lambda t: list(t.groups),
        lambda t: (t["a/b"].depth, t["a/b"].level, t.width, t["a/b"].is_leaf, t.is_root),
        lambda t: (t["c"].has_data, t.has_attrs, t["c"].is_empty),
        lambda t: (t["a/b"].relative_to(t["c"]), t["a/b"].find_common_ancestor(t["c"]).path),
        lambda t: (t["a"].name, t["a/b"].path, t["a/b"].parent.path, t["a/b"].root.path),
        lambda t: (sorted(t), len(t), "a" in t, "g" in t, "a/v" in t, "zz" in t),
        lambda t: sorted(t["a/b"].coords),
        lambda t: sorted(t["a/b"].to_dataset(inherit=False).coords),
        lambda t: t["a/b"]["x"].values.tolist(),
        lambda t: dict(t["a/b"].sizes),
    ):
        assert f(t) == f(x)


def test_nodes_are_views():
    t = build(ar)
    a = t["a"]
    a["new"] = a["v"] * 3
    assert t["a/new"].values.tolist() == [9.0, 12.0]
    t["a/b"].attrs["k"] = "v"
    assert t["a/b"].attrs == {"k": "v"}
    t["d"] = ar.Dataset({"q": ("z", [1.0])})
    t["a/e"] = ar.DataTree(ar.Dataset({"r": ("x", [5.0, 6.0])}))
    assert sorted(t.children) == ["a", "c", "d"] and t["a/e/r"].values.tolist() == [5.0, 6.0]
    del t["c"]
    del t["a/new"]
    assert "c" not in t and "new" not in t["a"]
    # Edits are checked against inherited coordinates; the tree is unchanged on error.
    with pytest.raises(ValueError):
        t["a/bad"] = ar.Dataset({"z": ("x", [1.0, 2.0, 3.0])})
    assert "bad" not in t["a"]
    # A removed node: views taken earlier say so; copies stay usable.
    b, kept = t["a/b"], t["a/b"].copy()
    del t["a/b"]
    with pytest.raises(KeyError, match="copy"):
        b.to_dataset()
    assert kept["w"].shape == (2, 3)
    # orphan: the node becomes its own tree.
    e = t["a/e"]
    e.orphan()
    assert e.is_root and e.name == "e" and "e" not in t["a"]


def test_computation_matches_xarray():
    t, x = build(ar), build(xr)
    for f in (
        lambda t: t * 2,
        lambda t: t + t,
        lambda t: -t,
        lambda t: t.mean("x"),
        lambda t: t.sum(),
        lambda t: t.isel(x=0),
        lambda t: t.sel(x=10.0),
        lambda t: t.map_over_datasets(lambda d: d * 2),
        lambda t: t.match("*/b"),
        lambda t: t.filter(lambda n: "v" in n.ds),
        lambda t: t.drop_nodes("c"),
        lambda t: t.prune(),
        lambda t: t.assign(z=t["g"] * 2),
    ):
        same_tree(f(t), f(x))
    same_tree(ar.map_over_datasets(lambda d: d * 2, t), xr.map_over_datasets(lambda d: d * 2, x))
    assert [p for p, _ in ar.group_subtrees(t, t)] == [p for p, _ in xr.group_subtrees(x, x)]
    gt = (t > 2)["a/v"].values.tolist()
    assert gt == (x > 2)["a/v"].values.tolist()


def test_comparison_copy_pickle():
    t = build(ar)
    assert t.equals(t.copy()) and t.identical(t.copy()) and t.isomorphic(t.copy())
    other = t.copy()
    other["a/v"] = other["a/v"] + 1
    assert not t.equals(other) and t.isomorphic(other)
    other.attrs["extra"] = "1"
    assert not t.identical(other)
    back = pickle.loads(pickle.dumps(t))
    assert t.identical(back)
    assert sorted(t.to_dict()) == ["/", "/a", "/a/b", "/c"]
    assert sorted(t.to_dict(relative=True)) == [".", "a", "a/b", "c"]


def test_files(tmp_path):
    t = build(ar)
    nc, store = tmp_path / "t.nc", tmp_path / "t.zarr"
    t.to_netcdf(nc)
    back = ar.open_datatree(nc)
    assert back.equals(t)
    same_tree(back, xr.open_datatree(nc))
    assert sorted(ar.open_groups(nc)) == sorted(xr.open_groups(nc))
    t.to_zarr(store)
    assert ar.open_datatree(store, engine="zarr").equals(t)
    same_tree(ar.open_datatree(store), xr.open_datatree(store, engine="zarr"))
    lazy = ar.open_datatree(nc)
    assert lazy.load() is lazy
    same_tree(lazy.compute(), t)
