"""DataArray.str against xarray's StringAccessor."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

xr = pytest.importorskip("xarray")

TEXT = np.array([["ab cd", "Cd-x"], ["e f g", ""], ["  pad ", "a|b|a"]])


def build(m):
    return m.DataArray(TEXT, dims=("x", "y"), coords={"x": [1, 2, 3]}, name="t", attrs={"k": "v"})


CASES = [
    lambda s: s.len(),
    lambda s: s.upper(),
    lambda s: s.lower(),
    lambda s: s.title(),
    lambda s: s.capitalize(),
    lambda s: s.swapcase(),
    lambda s: s.casefold(),
    lambda s: s.strip(),
    lambda s: s.lstrip(),
    lambda s: s.rstrip(" d"),
    lambda s: s[0],
    lambda s: s[1:3],
    lambda s: s[::-1],
    lambda s: s.get(5, default="?"),
    lambda s: s.slice_replace(1, 2, "Z"),
    lambda s: s.pad(7, side="both", fillchar="*"),
    lambda s: s.center(8),
    lambda s: s.ljust(6, "."),
    lambda s: s.rjust(6, "."),
    lambda s: s.zfill(6),
    lambda s: s.repeat(2),
    lambda s: s.startswith("a"),
    lambda s: s.endswith("x"),
    lambda s: s.contains("a"),
    lambda s: s.contains("A", case=False),
    lambda s: s.contains("|", regex=False),
    lambda s: s.match(r"\w"),
    lambda s: s.count("a"),
    lambda s: s.find("d"),
    lambda s: s.rfind("a"),
    lambda s: s.replace("a", "@"),
    lambda s: s.replace(r"\s+", "_"),
    lambda s: s.replace("a", "@", n=1),
    lambda s: s.isalpha(),
    lambda s: s.isspace(),
    lambda s: s.istitle(),
    lambda s: s.split(dim="w"),
    lambda s: s.split(dim="w", sep="|"),
    lambda s: s.rsplit(dim="w", maxsplit=1),
    lambda s: s.partition(dim="p"),
    lambda s: s.rpartition(dim="p", sep="-"),
    lambda s: s.extract(r"(\w)(\w)", dim="g"),
    lambda s: s.extract(r"(\w)\w", dim=None),
    lambda s: s.get_dummies(dim="tok"),
    lambda s: s.join("y", sep="+"),
    lambda s: s.wrap(3),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_matches_xarray(case):
    f = CASES[case]
    r, x = f(build(ar).str), f(build(xr).str)
    assert r.dims == x.dims
    assert r.values.tolist() == x.values.tolist(), (r.values, x.values)
    assert r.attrs == x.attrs and r.name == x.name
    for k in x.coords:
        assert np.asarray(r.coords[k].values).tolist() == np.asarray(x.coords[k].values).tolist()


def test_extractall_labels_its_dimensions_correctly():
    # xarray swaps them: its axis named group_dim indexes matches, match_dim the groups (see
    # its own docstring example). Here "g" indexes capture groups and "m" matches.
    r = build(ar).str.extractall(r"(\w)(\w)", group_dim="g", match_dim="m")
    x = build(xr).str.extractall(r"(\w)(\w)", group_dim="g", match_dim="m")
    assert r.dims == x.dims == ("x", "y", "g", "m")
    assert r.sizes["g"] == 2  # two capture groups
    assert np.swapaxes(r.values, 2, 3).tolist() == x.values.tolist()
    assert r.isel(x=0, y=0, m=1).values.tolist() == ["c", "d"]  # second match of "ab cd"


def test_cat_and_refusals():
    a, x = build(ar), build(xr)
    assert a.str.cat(a, sep="-").values.tolist() == x.str.cat(x, sep="-").values.tolist()
    with pytest.raises(TypeError, match="bytes"):
        a.str.encode("utf-8")
    with pytest.raises(TypeError, match="lists"):
        a.str.split(dim=None)
    with pytest.raises(TypeError, match="text"):
        _ = ar.DataArray([1.0], dims="x").str
