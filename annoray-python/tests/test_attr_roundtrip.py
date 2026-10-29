"""Attributes of every type round-trip between Python and Rust, in memory and through each file
format: every type and length with edge values, then seeded random cases.

What each path keeps is fixed by its format: netCDF-4 and memory keep every type; netCDF-3
(classic) has no unsigned or 64-bit integers, nor lists of strings; Zarr's JSON has no integer
or float widths. A one-element list of strings is a plain `str`, booleans are `int8`, Python
numbers are `int64`/`float64`, as netCDF4-python and xarray have them.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import annoray as ar

NUMERIC = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]
TEXT = ["", "a", "Hα", 'quote " and \\ backslash', "NaN", "🙂 emoji", "x" * 5000]


def edges(dtype: str) -> list[Any]:
    if dtype.startswith("float"):
        f = np.finfo(dtype)
        return [0.0, -0.0, 1.5, float(f.max), float(f.min), float(f.tiny), np.nan, np.inf, -np.inf]
    i = np.iinfo(dtype)
    return [0, 1, int(i.max), int(i.min)]


def make(kind: str, values: list[Any], scalar: bool) -> Any:
    """The Python value of a case."""
    if kind == "str":
        return values[0]
    if kind == "bytes":
        return values[0].encode()
    if kind == "strs":
        return list(values)
    if kind == "bool":
        a = np.array(values, bool)
    elif kind == "pyint":
        return int(values[0])
    elif kind == "pyfloat":
        return float(values[0])
    else:
        a = np.array(values, kind)
    return a[0] if scalar else a


def cases() -> list[tuple[str, list[Any], bool]]:
    out: list[tuple[str, list[Any], bool]] = []
    for t in TEXT:
        out += [("str", [t], True), ("bytes", [t], True)]
    for n in [0, 1, 2, 5]:
        out.append(("strs", list(itertools.islice(itertools.cycle(TEXT), n)), False))
    for dtype in NUMERIC:
        e = edges(dtype)
        out += [(dtype, [v], True) for v in e]
        out += [(dtype, [], False), (dtype, e[:1], False), (dtype, e, False)]
    out += [("bool", [True], True), ("bool", [True, False, True], False)]
    out += [("pyint", [v], True) for v in [0, -(2**63), 2**63 - 1]]
    out += [("pyfloat", [v], True) for v in [0.25, np.nan, -np.inf]]
    rng = np.random.default_rng(20261029)
    for _ in range(300):
        kind = str(rng.choice(["strs", *NUMERIC, "bool"]))
        n = int(rng.integers(0, 6))
        if kind == "strs":
            values = [str(rng.choice(TEXT)) for _ in range(n)]
        elif kind == "bool":
            values = list(rng.integers(0, 2, n).astype(bool))
        else:
            e = edges(kind)
            # By index: `rng.choice` of Python ints past int64 would make them floats.
            values = [
                e[int(rng.integers(len(e)))] if rng.random() < 0.3 else draw(rng, kind)
                for _ in range(n)
            ]
        out.append((kind, values, n == 1 and kind != "strs" and bool(rng.random() < 0.5)))
    return out


def draw(rng: np.random.Generator, dtype: str) -> Any:
    if dtype.startswith("float"):
        return float(rng.normal() * 10.0 ** int(rng.integers(-30, 30)))
    i = np.iinfo(dtype)
    return int(rng.integers(i.min, i.max, endpoint=True, dtype=dtype))


def numbers(kind: str, values: list[Any]) -> np.ndarray:
    return np.array(
        values, {"bool": "int8", "pyint": "int64", "pyfloat": "float64"}.get(kind, kind)
    )


def expected(kind: str, values: list[Any], scalar: bool, path: str) -> Any:
    """What the attribute reads back as on `path`: a value, or an exception type."""
    if kind in ("str", "bytes"):
        return values[0]
    if kind == "strs":
        if len(values) == 1:
            return values[0]
        if path == "nc3":
            return ValueError
        return list(values)
    a = numbers(kind, values)
    one = scalar or kind in ("pyint", "pyfloat") or len(values) == 1
    if path == "nc3":
        to = {"uint8": "int16", "uint16": "int32", "uint32": "float64", "uint64": "float64"}.get(
            a.dtype.name
        )
        if a.dtype.name == "int64":
            fits = a.size == 0 or (
                a.min() >= np.iinfo("int32").min and a.max() <= np.iinfo("int32").max
            )
            to = "int32" if fits else "float64"
        a = a.astype(to) if to else a
    if path == "zarr":
        if a.size == 0:
            return []
        if a.dtype.kind == "f":
            a = a.astype("float64")
        elif a.dtype.kind == "u" and a.max() > np.iinfo("int64").max:
            a = a.astype("uint64")
        else:
            a = a.astype("int64")
    return a[0] if one else a


def same(got: Any, want: Any) -> bool:
    if isinstance(want, (str, list)):
        return type(got) is type(want) and got == want
    if isinstance(want, np.generic) != isinstance(got, np.generic):
        return False
    g, w = np.asarray(got), np.asarray(want)
    if g.dtype != w.dtype or g.shape != w.shape:
        return False
    return bool(np.array_equal(g, w, equal_nan=w.dtype.kind == "f"))


CASES = cases()
IDS = [f"{k}-{len(v)}-{i}" for i, (k, v, _) in enumerate(CASES)]


def write_read(path: str, value: Any, tmp: Path) -> Any:
    ds = ar.Dataset({"v": ("x", [1.0], {"a": value})}, attrs={"a": value})
    if path == "memory":
        got = [ds.attrs["a"], ds["v"].attrs["a"]]
    elif path == "xarray":
        back = ar.from_xarray(ds.to_xarray())
        got = [back.attrs["a"], back["v"].attrs["a"]]
    elif path == "zarr":
        ds.to_zarr(tmp / "a.zarr")
        back = ar.open_dataset(tmp / "a.zarr")
        got = [back.attrs["a"], back["v"].attrs["a"]]
    else:
        if path == "nc3":
            ds.to_netcdf(tmp / "a.nc", format="NETCDF3_CLASSIC")
        else:
            ds.to_netcdf(tmp / "a.nc")
        back = ar.open_dataset(tmp / "a.nc")
        got = [back.attrs["a"], back["v"].attrs["a"]]
    return got


@pytest.mark.parametrize("path", ["memory", "xarray", "nc4", "nc3", "zarr"])
def test_every_attribute_round_trips(path, tmp_path):
    failures = []
    for i, (kind, values, scalar) in enumerate(CASES):
        value = make(kind, values, scalar)
        want = expected(kind, values, scalar, path)
        case_dir = tmp_path / str(i)
        case_dir.mkdir()
        if want is ValueError:
            try:
                write_read(path, value, case_dir)
            except ValueError:
                continue
            failures.append(f"{IDS[i]}: {value!r} gave no ValueError")
            continue
        try:
            got = write_read(path, value, case_dir)
        except (ValueError, TypeError, OSError, OverflowError) as e:
            failures.append(f"{IDS[i]}: {value!r} raised {type(e).__name__}: {e}")
            continue
        for where, g in zip(["dataset", "variable"], got, strict=True):
            if not same(g, want):
                failures.append(f"{IDS[i]} ({where}): {value!r} came back {g!r}, expected {want!r}")
    assert not failures, f"{len(failures)} of {len(CASES)} cases:\n" + "\n".join(failures[:40])


def test_xarray_reads_annoray_zarr_attributes(tmp_path):
    """NaN and the infinities are written as zarr-python writes them, so xarray reads them as
    floats, and a string "NaN" stays a string."""
    attrs = {"nan": np.nan, "inf": np.inf, "list": np.array([1.0, -np.inf]), "s": "NaN"}
    ar.Dataset({"v": ("x", [1.0])}, attrs=attrs).to_zarr(tmp_path / "a.zarr")
    back = xr.open_zarr(tmp_path / "a.zarr").attrs
    assert np.isnan(back["nan"]) and back["inf"] == np.inf
    assert back["list"] == [1.0, -np.inf] and back["s"] == "NaN"
    # And annoray reads what zarr-python writes.
    xr.Dataset({"v": ("x", [1.0])}, attrs=attrs).to_zarr(tmp_path / "x.zarr")
    ours = ar.open_dataset(tmp_path / "x.zarr").attrs
    assert np.isnan(ours["nan"]) and ours["inf"] == np.inf and ours["s"] == "NaN"
    assert list(ours["list"]) == [1.0, -np.inf]
