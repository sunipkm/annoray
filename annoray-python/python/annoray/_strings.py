"""``DataArray.str``: vectorised string methods on text data (xarray's ``StringAccessor``).

Results keep the array's dimensions, coordinates, name and attributes; methods that produce
several values per element (``split`` with ``dim=``, ``partition``, ``extract`` and
``extractall``, ``get_dummies``) add a dimension, padded with ``""`` as in xarray. Text has
no bytes form in annoray, so ``encode``/``decode`` are refused, and so are results that are
lists per element (``findall``, ``split(dim=None)``). Data is read.

Deviation: ``extractall`` labels its dimensions by what they index — ``group_dim`` the
capture groups, ``match_dim`` the matches. xarray swaps the two (its ``group_dim`` axis
indexes matches).
"""

from __future__ import annotations

import re
import textwrap
import unicodedata
from collections.abc import Callable
from typing import Any

import numpy as np

from annoray._annoray import DataArray


class StringAccessor:
    __slots__ = ("_da",)

    def __init__(self, da: DataArray) -> None:
        if np.asarray(da.values).dtype.kind not in "OU":
            raise TypeError(".str needs text data")
        self._da = da

    # ----- plumbing -----

    def _values(self) -> np.ndarray:
        return np.asarray(self._da.values, dtype=object)

    def _wrap(self, values: Any, extra: tuple[str, ...] = (), coords: Any = None) -> DataArray:
        da = self._da
        c = dict(da.coords)
        if coords:
            c.update(coords)
        return DataArray(
            np.asarray(values),
            dims=(*da.dims, *extra),
            coords=c,
            name=da.name,
            attrs=dict(da.attrs),
        )

    def _map(self, f: Callable[[str], Any], dtype: Any = object) -> DataArray:
        v = self._values()
        out = np.empty(v.shape, dtype=dtype)
        for i, s in np.ndenumerate(v):
            out[i] = f(s)
        return self._wrap(out)

    def _spread(self, f: Callable[[str], list[str]], dim: str, right: bool = False) -> DataArray:
        """A new trailing dimension ``dim`` holding ``f``'s list per element, padded with
        ``""`` (on the left when ``right``-aligned)."""
        v = self._values()
        parts = {i: f(s) for i, s in np.ndenumerate(v)}
        n = max((len(p) for p in parts.values()), default=0)
        out = np.full((*v.shape, n), "", dtype=object)
        for i, p in parts.items():
            offset = n - len(p) if right else 0
            for j, s in enumerate(p):
                out[(*i, offset + j)] = s
        return self._wrap(out, (dim,))

    @staticmethod
    def _flags(case: bool | None, flags: int) -> int:
        return flags | (re.IGNORECASE if case is False else 0)

    # ----- one value per element -----

    def __getitem__(self, key: int | slice) -> DataArray:
        if isinstance(key, slice):
            return self.slice(key.start, key.stop, key.step)
        return self.get(key)

    def len(self) -> DataArray:
        return self._map(len, np.int64)

    def get(self, i: int, default: str = "") -> DataArray:
        return self._map(lambda s: s[i] if -len(s) <= i < len(s) else default)

    def slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> DataArray:
        return self._map(lambda s: s[start:stop:step])

    def slice_replace(
        self, start: int | None = None, stop: int | None = None, repl: str = ""
    ) -> DataArray:
        def f(s: str) -> str:
            stop_ = len(s[:stop]) if stop is not None else len(s)
            start_ = len(s[:start]) if start is not None else 0
            return s[:start_] + repl + s[max(stop_, start_) :]

        return self._map(f)

    def upper(self) -> DataArray:
        return self._map(str.upper)

    def lower(self) -> DataArray:
        return self._map(str.lower)

    def title(self) -> DataArray:
        return self._map(str.title)

    def capitalize(self) -> DataArray:
        return self._map(str.capitalize)

    def swapcase(self) -> DataArray:
        return self._map(str.swapcase)

    def casefold(self) -> DataArray:
        return self._map(str.casefold)

    def normalize(self, form: Any) -> DataArray:
        return self._map(lambda s: unicodedata.normalize(form, s))

    def strip(self, to_strip: str | None = None, side: str = "both") -> DataArray:
        f = {"both": str.strip, "left": str.lstrip, "right": str.rstrip}[side]
        return self._map(lambda s: f(s, to_strip))

    def lstrip(self, to_strip: str | None = None) -> DataArray:
        return self.strip(to_strip, "left")

    def rstrip(self, to_strip: str | None = None) -> DataArray:
        return self.strip(to_strip, "right")

    def pad(self, width: int, side: str = "left", fillchar: str = " ") -> DataArray:
        if side == "left":
            return self._map(lambda s: s.rjust(width, fillchar))
        if side == "right":
            return self._map(lambda s: s.ljust(width, fillchar))
        if side == "both":
            return self._map(lambda s: s.center(width, fillchar))
        raise ValueError(f"side must be 'left', 'right' or 'both', not {side!r}")

    def center(self, width: int, fillchar: str = " ") -> DataArray:
        return self.pad(width, "both", fillchar)

    def ljust(self, width: int, fillchar: str = " ") -> DataArray:
        return self.pad(width, "right", fillchar)

    def rjust(self, width: int, fillchar: str = " ") -> DataArray:
        return self.pad(width, "left", fillchar)

    def zfill(self, width: int) -> DataArray:
        return self._map(lambda s: s.zfill(width))

    def repeat(self, repeats: int) -> DataArray:
        return self._map(lambda s: s * repeats)

    def wrap(self, width: int, **kwargs: Any) -> DataArray:
        return self._map(lambda s: "\n".join(textwrap.wrap(s, width, **kwargs)))

    def translate(self, table: dict[int, Any]) -> DataArray:
        return self._map(lambda s: s.translate(table))

    def format(self, *args: Any, **kwargs: Any) -> DataArray:
        return self._map(lambda s: s.format(*args, **kwargs))

    def startswith(self, pat: str) -> DataArray:
        return self._map(lambda s: s.startswith(pat), bool)

    def endswith(self, pat: str) -> DataArray:
        return self._map(lambda s: s.endswith(pat), bool)

    def contains(
        self, pat: str, case: bool | None = None, flags: int = 0, regex: bool = True
    ) -> DataArray:
        if not regex:
            if case is False:
                return self._map(lambda s: pat.casefold() in s.casefold(), bool)
            return self._map(lambda s: pat in s, bool)
        r = re.compile(pat, self._flags(case, flags))
        return self._map(lambda s: r.search(s) is not None, bool)

    def match(self, pat: str, case: bool | None = None, flags: int = 0) -> DataArray:
        r = re.compile(pat, self._flags(case, flags))
        return self._map(lambda s: r.match(s) is not None, bool)

    def count(self, pat: str, flags: int = 0, case: bool | None = None) -> DataArray:
        r = re.compile(pat, self._flags(case, flags))
        return self._map(lambda s: len(r.findall(s)), np.int64)

    def find(
        self, sub: str, start: int = 0, end: int | None = None, side: str = "left"
    ) -> DataArray:
        f = str.find if side == "left" else str.rfind
        return self._map(lambda s: f(s, sub, start, end), np.int64)

    def rfind(self, sub: str, start: int = 0, end: int | None = None) -> DataArray:
        return self.find(sub, start, end, side="right")

    def index(
        self, sub: str, start: int = 0, end: int | None = None, side: str = "left"
    ) -> DataArray:
        f = str.index if side == "left" else str.rindex
        return self._map(lambda s: f(s, sub, start, end), np.int64)

    def rindex(self, sub: str, start: int = 0, end: int | None = None) -> DataArray:
        return self.index(sub, start, end, side="right")

    def replace(
        self,
        pat: str,
        repl: str | Callable[[re.Match[str]], str],
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = True,
    ) -> DataArray:
        if not regex:
            if callable(repl):
                raise ValueError("a callable repl needs regex=True")
            return self._map(lambda s: s.replace(pat, repl, n))
        r = re.compile(pat, self._flags(case, flags))
        return self._map(lambda s: r.sub(repl, s, count=max(n, 0)))

    def cat(self, *others: Any, sep: str = "") -> DataArray:
        from annoray import broadcast

        arrays = broadcast(self._da, *[o for o in others if isinstance(o, DataArray)])
        parts = [np.asarray(a.values, dtype=object) for a in arrays]
        constants = [o for o in others if not isinstance(o, DataArray)]
        first = arrays[0]
        out = np.empty(parts[0].shape, dtype=object)
        for i in np.ndindex(out.shape):
            out[i] = sep.join([*(str(p[i]) for p in parts), *map(str, constants)])
        return DataArray(
            out,
            dims=first.dims,
            coords=dict(first.coords),
            name=first.name,
            attrs=dict(first.attrs),
        )

    def join(self, dim: str | None = None, sep: str = "") -> DataArray:
        da = self._da
        if dim is None:
            if da.ndim != 1:
                raise ValueError("join needs dim= for an array of several dimensions")
            dim = da.dims[0]
        axis = da.dims.index(dim)
        v = np.moveaxis(self._values(), axis, -1)
        out = np.empty(v.shape[:-1], dtype=object)
        for i in np.ndindex(out.shape):
            out[i] = sep.join(v[i])
        dims = tuple(d for d in da.dims if d != dim)
        coords = {k: c for k, c in da.coords.items() if dim not in c.dims}
        return DataArray(out, dims=dims, coords=coords, name=da.name, attrs=dict(da.attrs))

    def isalpha(self) -> DataArray:
        return self._map(str.isalpha, bool)

    def isalnum(self) -> DataArray:
        return self._map(str.isalnum, bool)

    def isdigit(self) -> DataArray:
        return self._map(str.isdigit, bool)

    def isdecimal(self) -> DataArray:
        return self._map(str.isdecimal, bool)

    def isnumeric(self) -> DataArray:
        return self._map(str.isnumeric, bool)

    def islower(self) -> DataArray:
        return self._map(str.islower, bool)

    def isupper(self) -> DataArray:
        return self._map(str.isupper, bool)

    def isspace(self) -> DataArray:
        return self._map(str.isspace, bool)

    def istitle(self) -> DataArray:
        return self._map(str.istitle, bool)

    # ----- several values per element: a new dimension -----

    def split(self, dim: str | None, sep: str | None = None, maxsplit: int = -1) -> DataArray:
        if dim is None:
            raise TypeError(
                "split(dim=None) gives lists per element, which annoray cannot hold; give dim="
            )
        return self._spread(lambda s: s.split(sep, maxsplit), dim)

    def rsplit(self, dim: str | None, sep: str | None = None, maxsplit: int = -1) -> DataArray:
        if dim is None:
            raise TypeError(
                "rsplit(dim=None) gives lists per element, which annoray cannot hold; give dim="
            )
        return self._spread(lambda s: s.rsplit(sep, maxsplit), dim, right=True)

    def partition(self, dim: str | None = "partition", sep: str = " ") -> DataArray:
        if dim is None:
            raise TypeError("partition(dim=None) gives tuples per element; give dim=")
        return self._spread(lambda s: list(s.partition(sep)), dim)

    def rpartition(self, dim: str | None = "partition", sep: str = " ") -> DataArray:
        if dim is None:
            raise TypeError("rpartition(dim=None) gives tuples per element; give dim=")
        return self._spread(lambda s: list(s.rpartition(sep)), dim)

    def extract(
        self, pat: str, dim: str | None, case: bool | None = None, flags: int = 0
    ) -> DataArray:
        r = re.compile(pat, self._flags(case, flags))
        if dim is None:
            if r.groups != 1:
                raise ValueError("extract(dim=None) needs a pattern with exactly one group")
            return self._map(lambda s: (m.group(1) or "") if (m := r.search(s)) else "")
        return self._spread(
            lambda s: [g or "" for g in m.groups()] if (m := r.search(s)) else [""] * r.groups, dim
        )

    def extractall(
        self, pat: str, group_dim: str, match_dim: str, case: bool | None = None, flags: int = 0
    ) -> DataArray:
        r = re.compile(pat, self._flags(case, flags))
        v = self._values()
        found = {
            i: [[g or "" for g in m.groups()] for m in r.finditer(s)] for i, s in np.ndenumerate(v)
        }
        n = max((len(f) for f in found.values()), default=0)
        out = np.full((*v.shape, r.groups, n), "", dtype=object)
        for i, matches in found.items():
            for j, groups in enumerate(matches):
                for k, g in enumerate(groups):
                    out[(*i, k, j)] = g
        return self._wrap(out, (group_dim, match_dim))

    def get_dummies(self, dim: str, sep: str = "|") -> DataArray:
        v = self._values()
        tokens = sorted({t for s in v.ravel() for t in s.split(sep) if t})
        out = np.zeros((*v.shape, len(tokens)), dtype=bool)
        for i, s in np.ndenumerate(v):
            present = set(s.split(sep))
            for j, t in enumerate(tokens):
                out[(*i, j)] = t in present
        return self._wrap(out, (dim,), coords={dim: (dim, tokens)})

    def findall(self, *args: Any, **kwargs: Any) -> DataArray:
        raise TypeError(
            "findall gives lists per element, which annoray cannot hold; use extractall"
        )

    def encode(self, *args: Any, **kwargs: Any) -> DataArray:
        raise TypeError("annoray has no bytes data type; text stays text")

    def decode(self, *args: Any, **kwargs: Any) -> DataArray:
        raise TypeError("annoray has no bytes data type; text is already decoded")
