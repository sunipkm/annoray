"""``.loc``: label-based indexing with positional or named keys (xarray's ``.loc``)."""

from __future__ import annotations

from typing import Any


class LocIndexer:
    """``obj.loc[key]``: ``obj.sel`` with the key given by position (``da.loc[10.0, "a"]``,
    ``da.loc[..., 2.0]``, slices of labels) or as a mapping of dimension names
    (``ds.loc[{"x": 10.0}]``). Read only: annoray objects are immutable."""

    __slots__ = ("_obj",)

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, dict):
            return self._obj.sel(key)
        from annoray import Dataset

        if isinstance(self._obj, Dataset):
            raise TypeError("Dataset.loc takes a mapping of dimension names to labels")
        dims = self._obj.dims
        items = key if isinstance(key, tuple) else (key,)
        n_ellipsis = sum(k is Ellipsis for k in items)
        if n_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if n_ellipsis:
            i = next(j for j, k in enumerate(items) if k is Ellipsis)
            fill = (slice(None),) * (len(dims) - len(items) + 1)
            items = items[:i] + fill + items[i + 1 :]
        if len(items) > len(dims):
            raise IndexError(f"too many indices: {len(items)} for {len(dims)} dimensions {dims}")
        indexers = {
            d: k for d, k in zip(dims, items) if not (isinstance(k, slice) and k == slice(None))
        }
        return self._obj.sel(indexers)

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError(
            "annoray objects are immutable: assign through where(...), copy(data=...) or "
            "assign(...)"
        )
