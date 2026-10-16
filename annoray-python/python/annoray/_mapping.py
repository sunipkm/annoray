"""Dicts that write changes back to the DataArray or Dataset they came from: ``attrs``
(``da.attrs["k"] = v``) and ``coords`` (``ds.coords["x"] = …``).

They are ``dict`` subclasses, so ``json.dumps``, ``isinstance(…, dict)`` and copying behave
as for plain dicts; a pickle or ``copy()`` is a plain, detached ``dict``. Each change is
applied to the owner first and the dict refreshed from it, so a refused change leaves both
as they were.
"""

from __future__ import annotations

from typing import Any


class _LiveDict(dict):  # pyright: ignore[reportMissingTypeArgument]
    __slots__ = ("_owner",)

    def __init__(self, owner: Any, data: dict[str, Any]) -> None:
        super().__init__(data)
        self._owner = owner

    def _refresh(self) -> None:
        raise NotImplementedError

    def __reduce__(self) -> Any:
        return (dict, (dict(self),))

    def copy(self) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return dict(self)

    def __ior__(self, other: Any) -> Any:
        self.update(other)
        return self

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key: str, *default: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        value = self[key]
        del self[key]
        return value

    def popitem(self) -> tuple[str, Any]:
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(self))
        return key, self.pop(key)

    def clear(self) -> None:
        for key in list(self):
            del self[key]


class AttrsDict(_LiveDict):
    """``obj.attrs``: changes apply to ``obj`` (and, for an array taken from a dataset, to
    that dataset's variable). Only the keys changed are converted; the others keep their
    exact types (a file's ``int16`` or ``float32`` attribute is written back as it was)."""

    __slots__ = ()

    def _edit(self, changed: dict[str, Any], removed: list[str]) -> None:
        self._owner._edit_attrs(changed, removed)
        self._refresh()

    def _refresh(self) -> None:
        dict.clear(self)
        dict.update(self, self._owner._plain_attrs())

    def __setitem__(self, key: str, value: Any) -> None:
        self._edit({key: value}, [])

    def __delitem__(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        self._edit({}, [key])

    def update(self, *args: Any, **kwargs: Any) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self._edit(dict(*args, **kwargs), [])

    def clear(self) -> None:
        self._edit({}, list(self))


class EncodingDict(AttrsDict):
    """``obj.encoding``: changes apply to ``obj`` (and to the dataset variable an array came
    from). Keys are xarray's (netCDF4-python's) names."""

    __slots__ = ()

    def _edit(self, changed: dict[str, Any], removed: list[str]) -> None:
        self._owner._edit_encoding(changed, removed)
        self._refresh()

    def _refresh(self) -> None:
        dict.clear(self)
        dict.update(self, self._owner._plain_encoding())


class CoordsDict(_LiveDict):
    """``obj.coords``: ``obj.coords[name] = value`` adds or replaces a coordinate, ``del``
    removes one."""

    __slots__ = ()

    def _refresh(self) -> None:
        dict.clear(self)
        dict.update(self, self._owner._plain_coords())

    def __setitem__(self, key: str, value: Any) -> None:
        self._owner._set_coord(key, value)
        self._refresh()

    def __delitem__(self, key: str) -> None:
        self._owner._del_coord(key)
        self._refresh()

    def update(self, *args: Any, **kwargs: Any) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        for key, value in dict(*args, **kwargs).items():
            self._owner._set_coord(key, value)
        self._refresh()
