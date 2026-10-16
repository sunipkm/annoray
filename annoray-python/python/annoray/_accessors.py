"""``register_dataarray_accessor`` / ``register_dataset_accessor``: custom namespaces
(``da.myaccessor.method()``), as xarray's. An accessor is made on first access and cached
for the object's lifetime; a name that is already an attribute (or, for datasets, shadows a
variable) is registered with a warning, and the built-in attribute keeps precedence."""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, dict[str, type]] = {"DataArray": {}, "Dataset": {}}
_CACHE: dict[int, dict[str, Any]] = {}


class AccessorRegistrationWarning(Warning):
    """A registered accessor name is already taken."""


def _register(kind: str, cls: type, name: str) -> Callable[[type], type]:
    def decorator(accessor: type) -> type:
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {accessor!r} under name {name!r} for type {cls!r} "
                "is overriding a preexisting attribute with the same name.",
                AccessorRegistrationWarning,
                stacklevel=2,
            )
        _REGISTRY[kind][name] = accessor
        return accessor

    return decorator


def register_dataarray_accessor(name: str) -> Callable[[type], type]:
    """Register a custom accessor on DataArray objects: ``@register_dataarray_accessor("geo")``
    then ``da.geo``; the class is called with the array."""
    from annoray._annoray import DataArray

    return _register("DataArray", DataArray, name)


def register_dataset_accessor(name: str) -> Callable[[type], type]:
    """Register a custom accessor on Dataset objects (see
    ``register_dataarray_accessor``)."""
    from annoray._annoray import Dataset

    return _register("Dataset", Dataset, name)


def lookup(obj: Any, kind: str, name: str) -> Any:
    """The accessor ``name`` of ``obj`` (made once per object), or ``None``."""
    accessor = _REGISTRY[kind].get(name)
    if accessor is None:
        return None
    key = id(obj)
    cached = _CACHE.get(key)
    if cached is None:
        cached = _CACHE[key] = {}
        weakref.finalize(obj, _CACHE.pop, key, None)
    if name not in cached:
        try:
            cached[name] = accessor(obj)
        except AttributeError as e:
            # As xarray: an AttributeError inside __init__ would be read as a missing attribute.
            raise RuntimeError(f"error initializing {name!r} accessor.") from e
    return cached[name]
