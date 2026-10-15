"""xarray backend: ``xr.open_dataset(path, engine="annoray")``.

Files are opened by annoray's pure-Rust readers (netCDF-3, netCDF-4/HDF5), memory-mapped,
and read slice by slice on demand with the GIL released. The variables handed to xarray are
*raw* (as stored), so xarray's own CF decoder runs on top and results match its other
engines.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from typing import Any

import numpy as np
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    _normalize_path,
    datatree_from_dict_with_io_cleanup,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict, close_on_error
from xarray.core.variable import Variable

from annoray._annoray import RawGroup

_MAGIC = (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n")


def _basic_box(key: tuple, shape: tuple[int, ...]):
    """A basic-indexing key as the box to read plus the selection within it."""
    start, count, local = [], [], []
    for k, n in zip(key, shape, strict=True):
        if isinstance(k, slice):
            r = range(*k.indices(n))
            if len(r) == 0:
                start.append(0)
                count.append(0)
                local.append(slice(0, 0))
                continue
            lo, hi = min(r[0], r[-1]), max(r[0], r[-1]) + 1
            start.append(lo)
            count.append(hi - lo)
            local.append(
                slice(0, hi - lo, r.step) if r.step > 0 else slice(hi - lo - 1, None, r.step)
            )
        else:
            i = int(k)
            if i < 0:
                i += n
            if not 0 <= i < n:
                raise IndexError(f"index {k} is out of bounds for size {n}")
            start.append(i)
            count.append(1)
            local.append(0)
    return start, count, tuple(local)


class AnnorayArray(BackendArray):
    """One variable, read on demand: only the box a selection needs is read from the file."""

    def __init__(self, group: RawGroup, name: str, shape, dtype):
        self.group = group
        self.name = name
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key: tuple) -> np.ndarray:
        start, count, local = _basic_box(key, self.shape)
        if 0 in count:
            box = np.empty(count, dtype=self.dtype)
        else:
            box = self.group.read(self.name, start, count)
            if self.dtype.kind == "S":
                # netCDF characters arrive as bytes.
                box = box.view(self.dtype)
        return np.asarray(box[local])


class AnnorayStore(AbstractDataStore):
    """A group of a file opened by annoray, serving raw variables."""

    def __init__(self, group: RawGroup, filename: str):
        self._group: RawGroup | None = group
        self._filename = filename
        self.lock = threading.Lock()  # unused: reads are thread-safe in Rust

    @classmethod
    def open(cls, filename, group: str | None = None) -> AnnorayStore:
        filename = _normalize_path(filename)
        if not isinstance(filename, str | os.PathLike):
            raise TypeError("the annoray engine reads files by path")
        return cls(RawGroup.open(os.fspath(filename), _strip(group)), os.fspath(filename))

    def open_store_variable(self, name: str) -> Variable:
        info = self.group.info(name)
        data = indexing.LazilyIndexedArray(
            AnnorayArray(self.group, name, info["shape"], info["dtype"])
        )
        encoding: dict[str, Any] = {
            "dtype": str if info["dtype"] == "object" else np.dtype(info["dtype"])
        }
        if info["chunks"] is not None:
            encoding["contiguous"] = False
            encoding["chunksizes"] = tuple(info["chunks"])
            encoding["preferred_chunks"] = dict(zip(info["dims"], info["chunks"], strict=True))
        else:
            encoding["contiguous"] = True
            encoding["chunksizes"] = None
        encoding["zlib"] = info["deflate"] is not None
        encoding["complevel"] = info["deflate"] or 0
        if info["compression"] is not None:
            encoding["compression"] = info["compression"]
        encoding["shuffle"] = bool(info["shuffle"])
        encoding["source"] = self._filename
        encoding["original_shape"] = tuple(info["shape"])
        return Variable(info["dims"], data, info["attrs"], encoding)

    def get_variables(self):
        return FrozenDict((k, self.open_store_variable(k)) for k in self.group.variables)

    def get_attrs(self):
        return FrozenDict(self.group.attrs)

    def get_dimensions(self):
        return FrozenDict((k, n) for k, (n, _) in self.group.dims.items())

    def get_encoding(self):
        return {"unlimited_dims": {k for k, (_, u) in self.group.dims.items() if u}}

    @property
    def group(self) -> RawGroup:
        if self._group is None:
            raise ValueError(f"{self._filename}: the file is closed")
        return self._group

    def close(self):
        # Dropping the last reference unmaps the file.
        self._group = None


def _strip(group: str | None) -> str | None:
    return group.strip("/") or None if group else None


def _iter_groups(group: RawGroup, prefix: str = "/"):
    yield prefix, group
    for name in group.groups:
        yield from _iter_groups(group.child(name), prefix.rstrip("/") + "/" + name)


class AnnorayBackendEntrypoint(BackendEntrypoint):
    """Open netCDF-3, netCDF-4 and HDF5 files with annoray's pure-Rust readers."""

    description = "Open netCDF and HDF5 files with annoray (pure Rust, no C libraries)"
    url = "https://github.com/sunipkm/xarray-rs"
    open_dataset_parameters = (
        "filename_or_obj",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "decode_coords",
        "drop_variables",
        "use_cftime",
        "decode_timedelta",
        "group",
    )

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
        group: str | None = None,
    ):
        store = AnnorayStore.open(filename_or_obj, group)
        with close_on_error(store):
            return StoreBackendEntrypoint().open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )

    def open_groups_as_dict(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
        group: str | None = None,
        **kwargs,
    ):
        root = AnnorayStore.open(filename_or_obj, group)
        out = {}
        for path, g in _iter_groups(root.group):
            store = AnnorayStore(g, root._filename)
            with close_on_error(store):
                out[path] = StoreBackendEntrypoint().open_dataset(
                    store,
                    mask_and_scale=mask_and_scale,
                    decode_times=decode_times,
                    concat_characters=concat_characters,
                    decode_coords=decode_coords,
                    drop_variables=drop_variables,
                    use_cftime=use_cftime,
                    decode_timedelta=decode_timedelta,
                )
        return out

    def open_datatree(self, filename_or_obj, **kwargs):
        return datatree_from_dict_with_io_cleanup(
            self.open_groups_as_dict(filename_or_obj, **kwargs)
        )

    def guess_can_open(self, filename_or_obj) -> bool:
        try:
            filename_or_obj = _normalize_path(filename_or_obj)
            if isinstance(filename_or_obj, str | os.PathLike):
                with open(filename_or_obj, "rb") as f:
                    return f.read(8).startswith(_MAGIC)
        except (OSError, TypeError):
            pass
        return False
