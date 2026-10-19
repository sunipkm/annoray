"""xarray's opening and writing options on top of annoray's readers and writers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Any, Literal

from annoray._annoray import DataArray, Dataset
from annoray._annoray import open_dataset as _open_dataset
from annoray._annoray import open_mfdataset as _open_mfdataset

# Formats annoray reads itself: any of these engines (or none) opens the file natively.
_ENGINES = {None, "annoray", "netcdf4", "h5netcdf", "scipy", "h5py", "zarr"}
# Options that only matter to dask or xarray's file cache: accepted with no effect.
_IGNORED = {"chunks", "cache", "lock", "inline_array", "chunked_array_type", "from_array_kwargs"}


def _check(engine: Any, kwargs: dict[str, Any]) -> None:
    if engine not in _ENGINES:
        raise ValueError(
            f"engine {engine!r}: annoray reads netCDF-3/4, HDF5 and Zarr itself; other formats "
            "are not supported"
        )
    unknown = set(kwargs) - _IGNORED - {"backend_kwargs"}
    if unknown:
        raise TypeError(f"open_dataset() got unexpected keyword arguments {sorted(unknown)}")


def _use_cftime(ds: Dataset) -> Dataset:
    """Standard-calendar time coordinates as CF dates (``use_cftime=True``)."""
    for d in ds.dims:
        c = ds.coords.get(d)
        if c is not None and c.dims == (d,) and c.dtype.kind == "M":
            ds = ds.convert_calendar("standard", dim=d, use_cftime=True)
    return ds


def open_dataset(
    filename_or_obj: str | os.PathLike[str] | bytes | IO[bytes],
    *,
    engine: str | None = None,
    group: str | None = None,
    decode_cf: bool | None = None,
    mask_and_scale: bool | None = None,
    decode_times: bool | None = None,
    decode_timedelta: bool | None = None,
    use_cftime: bool | None = None,
    concat_characters: bool | None = None,
    decode_coords: bool | Literal["coordinates", "all"] | None = None,
    drop_variables: str | Sequence[str] | None = None,
    decode_units: bool = True,
    lazy: bool = True,
    **kwargs: Any,
) -> Dataset:
    """Open a netCDF-3/4, HDF5 file or Zarr store lazily and CF-decode it (xarray's
    ``open_dataset``). ``decode_cf=False`` turns every decoding step off; ``chunks`` and the
    other dask options are accepted and have no effect (data is read lazily, chunk by
    chunk, without dask); ``use_cftime=True`` gives CF dates for standard-calendar time
    coordinates; ``decode_units`` (annoray) parses ``units`` attributes into units."""
    _check(engine, kwargs)
    if decode_timedelta is False:
        raise NotImplementedError("decode_timedelta=False")
    default = decode_cf is not False

    def flag(v: Any) -> bool:
        return default if v is None else bool(v)

    ds = _open_dataset(
        filename_or_obj,
        group=group,
        mask_and_scale=flag(mask_and_scale),
        decode_times=flag(decode_times),
        decode_units=decode_units and default,
        decode_coords=decode_coords if isinstance(decode_coords, str) else flag(decode_coords),
        concat_characters=flag(concat_characters),
        lazy=lazy,
    )
    if drop_variables is not None:
        ds = ds.drop_vars(drop_variables, errors="ignore")
    if use_cftime:
        ds = _use_cftime(ds)
    return ds


def load_dataset(filename_or_obj: Any, **kwargs: Any) -> Dataset:
    """``open_dataset`` with every variable read into memory."""
    return open_dataset(filename_or_obj, **kwargs).load()


def load_dataarray(filename_or_obj: Any, **kwargs: Any) -> DataArray:
    """``open_dataarray`` with the data read into memory."""
    from annoray._functions import open_dataarray

    return open_dataarray(filename_or_obj, **kwargs).load()


def open_mfdataset(
    paths: Any,
    *,
    combine: str = "by_coords",
    concat_dim: str | None = None,
    preprocess: Callable[[Dataset], Dataset] | None = None,
    engine: str | None = None,
    parallel: bool = False,
    join: str = "outer",
    lazy: bool = True,
    **kwargs: Any,
) -> Dataset:
    """Open several files as one dataset (xarray's ``open_mfdataset``): a glob pattern or
    a list of paths, combined by their coordinates (``combine="by_coords"``) or
    concatenated along ``concat_dim`` in the order given (``combine="nested"``).
    ``preprocess`` is applied to each dataset first; ``parallel`` and the dask options are
    accepted with no effect."""
    _check(engine, {k: v for k, v in kwargs.items() if k not in ("data_vars", "coords", "compat")})
    if combine not in ("by_coords", "nested"):
        raise ValueError(f"combine must be 'by_coords' or 'nested', not {combine!r}")
    if combine == "nested" and concat_dim is None:
        raise ValueError("combine='nested' needs concat_dim")
    if preprocess is None:
        return _open_mfdataset(
            paths, concat_dim=concat_dim if combine == "nested" else None, lazy=lazy
        )
    import glob

    from annoray import combine_by_coords
    from annoray._functions import combine_nested

    files = sorted(glob.glob(str(paths))) if isinstance(paths, (str, os.PathLike)) else list(paths)
    if not files:
        raise OSError(f"no files match {paths!r}")
    parts = [preprocess(open_dataset(f, lazy=lazy)) for f in files]
    if combine == "nested":
        return combine_nested(parts, concat_dim=concat_dim, join=join)  # pyright: ignore[reportArgumentType]
    return combine_by_coords(parts)


def append_netcdf(ds: Dataset, path: Any, **kwargs: Any) -> None:
    """``to_netcdf(mode="a")``: the file's variables, with ``ds``'s added or replaced (and
    its attributes updated), written to a temporary file next to it, then moved over it, so
    a failed write leaves the file as it was."""
    path = Path(path)
    if not path.exists():
        ds.to_netcdf(path, **kwargs)
        return
    merged = open_dataset(path).load()
    for k, c in ds.coords.items():
        merged.coords[k] = c
    for k, v in ds.data_vars.items():
        merged[k] = v
    merged.attrs.update(dict(ds.attrs))
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        merged.to_netcdf(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
