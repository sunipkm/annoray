---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Reading and writing files

annoray reads and writes netCDF-3, netCDF-4/HDF5 and Zarr with its own pure-Rust code: no
netCDF-C, HDF5 or plugin libraries are involved. This chapter covers what is read from
netCDF-4 and HDF5 files, how values are encoded on the way back, and how a file is written
in place: streamed while its values are produced, appended to, or rewritten a region at a
time.

```{code-cell} python
import io
import tempfile
from pathlib import Path

import numpy as np
import annoray as ar

tmp = Path(tempfile.mkdtemp())
```

## Opening

`open_dataset` takes a path, the bytes of a file (`bytes`, `bytearray`, `memoryview`) or a
binary file-like object, which is read whole from its start, as xarray does. The format is
recognised from the contents. Values are read lazily: opening reads the metadata and the
dimension coordinates, and other values are read when they are used.

```{code-cell} python
ds = ar.Dataset(
    {"t": (("time", "x"), np.arange(6.0).reshape(3, 2), {"units": "K"})},
    coords={"x": ("x", [10.0, 20.0])},
)
ds.to_netcdf(tmp / "small.nc")
data = (tmp / "small.nc").read_bytes()
print(ar.open_dataset(data)["t"].values)
print(ar.open_dataset(io.BytesIO(data))["t"].units)
```

### Coordinates named by CF attributes

`decode_coords="all"` also makes coordinates of the variables that CF attributes such as
`bounds`, `grid_mapping`, `cell_measures` and `formula_terms` name; the attributes move to
the variables' `encoding` and are written back from there. An attribute that names a
variable the file does not have stays an attribute, with a warning (xarray drops it).

```{code-cell} python
cf = ar.Dataset(
    {
        "v": (("x",), np.zeros(3), {"grid_mapping": "crs"}),
        "crs": ((), 0, {"grid_mapping_name": "latitude_longitude"}),
    },
    coords={"x": ("x", [1.0, 2.0, 3.0], {"bounds": "x_bnds"})},
)
cf = cf.assign({"x_bnds": (("x", "nv"), np.zeros((3, 2)))})
cf.to_netcdf(tmp / "cf.nc")
opened = ar.open_dataset(tmp / "cf.nc", decode_coords="all")
print(sorted(opened.coords), opened["v"].encoding["grid_mapping"])
```

### Writing over an open file

Files are opened by mapping them into memory, so a dataset reads from its file for as long as
it is open. Writing to a path that already holds a file (with `to_netcdf`, or
`NetcdfWriter.create`) therefore never truncates it: the new file is written beside it and
renamed over it once complete. Datasets opened from the old file keep reading the old data,
a dataset can be written back to the file it was opened from, and a failed write leaves the
old file as it was. A file reached through a symbolic link is replaced where the link points,
and keeps its permissions.

```{code-cell} python
ds = ar.open_dataset(tmp / "small.nc")
ds.assign(t2=ds["t"] * 2.0).to_netcdf(tmp / "small.nc")  # back over its own file
print(list(ds.data_vars), list(ar.open_dataset(tmp / "small.nc").data_vars))
```

### Files from untrusted sources

The readers check every offset and length against the file, so a corrupt or hostile file
gives an error, not a crash. A small file can still declare large arrays, though, or
chunks that a compressor expands many times over. `limits=` on `open_dataset`,
`open_datatree` and `open_mfdataset` sets ceilings on what it may make annoray allocate. The
file's metadata is checked before any values are read, and every read after that.

| Limit | What it bounds |
|---|---|
| `max_read_bytes` | One read: a variable read whole, a slice of a lazy one, or one stored chunk (decoded whole for any read touching it) |
| `max_total_bytes` | The values all variables of the file declare together |
| `max_elements` | The values of one variable |
| `max_rank` | The dimensions of one variable |
| `max_variables` | The variables of the file, in all groups |
| `max_attributes` | The attributes of one variable or group |

`limits="untrusted"` sets them all: reads of at most 256 MiB, 1 GiB of values, 2²⁸ values per
variable, 32 dimensions, and 10,000 variables and 10,000 attributes per variable or group. A
dict sets some (`None` for no limit), and starts from that preset with `"preset": "untrusted"`.
A breach is `ReadLimitError` (a `ValueError`), except for a netCDF-4 string, compound or
variable-length variable over `max_read_bytes`: these are read when the file is opened, so
such a variable is skipped with a warning, as variables that cannot be read are. Sizes are
those of the values in memory.

```{code-cell} python
grid = ar.Dataset({"t": (("y", "x"), np.zeros((500, 400)))})
grid.to_netcdf(tmp / "grid.nc")
opened = ar.open_dataset(tmp / "grid.nc", limits={"preset": "untrusted", "max_read_bytes": 2**20})
print(float(opened["t"].isel(y=0).sum()))  # a slice of 3.2 kB reads
try:
    opened["t"].load()  # all 1.6 MB does not
except ar.ReadLimitError as e:
    print("ReadLimitError:", e)
```

## What is read from netCDF-4 and HDF5

| In the file | In annoray |
|---|---|
| Complex numbers | `complex64`/`complex128` values, from a compound of two floats named `r`/`i` (h5py, netCDF4-python's nc-complex) or `real`/`imag`; written as h5py's `r`/`i` compound. They are read when the file is opened, not lazily (Zarr's complex arrays are read lazily). netCDF-3 has no complex type. |
| Compound (record) variables | One variable per field, `var.field` (nested fields dotted, array members with dimensions of their own); written back as one compound variable when every field is written. |
| Variable-length (ragged) variables | A CF contiguous ragged array: the values along `var_obs`, the lengths in `var_count`; written back as a variable-length variable. |
| Enumerations | The integer values, with the names in `encoding["dtype"].metadata` (`enum`, `enum_name`), as in xarray; written back as the same type. h5py's booleans read as booleans. |
| Virtual datasets | Gathered from their sources (in the same file or others, found relative to it); a missing source file is an error. |
| Soft and external links | Followed to datasets; links to groups, dangling and cyclic links are skipped with a warning. |
| Half-precision floats | `float32` values with `encoding["dtype"] = "float16"`. |
| Attributes of any size | Large attributes are kept in dense storage, and written there. |

Compressed chunks are decoded for every filter in common use. Writing uses the filters with
an encoder in this table.

| Filter | Read | Write |
|---|---|---|
| deflate (zlib), shuffle, Fletcher-32 | yes | yes |
| Zstandard, LZ4, Blosc (LZ4, zlib, Zstandard) | yes | yes |
| Blosc with BloscLZ, bzip2, LZF, Bitshuffle | yes | no |
| n-bit, scale-offset, szip | yes | no |
| Blosc2 (plain and `b2nd` chunks), ZFP (every mode) | yes | no |
| SZ3, SZ (`float` and `double` data) | yes | no |

Lossy codecs are decoded bit for bit as their reference libraries decode, built with fused
multiply-adds where the reference's compiler fuses them (clang, and GCC on ARM); builds
without them (x86-64) can differ in the last bit of some `double` values. SZ's point-wise
relative mode also depends on the platform's `pow` in the last bit.

### Compressing a whole file

`compress=` on `to_netcdf` (of a `Dataset`, `DataTree` or, through `annoray.to_netcdf`, an
xarray dataset) compresses every variable whose encoding sets no compression of its own, so
`encoding` need not list them. It is `True` (deflate level 4), a deflate level 1–9, or a dict
of the encoding's compression keys (`compression`, `complevel`, `zlib`, `shuffle`,
`blosc_shuffle`) and `min_elements`. Shuffle is on unless the dict turns it off. Variables of
fewer than `min_elements` values (256 by default), scalars and strings are written as they
are; netCDF-3 ignores the option. xarray has no such option.

```{code-cell} python
big = ar.Dataset({"v": (("y", "x"), np.random.default_rng(0).normal(size=(200, 100)))})
big.to_netcdf(tmp / "deflate.nc", compress=True)
big.to_netcdf(tmp / "zstd.nc", compress={"compression": "zstd", "complevel": 3})
for name in ["deflate.nc", "zstd.nc"]:
    enc = ar.open_dataset(tmp / name)["v"].encoding
    print(name, enc.get("zlib"), enc.get("compression"), enc["shuffle"])
```

## Writing in place

`NetcdfWriter` writes into a netCDF-4 file without rewriting it. `NetcdfWriter.create` makes
a new file from a template: its dimensions, variables, attributes and encodings. The
coordinates are written at once; the data variables read as fill values until they are
written. A dimension listed in `unlimited_dims` may start at length 0 and grow.

```{code-cell} python
template = ar.Dataset(
    {"temp": (("time", "x"), np.zeros((0, 4), "f4"), {"units": "K"})},
    coords={
        "time": ("time", np.array([], "datetime64[ns]")),
        "x": ("x", np.arange(4.0)),
    },
)
encoding = {"time": {"units": "hours since 2000-01-01"}, "temp": {"zlib": True}}
```

Each step's values are appended as they are produced (`append` grows every variable along the
dimension). `dataset()` is the file as it now is, opened lazily, so what has been written can
be read while writing continues.

```{code-cell} python
with ar.NetcdfWriter.create(
    tmp / "run.nc", template, unlimited_dims=["time"], encoding=encoding
) as w:
    for hour in range(3):
        step = ar.Dataset(
            {"temp": (("time", "x"), np.full((1, 4), 280.0 + hour, "f4"), {"units": "K"})},
            coords={"time": ("time", [np.datetime64("2000-01-01T00") + np.timedelta64(hour, "h")])},
        )
        w.append(step, "time")
    print(w.dataset()["temp"].values[:, 0])
```

Values are encoded with the file's own encoding of each variable: its packing, fill value,
time units and on-disk type. They are converted to the variable's unit in the file first:
values in kelvin go into a variable in degrees Celsius as Celsius. Times that the file's
units cannot hold exactly, in an integer type, are an error, as in xarray.

`write` puts values into part of the file: along each dimension of `region` from the given
start, along the others from 0. Coordinates along none of the region's dimensions, such as
fixed latitudes, are left as they are.

```{code-cell} python
with ar.NetcdfWriter(tmp / "run.nc") as w:
    fix = ar.Dataset({"temp": (("time", "x"), np.zeros((1, 2), "f4"), {"units": "K"})})
    w.write(fix, region={"time": 1, "x": slice(2, 4)})
print(ar.open_dataset(tmp / "run.nc")["temp"].values)
```

`to_netcdf(path, mode="a", append_dim=...)` and `to_netcdf(path, mode="a", region=...)` do
the same in one call, as xarray's `to_zarr` does for Zarr stores.

```{code-cell} python
step = ar.Dataset(
    {"temp": (("time", "x"), np.full((1, 4), 300.0, "f4"), {"units": "K"})},
    coords={"time": ("time", [np.datetime64("2000-01-01T03")])},
)
step.to_netcdf(tmp / "run.nc", mode="a", append_dim="time")
print(ar.open_dataset(tmp / "run.nc").sizes["time"])
```

### How the file changes

Nothing in the file moves. New and rewritten chunks, and the chunk indexes rebuilt to take
them, go at the end of the file; the header messages that point to them keep their size and
are rewritten where they are, with their checksums. The space a rewritten chunk used before
is not reclaimed (`h5repack` reclaims it). Every value is encoded and checked before
anything is written, so values that cannot be stored leave the file as it was. Changes are
buffered until `flush()`, `dataset()` or `close()` (or the end of a `with` block); a writer
dropped without closing loses what it buffered.

Files written by netCDF-C, by h5py with its default settings and by annoray can be written
in place. Datasets indexed by the structures HDF5 uses with `libver="latest"` (fixed and
extensible arrays, version-2 B-trees) cannot, and are refused with an error. Readers that
opened the file before the writer keep seeing it as it was when they opened it: chunks are
never overwritten.
