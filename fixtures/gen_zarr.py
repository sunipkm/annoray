"""Zarr v2 and v3 stores written by zarr-python, with their expected values, for annoray-zarr.

Writes fixtures/zarr/v2.zarr, fixtures/zarr/v3.zarr and fixtures/zarr/expected.json (shape,
dtype kind and flattened values of every array; NaN as "NaN").

    python fixtures/gen_zarr.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numcodecs
import numpy as np
import zarr
from zarr.codecs import (
    BloscCodec,
    BytesCodec,
    Crc32cCodec,
    GzipCodec,
    TransposeCodec,
    ZstdCodec,
)

HERE = Path(__file__).parent / "zarr"
RNG = np.random.default_rng(42)


def values(shape, dtype):
    n = int(np.prod(shape))
    if dtype == "bool":
        return (np.arange(n) % 3 == 0).reshape(shape)
    if np.dtype(dtype).kind in "iu":
        info = np.iinfo(dtype)
        v = RNG.integers(max(info.min, -1000), min(info.max, 1000), n)
        return v.astype(dtype).reshape(shape)
    if np.dtype(dtype).kind == "f":
        v = RNG.standard_normal(n).astype(dtype)
        if n > 3:
            v[3] = np.nan
        return v.reshape(shape)
    raise ValueError(dtype)


def text(shape):
    words = ["", "a", "bb", "ccc", "dé", "e⚡f"]
    return np.array([words[i % len(words)] for i in range(int(np.prod(shape)))], dtype=object).reshape(shape)


def plain(v):
    out = []
    for x in np.asarray(v).ravel().tolist():
        if isinstance(x, float) and x != x:
            out.append("NaN")
        elif isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(x)
    return out


def main():
    shutil.rmtree(HERE, ignore_errors=True)
    HERE.mkdir(parents=True)
    expected = {}

    def record(fmt, name, data, kind=None):
        expected[f"{fmt}/{name}"] = {
            "shape": list(np.shape(data)),
            "kind": kind or np.asarray(data).dtype.kind,
            "values": plain(data),
        }

    # ----- v2 -----
    root = zarr.open_group(HERE / "v2.zarr", mode="w", zarr_format=2)
    shape, chunks = (7, 5), (3, 2)  # partial edge chunks
    for dt in ["bool", "int8", "uint16", "int32", "int64", "uint64", "float32", "float64"]:
        v = values(shape, dt)
        a = root.create_array(dt, shape=shape, chunks=chunks, dtype=dt)
        a[...] = v
        record(2, dt, v)
    v = values(shape, "float64")
    for name, comp in {
        "none": None,
        "blosc_lz4_shuffle": numcodecs.Blosc(cname="lz4", clevel=5, shuffle=1),
        "blosc_zstd_bitshuffle": numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2),
        "blosc_zlib_noshuffle": numcodecs.Blosc(cname="zlib", clevel=4, shuffle=0),
        "blosc_blosclz": numcodecs.Blosc(cname="blosclz", clevel=5, shuffle=1),
        "zlib": numcodecs.Zlib(level=6),
        "gzip": numcodecs.GZip(level=6),
        "zstd": numcodecs.Zstd(level=3),
        "lz4": numcodecs.LZ4(),
        "bz2": numcodecs.BZ2(level=5),
    }.items():
        a = root.create_array(f"comp_{name}", shape=shape, chunks=chunks, dtype="f8", compressors=comp)
        a[...] = v
        record(2, f"comp_{name}", v)
    a = root.create_array(
        "shuffle_filter", shape=shape, chunks=chunks, dtype="i4", filters=[numcodecs.Shuffle(elementsize=4)]
    )
    vi = values(shape, "int32")
    a[...] = vi
    record(2, "shuffle_filter", vi)
    a = root.create_array("order_f", shape=shape, chunks=chunks, dtype="f8", order="F")
    a[...] = v
    record(2, "order_f", v)
    a = root.create_array("big_endian", shape=shape, chunks=chunks, dtype=">f8")
    a[...] = v
    record(2, "big_endian", v)
    a = root.create_array(
        "slash_sep", shape=shape, chunks=chunks, dtype="f8", chunk_key_encoding={"name": "v2", "separator": "/"}
    )
    a[...] = v
    record(2, "slash_sep", v)
    # A missing chunk: the fill value.
    a = root.create_array("missing", shape=(6,), chunks=(2,), dtype="f8", fill_value=-9.5)
    a[0:2] = [1.0, 2.0]
    record(2, "missing", np.array([1.0, 2.0, -9.5, -9.5, -9.5, -9.5]))
    a = root.create_array("scalar", shape=(), dtype="i4")
    a[()] = 42
    record(2, "scalar", np.array(42, dtype="i4"))
    s = text((3, 4))
    a = root.create_array("vlen", shape=(3, 4), chunks=(2, 3), dtype=str)
    a[...] = s
    record(2, "vlen", s, "U")
    u = np.array(["x", "yy", "zzz", "é"], dtype="<U3")
    a = root.create_array("fixed_unicode", shape=(4,), chunks=(3,), dtype="<U3")
    a[...] = u
    record(2, "fixed_unicode", u)
    b = np.array([b"ab", b"c", b"", b"defg"], dtype="S4")
    a = root.create_array("fixed_bytes", shape=(4,), chunks=(4,), dtype="S4")
    a[...] = b
    record(2, "fixed_bytes", b, "U")
    root.attrs["title"] = "v2 fixtures"
    zarr.consolidate_metadata(HERE / "v2.zarr")

    # ----- v3 -----
    root = zarr.open_group(HERE / "v3.zarr", mode="w", zarr_format=3)
    for dt in ["bool", "int8", "uint16", "int32", "int64", "uint64", "float32", "float64"]:
        v3 = values(shape, dt)
        a = root.create_array(dt, shape=shape, chunks=chunks, dtype=dt, dimension_names=["y", "x"])
        a[...] = v3
        record(3, dt, v3)
    for name, comp in {
        "none": None,
        "gzip": GzipCodec(level=6),
        "zstd": ZstdCodec(level=3),
        "zstd_checksum": ZstdCodec(level=3, checksum=True),
        "blosc": BloscCodec(cname="lz4", clevel=5, shuffle="shuffle"),
        "blosc_bitshuffle": BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"),
        "crc32c": [Crc32cCodec()],
    }.items():
        a = root.create_array(f"comp_{name}", shape=shape, chunks=chunks, dtype="f8", compressors=comp)
        a[...] = v
        record(3, f"comp_{name}", v)
    a = root.create_array(
        "big_endian", shape=shape, chunks=chunks, dtype="f8", serializer=BytesCodec(endian="big")
    )
    a[...] = v
    record(3, "big_endian", v)
    a = root.create_array(
        "transpose", shape=shape, chunks=chunks, dtype="f8", filters=[TransposeCodec(order=(1, 0))]
    )
    a[...] = v
    record(3, "transpose", v)
    a = root.create_array("sharded", shape=(8, 9), chunks=(2, 3), shards=(4, 9), dtype="i4")
    vs = values((8, 9), "int32")
    vs[4:, :] = 0  # a shard left empty (all fill)
    a[...] = vs
    record(3, "sharded", vs)
    a = root.create_array("missing", shape=(6,), chunks=(2,), dtype="f8", fill_value=-9.5)
    a[0:2] = [1.0, 2.0]
    record(3, "missing", np.array([1.0, 2.0, -9.5, -9.5, -9.5, -9.5]))
    a = root.create_array("scalar", shape=(), dtype="i4")
    a[()] = 42
    record(3, "scalar", np.array(42, dtype="i4"))
    s = text((3, 4))
    a = root.create_array("string", shape=(3, 4), chunks=(2, 3), dtype=str)
    a[...] = s
    record(3, "string", s, "U")
    a = root.create_array("fixed_unicode", shape=(4,), chunks=(3,), dtype="<U3")
    a[...] = u
    record(3, "fixed_unicode", u)
    a = root.create_array(
        "v2_keys",
        shape=shape,
        chunks=chunks,
        dtype="f8",
        chunk_key_encoding={"name": "v2", "separator": "."},
    )
    a[...] = v
    record(3, "v2_keys", v)
    g = root.create_group("sub", attributes={"level": 1})
    a = g.create_array("inner", shape=(3,), dtype="i2")
    a[...] = [1, 2, 3]
    record(3, "sub/inner", np.array([1, 2, 3], dtype="i2"))
    root.attrs["title"] = "v3 fixtures"

    (HERE / "expected.json").write_text(json.dumps(expected, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(expected)} arrays to {HERE}")
    write_xarray()


def xarray_dataset():
    """A CF dataset as xarray writes it: times, a NaN fill, an integer with _FillValue, packed
    data, text, coordinates and attributes, in several chunks with one missing."""
    import pandas as pd
    import xarray as xr

    t = pd.date_range("2000-01-01", periods=6, freq="12h")
    temp = np.arange(6 * 4, dtype="f4").reshape(6, 4) / 3
    temp[1, 2] = np.nan
    temp[4:, :] = np.nan  # a chunk left as fill
    ds = xr.Dataset(
        {
            "temp": (("time", "x"), temp, {"units": "degC", "long_name": "temperature"}),
            "count": (("x",), np.array([1, -1, 3, 4], dtype="i2")),
            "label": (("x",), np.array(["a", "bb", "ccc", "d"])),
            "packed": (("time",), np.linspace(0, 1, 6)),
        },
        coords={"time": t, "x": np.array([10.0, 20.0, 30.0, 40.0]), "station": ("x", ["s1", "s2", "s3", "s4"])},
        attrs={"title": "xarray fixture", "version": 2},
    )
    ds["count"].encoding = {"_FillValue": -1}
    ds["temp"].encoding = {"chunks": (2, 2)}
    ds["packed"].encoding = {"dtype": "int16", "scale_factor": 0.01, "add_offset": 0.0, "_FillValue": -32767}
    return ds


def write_xarray():
    ds = xarray_dataset()
    for fmt in (2, 3):
        ds.to_zarr(HERE / f"xarray_v{fmt}.zarr", zarr_format=fmt, mode="w", consolidated=True)
        ds.to_netcdf(HERE / f"xarray_v{fmt}.nc")
    print("wrote xarray stores (and netCDF twins) to", HERE)


if __name__ == "__main__":
    main()
