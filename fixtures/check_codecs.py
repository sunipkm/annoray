"""Check a file of plugin-compressed datasets written by annoray (annoray-hdf5/tests/write_codecs.rs)
with h5py and hdf5plugin (the HDF5 C library's own filter plugins).

Usage: python3 fixtures/check_codecs.py <written.h5>   (prints OK or raises)
Every dataset holds f(i, j) = i * 100 + j (divided by 7 for floats) at index (i, j).
"""

import sys
import typing
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 - registers the filter plugins
import numpy as np

expected_filters = {
    "zstd": "32015",
    "lz4": "32004",
    "blosc_lz4_byte": "32001",
    "blosc_zstd_bit": "32001",
    "blosc_zlib_none": "32001",
}
def dataset(f: h5py.File, name: str) -> h5py.Dataset:
    """`f[name]`, which must be a dataset."""
    d = f[name]
    assert isinstance(d, h5py.Dataset), name
    return d


with h5py.File(sys.argv[1]) as f:
    assert sorted(f) == sorted(expected_filters), sorted(f)
    for name, fid in expected_filters.items():
        d = dataset(f, name)
        filters = typing.cast("dict[str, Any]", d._filters)  # h5py types it as a cached_property
        assert fid in filters, (name, filters)
        i, j = np.indices(d.shape)
        want = (i * 100 + j).astype(d.dtype)
        if d.dtype.kind == "f":
            want = ((i * 100 + j) / 7).astype(d.dtype)
        np.testing.assert_array_equal(d[()], want, err_msg=name)
print("OK")
