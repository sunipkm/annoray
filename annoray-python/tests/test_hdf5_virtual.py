"""HDF5 virtual datasets: values gathered from source datasets (in the same file or others)
through their mappings, eagerly or slice by slice; unmapped elements hold the fill value."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

h5py = pytest.importorskip("h5py")


@pytest.fixture(params=["earliest", "latest"])
def vds(tmp_path, request):
    for i in range(4):
        with h5py.File(tmp_path / f"src{i}.h5", "w") as f:
            f["data"] = np.arange(100, dtype="f8").reshape(10, 10) + i * 1000
    stack = h5py.VirtualLayout(shape=(4, 10, 10), dtype="f8")
    for i in range(4):
        stack[i] = h5py.VirtualSource(f"src{i}.h5", "data", shape=(10, 10))
    path = tmp_path / "vds.h5"
    with h5py.File(path, "w", libver=request.param) as f:
        f.create_virtual_dataset("stack", stack, fillvalue=-1)
        f["a"] = np.arange(20, dtype="i4")
        mixed = h5py.VirtualLayout(shape=(30,), dtype="i4")
        mixed[0:20:2] = h5py.VirtualSource(f["a"])[0:10]
        mixed[25:30] = h5py.VirtualSource(f["a"])[15:20]
        f.create_virtual_dataset("mixed", mixed, fillvalue=-7)
        f["t"] = np.arange(36, dtype="u2").reshape(6, 6)
        tiles = h5py.VirtualLayout(shape=(6, 12), dtype="u2")
        tiles[:, :6] = h5py.VirtualSource(f["t"])
        tiles[:, 6:] = h5py.VirtualSource(f["t"])
        f.create_virtual_dataset("tiles", tiles)
        row = h5py.VirtualLayout(shape=(3, 20), dtype="i4")
        row[1, :] = h5py.VirtualSource(f["a"])
        f.create_virtual_dataset("row", row, fillvalue=0)
    with h5py.File(path) as f:
        want = {k: f[k][...] for k in ("stack", "mixed", "tiles", "row")}
    return path, want


@pytest.mark.parametrize("lazy", [True, False])
def test_read_virtual_datasets(vds, lazy):
    path, want = vds
    ds = ar.open_dataset(path, lazy=lazy, mask_and_scale=False)
    for name, values in want.items():
        np.testing.assert_array_equal(ds[name].values, values, err_msg=name)


def test_virtual_slices(vds):
    path, want = vds
    ds = ar.open_dataset(path, mask_and_scale=False)
    s = ds["stack"]
    d = s.dims
    got = s.isel({d[0]: slice(1, 3), d[1]: slice(2, 7), d[2]: 3}).values
    np.testing.assert_array_equal(got, want["stack"][1:3, 2:7, 3])
    m = ds["mixed"]
    # Strided mapping, fill between, and a second mapping.
    np.testing.assert_array_equal(m.isel({m.dims[0]: slice(15, 28)}).values, want["mixed"][15:28])


def test_missing_source_file_is_an_error(vds, tmp_path):
    path, _ = vds
    (tmp_path / "src2.h5").unlink()
    ds = ar.open_dataset(path, mask_and_scale=False)
    with pytest.raises(Exception, match="src2.h5"):
        _ = ds["stack"].values
