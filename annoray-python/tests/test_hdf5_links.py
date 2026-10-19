"""HDF5 soft and external links: links to datasets are followed; links that cannot be (to
groups, dangling, cyclic) are reported, never silently dropped."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import annoray as ar

h5py = pytest.importorskip("h5py")


def opened(path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = ar.open_dataset(path)
    return ds, [str(x.message) for x in w]


def test_soft_and_external_links_to_datasets(tmp_path):
    with h5py.File(tmp_path / "other.h5", "w") as g:
        g["sub/w"] = np.arange(3.0)
    with h5py.File(tmp_path / "f.h5", "w", libver="latest") as f:
        f["v"] = np.arange(4.0)
        f["g/inner"] = np.arange(2.0) + 10
        f["absolute"] = h5py.SoftLink("/v")
        f["g/relative"] = h5py.SoftLink("inner")
        f["ext"] = h5py.ExternalLink("other.h5", "/sub/w")
    ds, msgs = opened(tmp_path / "f.h5")
    np.testing.assert_array_equal(ds["absolute"].values, np.arange(4.0))
    np.testing.assert_array_equal(ds["ext"].values, np.arange(3.0))
    assert not msgs, msgs
    tree = ar.open_datatree(tmp_path / "f.h5")
    np.testing.assert_array_equal(tree["g"]["relative"].values, [10.0, 11.0])


def test_old_style_groups_with_soft_links(tmp_path):
    # libver="earliest": symbol-table groups, whose soft links have no object address.
    with h5py.File(tmp_path / "f.h5", "w", libver="earliest") as f:
        f["v"] = np.arange(4.0)
        f["s"] = h5py.SoftLink("/v")
    ds, _ = opened(tmp_path / "f.h5")
    np.testing.assert_array_equal(ds["s"].values, np.arange(4.0))


def test_links_that_are_not_followed_are_reported(tmp_path):
    with h5py.File(tmp_path / "f.h5", "w") as f:
        f["v"] = np.arange(4.0)
        f.create_group("grp")["x"] = np.arange(2.0)
        f["to_group"] = h5py.SoftLink("/grp")
        f["dangling"] = h5py.SoftLink("/nowhere")
        f["self"] = h5py.SoftLink("/self")
        f["missing_file"] = h5py.ExternalLink("absent.h5", "/x")
    ds, msgs = opened(tmp_path / "f.h5")
    assert list(ds.data_vars) == ["v"]
    text = " ".join(msgs)
    for name in ("to_group", "dangling", "self", "missing_file"):
        assert name in text, (name, msgs)
