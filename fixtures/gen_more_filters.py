"""HDF5 file with chunks of the Blosc2, ZFP, SZ and SZ3 plugins, for
annoray-hdf5/tests/more_filters.rs.

Needs h5py and hdf5plugin (which bundles the plugins), e.g. in a throwaway venv:
    python3 -m venv /tmp/v && /tmp/v/bin/pip install h5py hdf5plugin
    /tmp/v/bin/python fixtures/gen_more_filters.py          (from the repository root)

Writes fixtures/h5/more_filters.h5. Every dataset `<name>` has an uncompressed copy
`<name>_plain` holding what HDF5 itself reads back (for lossy filters: the decompressed
values, which a reader must reproduce exactly).
"""

import os
import sys

import h5py
import hdf5plugin
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "h5", "more_filters.h5")
rng = np.random.default_rng(7)
y, x = np.mgrid[0:37, 0:45]
smooth = np.sin(x / 7.0) * np.cos(y / 5.0) * 100 + rng.normal(0, 0.5, x.shape)
counts = (np.abs(smooth) * 10).astype("int32")
cube = np.cumsum(rng.normal(size=(9, 13, 17)), axis=2)

PENDING = []


def add(f, name, data, **kw):
    f.create_dataset(name, data=data, **kw)
    PENDING.append(name)


with h5py.File(OUT, "w") as f:
    # Blosc2: 1-D chunks are plain Blosc2 chunks; n-D ones are b2nd arrays (blocks tiled).
    for cname in ("blosclz", "lz4", "lz4hc", "zlib", "zstd"):
        for filt, tag in [
            (hdf5plugin.Blosc2.NOFILTER, "n"),
            (hdf5plugin.Blosc2.SHUFFLE, "s"),
            (hdf5plugin.Blosc2.BITSHUFFLE, "b"),
        ]:
            add(f, f"blosc2_{cname}_{tag}", smooth, chunks=(16, 20), **hdf5plugin.Blosc2(cname=cname, clevel=5, filters=filt))
    add(f, "blosc2_1d", smooth.ravel(), chunks=(500,), **hdf5plugin.Blosc2(cname="lz4", filters=hdf5plugin.Blosc2.SHUFFLE))
    add(f, "blosc2_i32", counts, chunks=(37, 45), **hdf5plugin.Blosc2(cname="zstd", clevel=9, filters=hdf5plugin.Blosc2.BITSHUFFLE))
    add(f, "blosc2_3d", cube, chunks=(4, 6, 17), **hdf5plugin.Blosc2(cname="zstd", filters=hdf5plugin.Blosc2.SHUFFLE))
    add(f, "blosc2_u8", (counts % 256).astype("u1"), chunks=(10, 45), **hdf5plugin.Blosc2(cname="lz4"))
    add(f, "blosc2_const", np.full((30, 30), 3.5), chunks=(15, 15), **hdf5plugin.Blosc2(cname="lz4"))
    add(f, "blosc2_zeros", np.zeros((30, 30), "i2"), chunks=(15, 15), **hdf5plugin.Blosc2(cname="zstd"))

    # ZFP (H5Z-ZFP): every mode, float and integer types, 1 to 4 dimensions, ragged edges.
    f32 = smooth.astype("f4")
    add(f, "zfp_rate_f64", smooth, chunks=(16, 20), **hdf5plugin.Zfp(rate=10.0))
    add(f, "zfp_rate_f32", f32, chunks=(16, 20), **hdf5plugin.Zfp(rate=8.0))
    add(f, "zfp_prec_f64", smooth, chunks=(16, 20), **hdf5plugin.Zfp(precision=20))
    add(f, "zfp_prec_f32", f32, chunks=(37, 45), **hdf5plugin.Zfp(precision=12))
    add(f, "zfp_acc_f64", smooth, chunks=(16, 20), **hdf5plugin.Zfp(accuracy=0.01))
    add(f, "zfp_acc_f32", f32, chunks=(16, 20), **hdf5plugin.Zfp(accuracy=0.5))
    add(f, "zfp_rev_f64", smooth, chunks=(16, 20), **hdf5plugin.Zfp(reversible=True))
    add(f, "zfp_rev_f32", f32, chunks=(16, 20), **hdf5plugin.Zfp(reversible=True))
    add(f, "zfp_rev_i32", counts, chunks=(16, 20), **hdf5plugin.Zfp(reversible=True))
    add(f, "zfp_prec_i64", counts.astype("i8"), chunks=(16, 20), **hdf5plugin.Zfp(precision=40))
    add(f, "zfp_1d", smooth.ravel(), chunks=(301,), **hdf5plugin.Zfp(precision=24))
    add(f, "zfp_3d", cube, chunks=(9, 13, 17), **hdf5plugin.Zfp(accuracy=0.001))
    add(f, "zfp_3d_rate", cube.astype("f4"), chunks=(4, 6, 17), **hdf5plugin.Zfp(rate=6.0))
    add(f, "zfp_4d", cube.reshape(3, 3, 13, 17), chunks=(3, 3, 13, 17), **hdf5plugin.Zfp(precision=30))
    add(f, "zfp_4d_rev", cube.reshape(3, 3, 13, 17), chunks=(3, 3, 7, 9), **hdf5plugin.Zfp(reversible=True))
    add(f, "zfp_zeros", np.zeros((20, 20)), chunks=(20, 20), **hdf5plugin.Zfp(precision=16))
    add(f, "zfp_expert", smooth, chunks=(16, 20), **hdf5plugin.Zfp(minbits=8, maxbits=400, maxprec=30, minexp=-20))

    # SZ3: the algorithm (interpolation, or Lorenzo/regression) is chosen per chunk by the
    # compressor; smooth fields favour interpolation, rough ones Lorenzo. 3-D Lorenzo goes
    # through SZ3's fast frontend.
    rough = rng.normal(size=(40, 50)).cumsum(axis=1)
    add(f, "sz3_2d_f64", smooth, chunks=(37, 45), **hdf5plugin.SZ3(absolute=0.01))
    add(f, "sz3_2d_f32", smooth.astype("f4"), chunks=(37, 45), **hdf5plugin.SZ3(absolute=0.1))
    add(f, "sz3_2d_rough", rough, chunks=(40, 50), **hdf5plugin.SZ3(absolute=0.05))
    add(f, "sz3_2d_rough_f32", rough.astype("f4"), chunks=(20, 25), **hdf5plugin.SZ3(absolute=0.2))
    add(f, "sz3_1d", smooth.ravel(), chunks=(500,), **hdf5plugin.SZ3(absolute=0.01))
    add(f, "sz3_1d_rough", rough.ravel(), chunks=(700,), **hdf5plugin.SZ3(absolute=0.01))
    add(f, "sz3_3d", cube, chunks=(9, 13, 17), **hdf5plugin.SZ3(absolute=0.001))
    add(f, "sz3_3d_rough", rng.normal(size=(12, 14, 16)), chunks=(12, 14, 16), **hdf5plugin.SZ3(absolute=0.3))
    add(f, "sz3_3d_f32", cube.astype("f4"), chunks=(9, 13, 17), **hdf5plugin.SZ3(absolute=0.01))
    add(f, "sz3_4d", cube.reshape(3, 3, 13, 17), chunks=(3, 3, 13, 17), **hdf5plugin.SZ3(absolute=0.01))
    add(f, "sz3_rel", smooth, chunks=(37, 45), **hdf5plugin.SZ3(relative=1e-3))
    add(f, "sz3_lossless", smooth, chunks=(37, 45), **hdf5plugin.SZ3(absolute=0.0))
    walk3 = rng.normal(size=(16, 16, 16)).cumsum(0).cumsum(1).cumsum(2)
    add(f, "sz3_3d_lorenzo", walk3, chunks=(16, 16, 16), **hdf5plugin.SZ3(absolute=0.01))
    add(f, "sz3_3d_lorenzo_f32", walk3.astype("f4"), chunks=(16, 16, 16), **hdf5plugin.SZ3(absolute=0.05))
    steps3 = np.repeat(np.repeat(np.repeat(rng.normal(size=(4, 4, 4)), 4, 0), 4, 1), 4, 2)
    add(f, "sz3_3d_steps", steps3, chunks=(16, 16, 16), **hdf5plugin.SZ3(absolute=1e-4))
    walk4 = rng.normal(size=(6, 6, 8, 8)).cumsum(3).cumsum(2)
    add(f, "sz3_4d_lorenzo", walk4, chunks=(6, 6, 8, 8), **hdf5plugin.SZ3(absolute=0.01))

# SZ (SZ 2): absolute, relative and point-wise relative bounds; 1 to 4 dimensions.
with h5py.File(OUT, "a") as f:
    walk3 = np.random.default_rng(11).normal(size=(16, 16, 16)).cumsum(0).cumsum(1).cumsum(2)
    add(f, "sz_1d", smooth.ravel(), chunks=(500,), **hdf5plugin.SZ(absolute=0.01))
    add(f, "sz_1d_f32", smooth.ravel().astype("f4"), chunks=(700,), **hdf5plugin.SZ(absolute=0.05))
    add(f, "sz_2d", smooth, chunks=(37, 45), **hdf5plugin.SZ(absolute=0.01))
    add(f, "sz_2d_f32", smooth.astype("f4"), chunks=(20, 25), **hdf5plugin.SZ(absolute=0.1))
    add(f, "sz_2d_rel", smooth, chunks=(37, 45), **hdf5plugin.SZ(relative=1e-3))
    add(f, "sz_2d_pwrel", smooth, chunks=(37, 45), **hdf5plugin.SZ(pointwise_relative=1e-2))
    add(f, "sz_2d_pwrel_f32", smooth.astype("f4"), chunks=(37, 45), **hdf5plugin.SZ(pointwise_relative=1e-2))
    add(f, "sz_1d_pwrel", smooth.ravel(), chunks=(500,), **hdf5plugin.SZ(pointwise_relative=1e-3))
    add(f, "sz_3d", cube, chunks=(9, 13, 17), **hdf5plugin.SZ(absolute=0.001))
    add(f, "sz_3d_f32", cube.astype("f4"), chunks=(9, 13, 17), **hdf5plugin.SZ(absolute=0.01))
    add(f, "sz_3d_walk", walk3, chunks=(16, 16, 16), **hdf5plugin.SZ(absolute=0.01))
    add(f, "sz_3d_pwrel", walk3, chunks=(16, 16, 16), **hdf5plugin.SZ(pointwise_relative=1e-2))
    add(f, "sz_4d", cube.reshape(3, 3, 13, 17), chunks=(3, 3, 13, 17), **hdf5plugin.SZ(absolute=0.01))
    ramp = np.cumsum(np.random.default_rng(12).normal(size=3000)).astype("f4")
    add(f, "sz_1d_f32_walk", ramp, chunks=(1000,), **hdf5plugin.SZ(absolute=0.5))
    add(f, "sz_1d_pwrel_f32", ramp, chunks=(1000,), **hdf5plugin.SZ(pointwise_relative=1e-2))
    add(f, "sz_3d_walk_f32", walk3.astype("f4"), chunks=(16, 16, 16), **hdf5plugin.SZ(absolute=0.05))
    add(f, "sz_3d_pwrel_f32", walk3.astype("f4"), chunks=(16, 16, 16), **hdf5plugin.SZ(pointwise_relative=1e-2))
    # Mostly one value: the predictor's "mean" shortcut.
    plateau = np.full((20, 24, 18), 7.25)
    plateau[5:9, 3:20, 2:10] += np.random.default_rng(13).normal(size=(4, 17, 8))
    add(f, "sz_3d_plateau", plateau, chunks=(20, 24, 18), **hdf5plugin.SZ(absolute=0.01))
    add(f, "sz_2d_plateau_f32", plateau[0].astype("f4") + plateau[6].astype("f4"), chunks=(24, 18), **hdf5plugin.SZ(absolute=0.01))

# SZ3 streams with regression (from fixtures/sz3_regression.cpp), as raw chunks of SZ3
# datasets: what HDF5 reads back from them is the expected value.
STREAMS = {
    "reg2d_f64": ("f8", (40, 50)),
    "reg2d_f32": ("f4", (40, 50)),
    "regonly2d": ("f8", (37, 45)),
    "reg3d_f64": ("f8", (13, 14, 15)),
    "reg3d_f32": ("f4", (13, 14, 15)),
    "reg1d": ("f8", (500,)),
    "reg4d": ("f8", (5, 6, 7, 8)),
    "lor2only3d": ("f8", (13, 14, 15)),
}
if len(sys.argv) == 3 and sys.argv[1] == "--sz3-streams":
    with h5py.File(OUT, "a") as f:
        for name, (dtype, shape) in STREAMS.items():
            with open(os.path.join(sys.argv[2], name + ".sz"), "rb") as s:
                raw = s.read()
            d = f.create_dataset(
                "sz3_" + name, shape=shape, dtype=dtype, chunks=shape, **hdf5plugin.SZ3(absolute=0.01)
            )
            d.id.write_direct_chunk((0,) * len(shape), raw)
            PENDING.append("sz3_" + name)

with h5py.File(OUT, "a") as f:
    for name in PENDING:
        d = f[name]
        assert isinstance(d, h5py.Dataset), name
        f.create_dataset(name + "_plain", data=d[()])
