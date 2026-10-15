"""HDF5 file with compressed and packed chunks, for annoray-hdf5/tests/filters.rs.

Needs h5py and hdf5plugin (which bundles HDF5's filter plugins), e.g. in a throwaway venv:
    python3 -m venv /tmp/v && /tmp/v/bin/pip install h5py hdf5plugin
    /tmp/v/bin/python fixtures/gen_filters.py          (from the repository root)

Writes fixtures/h5/filters.h5. Every dataset `<name>` has an uncompressed copy
`<name>_plain` holding what HDF5 itself reads back.

With `--big <out.h5>` it instead writes large 1024x1024 chunks (several Blosc blocks and
streams, many Bitshuffle blocks, multi-block bzip2 streams, ragged edge chunks) to a
scratch file, for the ignored test `annoray-hdf5/tests/filters.rs::big_chunks`.
"""

import ctypes
import glob
import os
import sys

import h5py
import hdf5plugin
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "h5", "filters.h5")
rng = np.random.default_rng(1)
ints = np.cumsum(rng.integers(-3, 4, size=(40, 30)), axis=1).astype("int32")
floats = np.round(rng.standard_normal((25, 16)), 2)
bytes_ = rng.integers(0, 4, size=(33, 7)).astype("uint8")
i16 = (np.arange(300, dtype="int16").reshape(20, 15) * 7 - 1000)


PENDING = []


def add(f, name, data, **kw):
    f.create_dataset(name, data=data, **kw)
    PENDING.append(name)


def add_plain_copies(path):
    """Write `<name>_plain` from what HDF5 reads back from disk. (Reading right after writing
    can be served from the chunk cache, which holds the data before filtering: for lossy
    filters such as scale-offset that is not what a reader gets.)"""
    with h5py.File(path, "a") as f:
        for name in PENDING:
            f.create_dataset(name + "_plain", data=f[name][()])
    PENDING.clear()


if len(sys.argv) == 3 and sys.argv[1] == "--big":
    shape, chunks = (1100, 1030), (1024, 1024)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    smooth = np.round(np.sin(x / 37.0) * np.cos(y / 53.0) * 100 + rng.normal(0, 0.5, shape), 2)
    counts = (np.abs(smooth) * 10).astype("int32")
    with h5py.File(sys.argv[2], "w") as f:
        add(f, "deflate", smooth, chunks=chunks, compression="gzip", shuffle=True)
        add(f, "lzf", counts, chunks=chunks, compression="lzf")
        add(f, "zstd", smooth, chunks=chunks, **hdf5plugin.Zstd(clevel=9))
        add(f, "lz4", counts, chunks=chunks, **hdf5plugin.LZ4())
        add(f, "lz4_blocks", smooth, chunks=chunks, **hdf5plugin.LZ4(nbytes=65536))
        add(f, "bzip2", counts, chunks=chunks, **hdf5plugin.BZip2(blocksize=1))
        for cname in ["blosclz", "lz4", "lz4hc", "zlib", "zstd"]:
            for shuf, tag in [(hdf5plugin.Blosc.SHUFFLE, "s"), (hdf5plugin.Blosc.BITSHUFFLE, "b")]:
                add(f, f"blosc_{cname}_{tag}", smooth, chunks=chunks, **hdf5plugin.Blosc(cname=cname, clevel=5, shuffle=shuf))
        add(f, "blosc_i32_noshuffle", counts, chunks=chunks, **hdf5plugin.Blosc(cname="lz4", clevel=9, shuffle=hdf5plugin.Blosc.NOSHUFFLE))
        add(f, "bitshuffle", counts, chunks=chunks, **hdf5plugin.Bitshuffle(cname="none"))
        add(f, "bitshuffle_lz4", smooth, chunks=chunks, **hdf5plugin.Bitshuffle(cname="lz4"))
        add(f, "bitshuffle_zstd", counts, chunks=chunks, **hdf5plugin.Bitshuffle(cname="zstd"))
        add(f, "scaleoffset_int", counts, chunks=chunks, scaleoffset=0)
        add(f, "scaleoffset_float", smooth, chunks=chunks, scaleoffset=2)
    add_plain_copies(sys.argv[2])
    for name in sorted(h5py.File(sys.argv[2])):
        if not name.endswith("_plain"):
            d = h5py.File(sys.argv[2])[name]
            masks = {d.id.read_direct_chunk(c.chunk_offset)[0] for c in (d.id.get_chunk_info(i) for i in range(d.id.get_num_chunks()))}
            print(name, d._filters, "skipped filters in some chunks" if masks != {0} else "")
    sys.exit(0)

with h5py.File(OUT, "w") as f:
    # Built into h5py.
    add(f, "lzf", ints, chunks=(16, 16), compression="lzf", shuffle=True)
    # Plugins (real encoders).
    add(f, "zstd", floats, chunks=(10, 8), **hdf5plugin.Zstd(clevel=3))
    add(f, "lz4", ints, chunks=(16, 16), **hdf5plugin.LZ4(nbytes=512))
    add(f, "bzip2", ints, chunks=(16, 16), **hdf5plugin.BZip2(blocksize=9))
    add(f, "bzip2_bytes", bytes_, chunks=(11, 7), **hdf5plugin.BZip2(blocksize=1))
    for cname in ["blosclz", "lz4", "lz4hc", "zlib", "zstd"]:
        for shuf, tag in [(hdf5plugin.Blosc.NOSHUFFLE, "n"), (hdf5plugin.Blosc.SHUFFLE, "s"), (hdf5plugin.Blosc.BITSHUFFLE, "b")]:
            add(f, f"blosc_{cname}_{tag}", floats, chunks=(10, 8), **hdf5plugin.Blosc(cname=cname, clevel=5, shuffle=shuf))
    add(f, "blosc_i16", i16, chunks=(7, 9), **hdf5plugin.Blosc(cname="lz4", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE))
    # 143 elements per block: format version 2 leaves such blocks unshuffled.
    add(f, "blosc_odd_bit", np.repeat(np.arange(26, dtype="int16"), 22).reshape(26, 22), chunks=(13, 11), **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
    add(f, "bitshuffle", ints, chunks=(16, 16), **hdf5plugin.Bitshuffle(cname="none"))
    add(f, "bitshuffle_lz4", floats, chunks=(10, 8), **hdf5plugin.Bitshuffle(cname="lz4"))
    add(f, "bitshuffle_zstd", i16, chunks=(7, 9), **hdf5plugin.Bitshuffle(cname="zstd"))
    # HDF5's scale-offset filter.
    add(f, "scaleoffset_int", ints, chunks=(16, 16), scaleoffset=0)
    add(f, "scaleoffset_fill", ints, chunks=(16, 16), scaleoffset=0, fillvalue=-99)
    add(f, "scaleoffset_float", floats, chunks=(10, 8), scaleoffset=2)
    add(f, "scaleoffset_u8", bytes_, chunks=(11, 7), scaleoffset=0)
add_plain_copies(OUT)

# N-bit needs a reduced-precision type, which HDF5 2.0 only writes with relaxed checks.
lib = ctypes.CDLL(glob.glob(os.path.join(os.path.dirname(h5py.__file__), "..", "h5py*.dylibs", "libhdf5.*"))[0]
                  if glob.glob(os.path.join(os.path.dirname(h5py.__file__), "..", "h5py*.dylibs", "libhdf5.*"))
                  else glob.glob(os.path.join(os.path.dirname(h5py.__file__), ".dylibs", "libhdf5.*"))[0])
fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
lib.H5Pset_relax_file_integrity_checks(ctypes.c_int64(fapl.id), ctypes.c_uint64(1))
fid = h5py.h5f.open(OUT.encode(), h5py.h5f.ACC_RDWR, fapl=fapl)
for name, base, prec, off, values in [
    ("nbit", h5py.h5t.STD_I32LE, 12, 2, (np.arange(40, dtype="i4") * 97 - 1900).reshape(5, 8)),
    ("nbit_u16", h5py.h5t.STD_U16BE, 9, 3, (np.arange(40, dtype=">u2") * 13).reshape(5, 8)),
]:
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((3, 4))
    dcpl.set_filter(h5py.h5z.FILTER_NBIT, 0, ())
    t = base.copy()
    t.set_precision(prec)
    t.set_offset(off)
    d = h5py.h5d.create(fid, name.encode(), t, h5py.h5s.create_simple(values.shape), dcpl=dcpl)
    d.write(h5py.h5s.ALL, h5py.h5s.ALL, values)
fid.close()
with h5py.File(OUT, "a") as f:
    for name in ["nbit", "nbit_u16"]:
        f.create_dataset(name + "_plain", data=f[name][()].astype("int64"))
print("wrote", OUT)
with h5py.File(OUT) as f:
    for k in sorted(f):
        if not k.endswith("_plain"):
            print(k, f[k]._filters)
