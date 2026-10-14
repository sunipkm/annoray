"""HDF5 fixtures for the pure-Rust reader, written with h5py, plus a canonical text dump of
each file's contents that the Rust tests must reproduce exactly.

Run from the repository root:  python3 fixtures/gen_hdf5.py
Writes fixtures/h5/<name>.h5 and fixtures/h5/<name>.dump.

Dump format, one line per object, sorted by path:
  G <path>
  D <path> <dtype> <shape> <values>
  A <path>@<name> <dtype> <shape> <values>
Values are space-separated; floats use repr(); strings are JSON-quoted.
"""

import json
import os

import h5py
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "h5")
os.makedirs(OUT, exist_ok=True)


def fmt_value(v):
    if isinstance(v, bytes):
        return json.dumps(v.decode("utf-8", "replace"))
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    return str(int(v))


def dtype_name(dt):
    if dt.kind == "O" or dt.kind in "SU":
        return "str"
    if dt.kind == "b":
        return "i8"  # h5py stores bools as an int8 enum
    return {"i": "i", "u": "u", "f": "f"}[dt.kind] + str(dt.itemsize * 8)


def dump_array(arr):
    arr = np.asarray(arr)
    if arr.dtype.names is not None:
        return None
    shape = "x".join(str(s) for s in arr.shape) if arr.shape else "scalar"
    vals = " ".join(fmt_value(v) for v in arr.ravel().tolist()) if arr.size else ""
    return f"{dtype_name(arr.dtype)} {shape} {vals}".rstrip()


def dump(path):
    lines = []

    def ref_name(f, r):
        return f[r].name if r else "null"

    def attrs(obj, p):
        for k in obj.attrs:
            v = obj.attrs[k]
            arr = np.asarray(v)
            flat = arr.ravel().tolist()
            if arr.dtype.names is not None and "dataset" in arr.dtype.names:
                pairs = " ".join(f"({ref_name(obj.file, x[0])},{int(x[1])})" for x in arr.ravel())
                lines.append(f"R {p}@{k} {pairs}")
                continue
            if flat and isinstance(flat[0], h5py.Reference):
                lines.append(f"R {p}@{k} " + " ".join(ref_name(obj.file, r) for r in flat))
                continue
            if flat and isinstance(flat[0], np.ndarray) and flat[0].size and isinstance(flat[0].ravel()[0], h5py.Reference):
                lines.append(f"R {p}@{k} " + " ".join("[" + ",".join(ref_name(obj.file, r) for r in x) + "]" for x in flat))
                continue
            d = dump_array(v)
            if d is not None:
                lines.append(f"A {p}@{k} {d}")

    with h5py.File(path, "r") as f:
        lines.append("G /")
        attrs(f, "/")

        def visit(name, obj):
            p = "/" + name
            if isinstance(obj, h5py.Group):
                lines.append(f"G {p}")
            else:
                if obj.dtype.names is not None or obj.dtype.kind == "V":
                    return
                lines.append(f"D {p} {dump_array(obj[()])}")
            attrs(obj, p)

        f.visititems(visit)
    with open(path.replace(".h5", ".dump"), "w") as out:
        out.write("\n".join(sorted(lines)) + "\n")


def contents(f, big_endian=False):
    """The same content, laid out according to the file's libver settings."""
    e = ">" if big_endian else "<"
    f.attrs["title"] = "hdf5 fixture"
    f.attrs["ints"] = np.array([1, -2, 3], dtype=e + "i4")
    f.attrs["pi"] = np.float64(np.pi)
    f.attrs["fixed_str"] = np.bytes_("fixed")
    f.attrs["vlen_strs"] = np.array(["a", "bc"], dtype=h5py.string_dtype())
    f.create_dataset("scalar", data=np.float32(1.25))
    f.create_dataset("contig_i16", data=np.arange(-5, 5, dtype=e + "i2"))
    f.create_dataset("contig_u64", data=np.array([0, 2**63, 2**64 - 1], dtype=e + "u8"))
    f.create_dataset("contig_f64_2d", data=np.arange(12, dtype=e + "f8").reshape(3, 4) / 7)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_layout(h5py.h5d.COMPACT)
    space = h5py.h5s.create_simple((10,))
    dsid = h5py.h5d.create(f.id, b"compact_u8", h5py.h5t.STD_U8LE, space, dcpl=dcpl)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, np.arange(10, dtype="u1"))
    f.create_dataset("empty", shape=(0, 3), dtype="f4")
    f.create_dataset("unwritten", shape=(4,), dtype="i4", fillvalue=-7)
    f.create_dataset("chunked", data=np.arange(100, dtype=e + "i4").reshape(10, 10), chunks=(3, 4))
    f.create_dataset("gzip_shuffle", data=np.linspace(0, 1, 50, dtype=e + "f4"), chunks=(7,), compression="gzip",
                     compression_opts=6, shuffle=True)
    f.create_dataset("fletcher", data=np.arange(20, dtype="i8"), chunks=(5,), fletcher32=True)
    part = f.create_dataset("partial_chunks", shape=(10,), dtype="f8", chunks=(4,), fillvalue=np.nan)
    part[0:4] = 1.5
    f.create_dataset("single_chunk", data=np.arange(6, dtype="i2"), chunks=(6,))
    f.create_dataset("resizable", data=np.arange(5, dtype="i4"), chunks=(2,), maxshape=(None,))
    f.create_dataset("resizable_2d", data=np.arange(12, dtype="f4").reshape(3, 4), chunks=(2, 2), maxshape=(None, None))
    f.create_dataset("fixed_strings", data=np.array([b"one", b"three", b""], dtype="S5"))
    f.create_dataset("vlen_strings", data=np.array(["αβγ", "", "plain"], dtype=h5py.string_dtype()))
    f.create_dataset("enum", data=np.array([0, 1, 1, 0], dtype="i1"),
                     dtype=h5py.enum_dtype({"OFF": 0, "ON": 1}, basetype="i1"))
    f.create_dataset("bools", data=np.array([True, False, True]))
    # Many chunks: paged fixed arrays (> 1024 chunks) and extensible arrays with super blocks.
    f.create_dataset("many_chunks", data=np.arange(1100, dtype="i2"), chunks=(1,))
    f.create_dataset("many_chunks_unlimited", data=(np.arange(2000) % 251).astype("u1"), chunks=(2,), maxshape=(None,))
    # Early allocation without filters: the implicit chunk index under libver latest.
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((4,))
    dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    dsid = h5py.h5d.create(f.id, b"early_alloc", h5py.h5t.STD_I32LE, h5py.h5s.create_simple((10,)), dcpl=dcpl)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, np.arange(10, dtype="i4") * 3)
    g = f.create_group("group")
    g.attrs["level"] = np.int32(1)
    g.create_dataset("inner", data=np.array([[1.5, 2.5]], dtype="f4"))
    deep = g.create_group("deeper/deepest")
    deep.create_dataset("leaf", data=np.int64(42))
    many = f.create_group("many_links")
    for i in range(40):
        many.create_dataset(f"d{i:02d}", data=np.int16(i))
    lots = f.create_dataset("many_attrs", data=np.int8(0))
    for i in range(30):
        lots.attrs[f"a{i:02d}"] = np.float32(i) / 2
    lots.attrs["long_text"] = "x" * 200


VARIANTS = {
    "earliest": dict(libver=("earliest", "latest")),
    "v18": dict(libver=("v108", "latest")),
    "latest": dict(libver="latest"),
}


def main():
    print(f"# h5py {h5py.__version__}, HDF5 {h5py.version.hdf5_version}")
    for name, kw in VARIANTS.items():
        for be in (False, True):
            fname = f"{name}{'_be' if be else ''}.h5"
            path = os.path.join(OUT, fname)
            with h5py.File(path, "w", **kw) as f:
                contents(f, big_endian=be)
            dump(path)
            print(fname, os.path.getsize(path), "bytes")
    # Old-style groups with creation order tracking off and a dense link group.
    path = os.path.join(OUT, "track_order.h5")
    with h5py.File(path, "w", libver="latest", track_order=True) as f:
        for i in range(20):
            f.create_group(f"g{i:02d}").attrs["i"] = i
    dump(path)
    print("track_order.h5", os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
