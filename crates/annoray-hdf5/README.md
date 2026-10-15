# annoray-hdf5

Pure-Rust HDF5 reader and writer (no libhdf5); the HDF5 layer of
[annoray](https://docs.rs/annoray)'s netCDF-4 backend, independent of it.

- **Format coverage (read):** superblock v0–v3; object header v1/v2; compact, contiguous and
  chunked layouts with all chunk indexes (v1/v2 B-tree, single chunk, implicit, fixed and
  extensible array); symbol-table, compact and dense (fractal heap) groups and attributes;
  fixed/variable-length strings, enums, bit-field integers, object references.
- **Filters (decode):** deflate, shuffle, fletcher32, LZF, N-bit, scale-offset, bzip2, Blosc
  (BloscLZ/LZ4/zlib/Zstandard, byte and bit shuffle), Bitshuffle; Zstandard and LZ4 behind the
  `zstd`/`lz4` features.
- **Writer:** superblock v2 with v2 object headers; contiguous or chunked datasets (v1 B-tree
  index) with deflate/shuffle, Zstandard, LZ4 or Blosc; dimension-scale references; values
  streamed from a `Source`, so datasets need not be memory-resident.
- **I/O model:** memory-mapped files; hyperslab reads decode only intersecting chunks, in
  parallel under a memory budget (`rayon`).
- **Hardened against untrusted input:** bounds-checked reads, checked size arithmetic,
  depth-limited tree walks and a decode-size limit; malformed files yield `Error`, not
  panics.

```rust,no_run
use annoray_hdf5::{File, Node};

let file = File::open("data.h5")?;
for (name, node) in file.root()?.members()? {
    if let Node::Dataset(ds) = node {
        println!("{name}: {:?} {:?}", ds.shape(), ds.read()?.dtype());
    }
}
# Ok::<(), annoray_hdf5::Error>(())
```

```rust
use annoray_hdf5::{Array, WDataset, WGroup, WNode, write_file};
use ndarray::array;

let mut temp = WDataset::new(Array::F64(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()));
temp.chunks = Some(vec![1, 2]);
temp.deflate = Some(4);
let root = WGroup {
    members: vec![("temp".into(), WNode::Dataset(temp))],
    ..WGroup::default()
};
let bytes = write_file(&root)?;
assert!(annoray_hdf5::is_hdf5(&bytes));
# Ok::<(), annoray_hdf5::Error>(())
```

| Feature | Default | |
|---|---|---|
| `rayon` | no | Parallel chunk decode/encode. |
| `zstd` | no | Zstandard (filter 32015). |
| `lz4` | no | LZ4 (filter 32004). |

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at the licensee's
option.
