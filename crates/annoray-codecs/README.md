# annoray-codecs

Pure-Rust compression codecs for scientific array formats, shared by
[annoray-hdf5](https://docs.rs/annoray-hdf5) (HDF5 filters) and
[annoray-zarr](https://docs.rs/annoray-zarr) (Zarr codecs); independent of both.

- **Blosc** (version-1 frames): BloscLZ, LZ4, zlib and Zstandard inside; byte and bit shuffle.
  Decode and encode (BloscLZ decode only).
- **bzip2**: decode, with block CRCs checked.
- **zlib / gzip**: decode (bounded; concatenated gzip members; CRC and length checked) and
  encode, via `miniz_oxide`.
- **Zstandard** and **LZ4** (raw blocks and numcodecs' `LZ4` framing) behind the `zstd` and
  `lz4` features, in pure Rust (`ruzstd`, `lz4_flex`).
- **Shuffles and checksums**: byte shuffle, bitshuffle's bit transposition, CRC-32 and
  CRC-32C.

Every decoder takes a limit on its output and treats input as untrusted: bounds-checked reads,
checked size arithmetic, and errors instead of panics.

```rust
use annoray_codecs::zlib::{gzip_compress, gzip_decompress};

let data = b"an example".repeat(100);
let packed = gzip_compress(&data, 6);
assert_eq!(gzip_decompress(&packed, data.len())?, data);
# Ok::<(), annoray_codecs::Error>(())
```

Licensed under MIT or Apache-2.0, at your option.
