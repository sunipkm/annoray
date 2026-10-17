# annoray-zarr

Pure-Rust Zarr v2 and v3 reader and writer for local directory stores; the Zarr layer of
[annoray](https://docs.rs/annoray), independent of it. Codecs come from
[annoray-codecs](https://docs.rs/annoray-codecs).

- **Formats:** v2 (`.zarray`, `.zgroup`, `.zattrs`; `C`/`F` order; `.` or `/` separators) and
  v3 (`zarr.json`; regular grids; `default` and `v2` chunk keys; `sharding_indexed`).
- **Types:** booleans, integers, floats, fixed-width bytes and UCS-4 text, variable-length
  UTF-8 text.
- **Codecs:** `bytes`, `transpose`, `vlen-utf8`, `sharding_indexed`, `blosc`, `gzip`, `zstd`,
  `crc32c`; numcodecs `zlib`, `lz4`, `shuffle`, `bz2` (read). Zstandard and LZ4 behind the
  `zstd`/`lz4` features.
- **I/O model:** region reads decode only intersecting chunks (missing chunks are the fill
  value), in parallel with the `rayon` feature; writes go chunk by chunk and skip all-fill
  chunks; consolidated metadata for xarray and zarr-python.
- **Hardened:** keys cannot escape the store, decoded sizes are bounded, arithmetic is
  checked; malformed stores give `Error`, not panics.
- **Checked against zarr-python 3** in both directions (`fixtures/gen_zarr.py`,
  `fixtures/check_zarr.py`).

```rust,no_run
use annoray_zarr::{Array, ArrayBuilder, DType, Store, ZarrFormat};

let store = Store::new("data.zarr");
let a = Array::open(&store, "temperature")?;
let block = a.read_region(&[0, 0], &[10, 10])?;

let out = ArrayBuilder::new(DType::F64, &[100, 100], &[50, 50])
    .create(&Store::new("out.zarr"), "x", ZarrFormat::V3)?;
# Ok::<(), annoray_zarr::Error>(())
```

| Feature | Default | |
|---|---|---|
| `rayon` | no | Decode the chunks of a region in parallel (four per thread in flight). |
| `zstd` | no | Zstandard (`zstd` codec, inside Blosc). |
| `lz4` | no | LZ4 (numcodecs `lz4`, inside Blosc). |

## License

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at the licensee's
option.
