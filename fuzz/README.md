# Fuzz targets

cargo-fuzz targets for the readers and the unit parser. Each input must give a result or an
error: a panic, an abort, a stack overflow or an allocation past the read limits is a bug.

| Target | Input |
|---|---|
| `read_bytes` | A netCDF-3, netCDF-4 or HDF5 file in memory, opened under `ReadLimits::untrusted()` (reads of at most 16 MiB, 64 MiB in all), lazily and eagerly, as a dataset and as a tree, with every variable read. |
| `hdf5` | The HDF5 reader alone: every dataset of every group read, under a 16 MiB load limit. |
| `units` | A unit string, or two separated by `\|`: parsing, printing and parsing back, arithmetic and conversion. |

The crate is its own workspace, built with nightly Rust and libFuzzer (`cargo install
cargo-fuzz`):

```sh
cd fuzz
cargo +nightly fuzz run -O read_bytes corpus/read_bytes ../fixtures/nc -- -rss_limit_mb=2048 -max_len=262144
cargo +nightly fuzz run -O hdf5 corpus/hdf5 ../fixtures/h5 ../fixtures/nc -- -rss_limit_mb=2048
cargo +nightly fuzz run -O units corpus/units
```

The fixtures seed the corpora; `corpus/` and `artifacts/` stay local. An input that fails is
kept in `fixtures/fuzz/` with a test that opens it (`annoray/tests/read_limits.rs`).
