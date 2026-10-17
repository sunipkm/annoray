# Performance

annoray is compared against xarray configured for speed: xarray with bottleneck, flox, numbagg
and dask installed, as a performance-minded user would have it. The script is
[`annoray-python/bench/bench_large.py`](https://github.com/sunipkm/annoray/blob/main/annoray-python/bench/bench_large.py). It checks
every annoray result against xarray's before timing it.

## Setup

| | |
|---|---|
| Machine | AMD Ryzen 7 9700X (8 cores, 16 threads), 60 GB RAM, Linux 7.0 |
| Software | Python 3.14.4, NumPy 2.5.4, xarray 2026.9.0, bottleneck 1.6.0, flox 0.11.2, numbagg 0.9.6, dask 2026.8.0, netCDF4 1.7.4, h5netcdf 1.8.1, zarr 3.4.1 |
| Data | A year of hourly 2 m temperature on a 1° grid: time 8760 × lat 181 × lon 360, `float32`, 2.3 GB of values |
| Files | netCDF-4 uncompressed (2.28 GB; chunks 24 × 181 × 360), netCDF-4 zlib (1.28 GB), Zarr v3 zstd (1.83 GB; same chunks) |

How the numbers were measured:

- **Times** are medians of 3 runs, after a warm-up run, with the page cache warm.
- **Peak memory** is the process's peak anonymous memory while the timed runs execute,
  sampled from `/proc/self/status`. It counts allocations only, not the pages of
  memory-mapped files, which the operating system can reclaim.
- **Isolation:** each file case runs in a fresh process, and one result is alive at a time.

```sh
python bench_large.py --generate             # write the files once (~5.4 GB)
python bench_large.py --repeat 3 --json out.json
python bench_large.py --only io              # or: --only compute
```

## Files

| file | operation | xarray | | annoray | |
|---|---|---|---|---|---|
| | | time | memory | time | memory |
| uncompressed | open + load all | 0.251 s (h5netcdf 0.191 s) | 4.61 GB | 0.217 s | 2.29 GB |
| zlib | open + load all | 2.071 s | 4.62 GB | 0.592 s | 2.54 GB |
| Zarr | open + load all | 0.499 s | 7.02 GB | 0.988 s | 2.95 GB |
| uncompressed | mean over time from disk | 0.518 s (dask) | 1.68 GB | 0.258 s | 0.16 GB |
| zlib | mean over time from disk | 2.253 s (dask) | 1.41 GB | 0.681 s | 0.40 GB |
| Zarr | mean over time from disk | 0.617 s (dask) | 0.64 GB | 1.230 s | 0.44 GB |
| uncompressed | one point, all times | 0.105 s | 0.18 GB | 0.011 s | 0.06 GB |
| zlib | one point, all times | 1.954 s | 0.19 GB | 0.380 s | 0.28 GB |
| uncompressed | one day, whole grid | 0.003 s | 0.09 GB | 0.001 s | 0.07 GB |
| — | write netCDF-4 from memory | 4.649 s | 2.48 GB | 4.841 s | 2.31 GB |
| — | write Zarr v3 from memory | 1.190 s | 2.48 GB | 4.513 s | 2.76 GB |

- **Reading.**
  - annoray loads with one copy of the data. Chunks are decoded in parallel straight into
    the result.
  - Streamed reductions keep a few slabs in memory.
- **Writing.**
  - Values that encoding leaves unchanged are borrowed from the dataset, not copied.
  - The netCDF-4 write is bound by the disk on this machine.
- **Zarr.** annoray's Zstandard is pure Rust (`ruzstd`). Its decoder is about 2× slower than
  libzstd's and its encoder about 4× slower, with about 20 % larger output. Zarr reads and
  writes with zstd are bound by it.

## In memory

Each operation runs on the loaded 2.3 GB array. The speedup is xarray's time divided by
annoray's.

| operation | xarray | annoray | speedup |
|---|---|---|---|
| mean over time | 0.061 s | 0.053 s | 1.15× |
| std over time | 0.113 s | 0.088 s | 1.28× |
| max over lat/lon | 1.487 s | 1.255 s | 1.18× |
| monthly climatology | 0.105 s | 0.044 s | 2.37× |
| anomaly from climatology | 0.460 s | 0.218 s | 2.11× |
| daily means (resample 1D) | 0.082 s | 0.078 s | 1.05× |
| rolling 24-h mean | 0.175 s | 0.268 s | 0.65× |
| rolling_exp mean (span 24) | 0.758 s | 0.529 s | 1.43× |
| 90th percentile over time | 0.819 s | 0.312 s | 2.63× |
| K → °C | 0.131 s | 0.148 s | 0.88× |
| a * 2 + a | 0.297 s | 0.337 s | 0.88× |
| where(a > 0 °C, a, 0 °C) | 0.473 s | 0.401 s | 1.18× |
| corr over time | 1.849 s | 1.017 s | 1.82× |
| convert_calendar noleap | 0.287 s | 0.289 s | 0.99× |
| sel box + month | <0.001 s | 0.002 s | — |
| interp to 0.5° lat | 0.210 s | 0.161 s | 1.31× |
| astype float64 | 0.233 s | 0.246 s | 0.95× |
| equals (copy) | 0.460 s | 0.227 s | 2.03× |

- **Rolling means.** xarray's time comes from numbagg's parallel JIT kernels; with bottleneck
  alone it takes 10 s.
- **Elementwise operations** (`K → °C`, `a * 2 + a`, `astype`) are bound by memory bandwidth
  in both libraries.
- **`sel`.** The selection itself is cheap in both libraries; annoray carries about 2 ms of
  fixed overhead.

## Notes

- **Threads.** annoray parallelises with rayon; set `RAYON_NUM_THREADS` to limit it. For
  example, the rolling mean takes 1.25 s on one thread and 0.27 s on 16.
- **Streaming budget.** The memory budget for streaming lazy data is set with
  `ar.set_stream_bytes` (Python) or `annoray::set_stream_bytes` (Rust).
- **OpenBLAS threads.** The benchmark sets `OPENBLAS_NUM_THREADS=1`. Otherwise OpenBLAS
  threads, which scipy and NumPy load, spin idle and take cores from both libraries.
