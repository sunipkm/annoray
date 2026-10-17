# Examples

Each example exists in Rust and in Python, and both print the same lines. The sources are
in [`annoray/examples`](https://github.com/sunipkm/annoray/tree/main/annoray/examples) and
[`annoray-python/examples`](https://github.com/sunipkm/annoray/tree/main/annoray-python/examples):

```sh
cargo run --release --example quickstart   # in annoray/
python examples/quickstart.py              # in annoray-python/
```

## Quickstart

Building arrays and datasets, selecting, arithmetic with units, reductions.

::::{tab-set}
:::{tab-item} Python
```{literalinclude} ../../annoray-python/examples/quickstart.py
:language: python
```
:::
:::{tab-item} Rust
```{literalinclude} ../../annoray/examples/quickstart.rs
:language: rust
```
:::
::::

## Quantities

Scalars with units, dimension-checked at run time in Python and at compile time in Rust.

::::{tab-set}
:::{tab-item} Python
```{literalinclude} ../../annoray-python/examples/quantities.py
:language: python
```
:::
:::{tab-item} Rust
```{literalinclude} ../../annoray/examples/quantities.rs
:language: rust
```
:::
::::

## Weather

Grouping, resampling and climatologies on a synthetic weather record.

::::{tab-set}
:::{tab-item} Python
```{literalinclude} ../../annoray-python/examples/weather.py
:language: python
```
:::
:::{tab-item} Rust
```{literalinclude} ../../annoray/examples/weather.rs
:language: rust
```
:::
::::

## Disk to disk

Compressed netCDF out, lazy reads, and computations streamed from disk without loading the
data.

::::{tab-set}
:::{tab-item} Python
```{literalinclude} ../../annoray-python/examples/disk_to_disk.py
:language: python
```
:::
:::{tab-item} Rust
```{literalinclude} ../../annoray/examples/disk_to_disk.rs
:language: rust
```
:::
::::
