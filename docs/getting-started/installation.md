# Installation

## Python

```sh
pip install annoray
```

Wheels use CPython's stable ABI: one per platform covers CPython 3.11 and later on Linux,
macOS and Windows. The dependencies are NumPy and matplotlib (imported only for plotting).
`import annoray` loads neither xarray nor pandas.

Optional extras:

| Extra | Adds |
|---|---|
| `annoray[xarray]` | `to_xarray()` / `from_xarray()`, and an xarray backend (`xr.open_dataset(path, engine="annoray")`). |
| `annoray[astropy]` | `to_astropy()` / `from_astropy()` on units, quantities and arrays; astropy units accepted wherever units are. |

Building from source needs a Rust toolchain (1.97 or later) and
[maturin](https://www.maturin.rs):

```sh
git clone https://github.com/sunipkm/annoray
cd annoray/annoray-python
pip install maturin
maturin develop --release
```

## Rust

```toml
[dependencies]
annoray = "0.1"
```

| Feature | Default | Adds |
|---|---|---|
| `rayon` | yes | Data-parallel kernels, and parallel HDF5/Zarr chunk (de)compression. |
| `zstd`, `lz4` | yes | Zstandard and LZ4 chunk codecs (pure Rust). |
| `uom` | no | Arrays and `sel` labels of uom SI quantities. |
| `whippyunits` | no | Arrays and `sel` labels of whippyunits quantities; zero-copy typed views. |
| `netcdf-c` | no | The netCDF-C/HDF5 C libraries as an alternative backend (static build; needs cmake). |

The units, HDF5, Zarr and codec layers are standalone crates as well: see
[Rust crates](../rust.md).
