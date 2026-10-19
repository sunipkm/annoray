# annoray

**Annotated arrays**: the xarray data model (labelled N-dimensional arrays with named
dimensions, coordinates and attributes) in Rust, with physical units built in and Python
bindings that follow xarray's API. netCDF-3, netCDF-4/HDF5 and Zarr are read and written by
pure-Rust backends, with no C dependencies.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/installation
:link-type: doc

Installation, a quick overview in Rust and Python, and what changes for code written for
xarray.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every Python class, method and function. The Rust API is on
[docs.rs](https://docs.rs/annoray).
:::

:::{grid-item-card} Differences from xarray
:link: xarray-differences
:link-type: doc

Where annoray departs from xarray, and why: unit-checked arithmetic, no unchecked casts,
immutable values.
:::

:::{grid-item-card} Performance
:link: performance
:link-type: doc

Benchmarks against xarray with bottleneck, flox, numbagg and dask on a 2.3 GB field.
:::
::::

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/overview
getting-started/examples
xarray-differences
```

```{toctree}
:hidden:
:caption: User guide

user-guide/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
rust
performance
```
