# Rust crates

The Rust API is documented on docs.rs, versioned with each crates.io release.

| Crate | API | |
|---|---|---|
| `annoray` | [docs.rs/annoray](https://docs.rs/annoray) | Data model, operations, CF encoding and decoding, netCDF/HDF5/Zarr I/O. |
| `annoray-units` | [docs.rs/annoray-units](https://docs.rs/annoray-units) | Units: UDUNITS-2/CF grammar, exact conversion, CF calendars, dimension-typed quantities. |
| `annoray-hdf5` | [docs.rs/annoray-hdf5](https://docs.rs/annoray-hdf5) | Pure-Rust HDF5 reader and writer. |
| `annoray-zarr` | [docs.rs/annoray-zarr](https://docs.rs/annoray-zarr) | Pure-Rust Zarr v2/v3 reader and writer for local stores. |
| `annoray-codecs` | [docs.rs/annoray-codecs](https://docs.rs/annoray-codecs) | Compression codecs shared by the HDF5 and Zarr crates. |
| `annoray-macros` | [docs.rs/annoray-macros](https://docs.rs/annoray-macros) | `isel!` / `sel!`, re-exported by `annoray`. |

The standalone crates do not depend on `annoray`:

```text
annoray ──┬── annoray-units
          ├── annoray-macros
          ├── annoray-hdf5 ──┐
          └── annoray-zarr ──┴── annoray-codecs
```

## Storage representations

`Variable`, `DataArray` and `Dataset` are generic over a representation, as ndarray's arrays
are:

| owned | shared view | mutable view |
|---|---|---|
| `Variable` | `VariableView<'a>` | `VariableViewMut<'a>` |
| `DataArray` | `DataArrayView<'a>` | `DataArrayViewMut<'a>` |
| `Dataset` | `DatasetView<'a>` | `DatasetViewMut<'a>` |

Selections return views; the `_mut` variants write in place under the borrow checker.
`DataArray::put` returns a copy with values replaced, converted into the array's unit and
element type.

## Errors

No function panics on bad input: every failure is a `Result`. `unwrap`, `expect`, panicking
indexing and unchecked arithmetic are denied by lint across the workspace.
