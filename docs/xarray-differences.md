# Differences from xarray

annoray follows xarray's data model and API closely enough that most xarray code runs
unchanged against `import annoray as ar`. Where it differs, the difference is deliberate:
xarray behaviours that silently produce wrong results (unchecked casts, unit-blind
arithmetic, writes that reach some objects and not others) are replaced by checked behaviour
that raises an error naming the explicit alternative. Known xarray bugs are not reproduced.

This page lists the differences by topic. Each entry gives annoray's behaviour, then xarray's.

## Data model and assignment

- **Values are immutable from Python.** `da[...] = v`, `da.loc[...] = v` and
  `da.values[...] = v` raise `TypeError`; `.values` is a read-only NumPy view of the Rust
  buffer. `da.put(value, **positions)` and `da.put_sel(value, method=, tolerance=, **labels)`
  return an updated copy and leave the original unchanged.
  - *xarray:* in-place writes. Whether a write also reaches other objects depends on how they
    were selected: basic indexing gives views, fancy indexing gives copies. Chained assignment
    (`ds.sel(...)["v"][0] = 1`) is silently lost.
  - `put` is stricter than xarray assignment:
    - positions must be in range and not repeated;
    - a dimension coordinate of `value` must match the target at the selected positions;
    - `value` is converted into the target's unit;
    - the element type never changes (`2.5` into an `int32` array is an error, not `2`).
  - Bulk edits: `arr = da.values.copy()`, edit `arr`, then `da.copy(data=arr)`. Datasets:
    `ds.assign(v=ds.v.put(...))`.
  - In Rust, data is mutated in place through `data_mut` / `isel_mut`, where the borrow
    checker rules out aliasing.
- **Containers are assignable.** These work as in xarray: `ds["v"] = ...`, `del ds["v"]`,
  `ds.update(...)`, `ds.coords[k] = ...`, `da.coords[k] = ...`, `da.name = ...`, and edits of
  `attrs` and `encoding`.
  - Every edit is validated before anything changes, so a refused edit leaves the object as
    it was.
  - `attrs`, `coords` and `encoding` are live `dict` subclasses.
  - Attributes that are not edited keep their exact netCDF types (`int16`, `float32`, arrays).
- **Arrays taken from a dataset are linked copies.** `ds["v"].attrs[...] = ...` writes
  through to the dataset, unless the dataset has since replaced `v`.
  - *xarray:* the array shares the whole variable with the dataset.
- **Single-coordinate indexes only.** Label lookups run on the dimension coordinate; there
  are no separate index objects.
  - These raise an error with a hint:
    - MultiIndexes: `set_index` with several coordinates or `append=True`;
    - `to_stacked_array`;
    - `set_xindex`;
    - `reset_index` without `drop=True`.
  - `stack` / `unstack` work without a tuple-valued coordinate.
- **`broadcast` / `broadcast_like` return copies.**
  - *xarray:* read-only views.
- **`rename_vars` on a dimension coordinate makes it a plain coordinate.**
  - *xarray:* keeps it as an index under the new name.

## Units

Units belong to the data, not to an attribute: `da.units` is a unit, and a CF `units`
attribute is decoded into it when a file is opened.

- **Values stay in their declared unit.** `mm` data is never normalised to `m`.
- **`+`, `-` and comparisons check dimensions.**
  - Operands with the same dimension but a different scale (`m` and `km`) are an error by
    default.
  - `UnitPolicy::AutoRescale` (Rust) converts to the left operand's unit instead.
- **Bare numbers next to unitful data.**
  - A bare number is accepted only as `0` (for ratio-scale and logarithmic units) or NaN.
  - Any other value must be given as a `Quantity`. This applies to comparisons, `where`,
    `fillna`, `clip`, `isin`, `put` and `query`. For example, `da > 280` on kelvin data raises `UnitError`;
    `da > ar.Quantity(280, "K")` works.
  - *xarray:* compares the stored numbers, whatever the unit.
- **Affine and logarithmic units.**
  - `degC - degC` is a kelvin-sized step; `degC + degC`, products of °C and sums of °C are
    errors.
  - `dB` and other logarithmic units convert only between the same reference.
- **NumPy ufuncs.** The unit-aware ufuncs run in Rust. Others raise `TypeError` on data with a
  unit instead of dropping the unit.
- **Math functions.** Transcendental functions need dimensionless input. `sin`, `cos` and
  `tan` take angles. Inverse trigonometric functions of unitful input return radians.
- **Python scalars.** `float(da)` and `int(da)` work only on dimensionless data. `da.item()`
  returns a `Quantity` when there is a unit.
- **Calculus.**
  - Derivatives carry the data unit per coordinate unit.
  - With a time coordinate, the step is in seconds (or `time_unit`) and the result carries
    that unit.
  - *xarray:* nanoseconds, with no unit.
- **Comparing and combining.**
  - `equals` converts units (1 km equals 1000 m); `identical` requires the same unit.
  - `concat`, `merge` and `combine_first` convert compatible units into the first object's
    unit.

## Element types

- **No unchecked casts.**
  - `astype` raises `ValueError` ("cannot convert ...") for NaN or ±∞ converted to an
    integer, for values out of range (300 as `uint8`), and for finite `float64` values
    beyond `float32`'s range.
  - *NumPy:* `INT_MIN`, saturation or wrap-around (300 as `uint8` is 44).
- **`full_like`** with an integer or boolean template needs an exact fill.
  - *NumPy:* truncates `0.5` to `0`.
- **Integer arithmetic is overflow-checked.** Integer sums are exact (`int64` / `uint64`).
- **Label types are not mixed.** Integer and float coordinates, or coordinates in different
  units, are not aligned by value; the error asks for an explicit conversion.
- **Plain Python numbers are weak scalars** (NEP 50): `float32` data stays `float32` in
  `da * 2.0`.
- **NumPy reductions** (`np.sum(da, ...)`) refuse `out=` and `dtype=` arguments.

## Indexing and selection

- **`.loc[...]` is read-only.** Use `put_sel` to write by label.
- **Rust selection macros.** `isel!` / `sel!` take Python slice syntax and return `Result`;
  they never panic.
- **`interp`.**
  - No extrapolation by default.
  - A NaN target gives NaN.
  - Text coordinates along the interpolated dimension are dropped.
  - annoray additions: log-scale interpolation (`method="log_coord"`, `"log_value"`,
    `"loglog"`) and extrapolation choices (`extrapolate="edge"`, `"extend"`, or a number for
    a power law).
- **`query`** uses a restricted expression evaluator: no `eval`, no pandas. Calls are limited
  to `Q(...)` quantities and `.str` methods.

## Alignment and combining

- **`concat` / `merge` use xarray's new defaults:** `join="exact"`, `compat="override"`,
  `coords="minimal"`. `ConcatOptions::legacy()` / `MergeOptions::legacy()` (Rust) restore
  the old outer-join behaviour.
- **Dataset with DataArray arithmetic** does not merge the array's attributes into the
  dataset's global attributes.
- **`groupby` on a dataset** keeps variables without the grouped dimension as they are.
  - *xarray:* repeats them for every group.

## Computation

- **Attributes are kept everywhere**, matching xarray 2026's `keep_attrs` default. Binary
  operations merge both operands' attributes and drop keys whose values conflict.
- **NaT is skipped by every reduction**, including `min` / `max` of dates.
- **Empty bins** (`resample`, `groupby_bins`) reduce as empty arrays do: sum 0, count 0, mean
  NaN.
- **`resample`** accepts unsorted times and NaT; NaT belongs to no bin.
  - *xarray:* requires a monotonic index.
- **`rolling_exp`** is native Rust, with no numba or numbagg dependency.
  - `var`, `std`, `cov` and `corr` use a numerically robust shifted algorithm. numbagg's
    `E[x²] − E[x]²` loses every digit when values are large compared with their spread.
  - `min_weight=` is honoured.
  - *xarray:* silently ignores `min_weight=`.
- **`polyfit`**
  - Residuals and covariance are computed per lane from the points actually used.
  - Lanes with too few points, or that are rank-deficient, give NaN instead of raising.
  - The solver is an internal SVD, with no LAPACK.
- **`interpolate_na`**
  - `max_gap` works without a coordinate.
  - The scipy-only methods are not provided: quadratic, polynomial, spline, barycentric and
    krogh.
- **`diff(n > 1, label="lower")`** applies the label at every step.
  - *xarray:* drops it after the first step.
- **`isin(set)`** treats the set as its elements.
- **`weighted`** rejects NaN weights.
- **No dask: everything computes eagerly.** Data opened from files stays lazy until used:
  elementwise operations compose, and reductions stream through the data under a memory
  budget (`ar.set_stream_bytes`).
  - `chunk`, `persist` and `unify_chunks` do nothing, and `chunks=` is accepted and ignored.
  - `set_options` accepts xarray's display and performance options without effect.
    Options that would change results are accepted only at annoray's fixed values.

## Time and calendars

- **Dates in the standard calendar** decode to `datetime64` only when every value is on or
  after 1582-10-15. Otherwise they decode to CF dates, never silently shifted.
- **`months since` / `years since` units are rejected**, as cftime does.
- **Masked integer times in non-standard calendars** decode to NaT.
  - *xarray:* raises `OverflowError` inside cftime.
- **Durations.** Variables with time units but no `since` stay quantities; `to_timedelta()`
  converts them.
- **`date_range`** in non-standard calendars supports fixed steps and period starts only.
- **`convert_calendar`** does not support `align_on="random"`.

## Text

- **Text is NumPy fixed-width `<U{n}`** everywhere: `.values`, coordinates, `to_xarray`,
  and pandas and dict conversions.
  - *xarray:* object arrays, `<U`, or `StringDType`, depending on the source.
- **Text has no missing value.** `reindex` and `pad` fill text with `""`.
  - *xarray:* NaN objects.
- **The `.str` accessor**
  - refuses `encode`, `decode` and `findall` (there is no bytes type) and `split(dim=None)`;
  - `extractall` labels its dimensions by what they index (xarray swaps `group_dim` and
    `match_dim`).

## I/O and encoding

- **Pure-Rust backends:** netCDF-3 (CDF-1/2/5), netCDF-4/HDF5 and Zarr v2/v3 are read and
  written without the C libraries. `engine` accepts only these formats, and unknown keywords
  are errors.
- **netCDF-3 writes types xarray refuses.** Unsigned integers are stored with
  `_Unsigned="true"`, and `int64` is narrowed to `int32` when the values fit.
- **Stacked dimensions** are written as CF "compression by gathering" (the cf_xarray layout)
  and decoded back.
- **Files are flushed to the operating system on close, not `fsync`ed**, as netCDF-C's
  `nc_close` does.
- **`to_netcdf(mode="a")`** writes a temporary file and moves it over the original, so a
  failed append leaves the file as it was. With `append_dim` or `region` (xarray has these
  for Zarr only) the file is written in place instead: see
  [Writing in place](user-guide/io.md#writing-in-place) and `NetcdfWriter`.
- **Opening** also takes the bytes of a file and binary file-like objects (read whole).
- **`decode_coords="all"`** keeps a CF attribute that names a variable the file lacks, with a
  warning. *xarray:* drops the attribute.
- **`close()` and `with` blocks do nothing.** Files are memory-mapped and released with the
  object.
- **HDF5.**
  - Compound variables are read as one variable per field (`var.field`) and variable-length
    variables as CF contiguous ragged arrays; both are written back in their own form.
    Reference, opaque and top-level array types are reported as unsupported, not misread.
  - Enumerations keep their names (in `encoding["dtype"].metadata`, as in xarray); h5py's
    booleans read as booleans.
  - Virtual datasets are read; a missing source file is an error. *HDF5:* fills its part
    with the fill value.
  - bzip2, BloscLZ, LZF, Bitshuffle, n-bit, scale-offset, szip, Blosc2, ZFP, SZ and SZ3
    chunks are read but not written.
- **Zarr.**
  - Local directory stores only.
  - Appending, region writes and complex types are not supported.
  - Arrays without dimension names are skipped with a warning.

## DataTree

- **Nodes are views of a shared tree.** Edits by path are validated before anything changes.
  - A node removed from its tree is no longer reachable through views taken before the
    removal; take `copy()` first.
  - *xarray:* the removed node lives on as its own tree.
- **`tree["a/v"]` returns a copy**, so editing its attributes does not change the tree.
- **`where` on DataTree** is an annoray addition.

## Python API

- `import annoray` loads neither xarray nor pandas. The dependencies are NumPy and
  matplotlib (imported only for plotting); xarray and astropy are optional extras.
- xarray's `Variable`, `IndexVariable`, `Coordinates` and `CFTimeIndex` classes are not
  provided.
- Unknown attributes cannot be set on objects. Accessors registered with
  `register_*_accessor` work.
- `len()` of a 0-d array raises `TypeError`.

## Not supported

- `curvefit` (it wraps scipy's `curve_fit`; annoray does not depend on scipy).
- MultiIndexes, `to_stacked_array`, `set_xindex`, `unstack(sparse=True)`.
- Remote Zarr stores, Zarr appends and region writes.
- SZ and SZ3 chunks of integer data; SZ's non-regression 2-D/3-D Lorenzo path and its
  point-wise relative mode without acceleration (not written with the default settings).
- Writing in place to HDF5 datasets indexed by fixed or extensible arrays or version-2
  B-trees (files written with h5py's `libver="latest"`).
- Complex numbers.
- dask arrays.
- Free-threaded CPython builds.
- Plotting with cftime axes, `hue_style` / marker-size legends, `lines()` and faceted `hist`.
