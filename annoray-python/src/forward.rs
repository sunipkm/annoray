//! Methods implemented in Python and exposed here: pandas and dict conversions
//! (`annoray._pandas`) and compositions of native operations (`annoray._methods`).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use crate::native::{PyDataArray, PyDataset};

/// `<module>.<func>(*first, *args, **kwargs)`.
fn forward<'py>(
    py: Python<'py>,
    module: &str,
    func: &str,
    first: Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut all = vec![first];
    all.extend(args.iter());
    py.import(module)?
        .getattr(func)?
        .call(PyTuple::new(py, all)?, kwargs)
}

macro_rules! forwarders {
    ($ty:ident, $module:literal: $($(#[$doc:meta])* $method:ident => $func:literal),* $(,)?) => {
        #[pymethods]
        impl $ty {
            $(
                $(#[$doc])*
                #[pyo3(signature = (*args, **kwargs))]
                fn $method<'py>(
                    slf: &Bound<'py, Self>,
                    args: &Bound<'py, PyTuple>,
                    kwargs: Option<&Bound<'py, PyDict>>,
                ) -> PyResult<Bound<'py, PyAny>> {
                    forward(slf.py(), $module, $func, slf.clone().into_any(), args, kwargs)
                }
            )*
        }
    };
}

macro_rules! constructors {
    ($ty:ident: $($(#[$doc:meta])* $method:ident => $func:literal),* $(,)?) => {
        #[pymethods]
        impl $ty {
            $(
                $(#[$doc])*
                #[classmethod]
                #[pyo3(signature = (*args, **kwargs))]
                fn $method<'py>(
                    cls: &Bound<'py, PyType>,
                    args: &Bound<'py, PyTuple>,
                    kwargs: Option<&Bound<'py, PyDict>>,
                ) -> PyResult<Bound<'py, PyAny>> {
                    let py = cls.py();
                    py.import("annoray._pandas")?.getattr($func)?.call(args, kwargs)
                }
            )*
        }
    };
}

forwarders!(PyDataArray, "annoray._pandas":
    /// A pandas object: a scalar array for 0-d, a Series for 1-D, a DataFrame for 2-D
    /// (units in `.attrs["units"]`).
    to_pandas => "da_to_pandas",
    /// A Series indexed by the dimensions (a MultiIndex for several).
    to_series => "da_to_series",
    /// A DataFrame indexed by the dimensions, with the data and non-index coordinates as
    /// columns (`name=` for an unnamed array, `dim_order=`).
    to_dataframe => "da_to_dataframe",
    /// A dict of plain Python data (`data="list"`, `"array"`, or `False` for dtype and
    /// shape only); the unit is the `"units"` attribute.
    to_dict => "da_to_dict",
);

forwarders!(PyDataset, "annoray._pandas":
    /// A DataFrame indexed by the dimensions, with every variable broadcast to them as a
    /// column (non-index coordinates included; units in `.attrs["units"]`).
    to_dataframe => "ds_to_dataframe",
    /// A DataFrame for a one-dimensional dataset (a Series for a 0-d one).
    to_pandas => "ds_to_pandas",
    /// A dict of plain Python data; see `DataArray.to_dict`.
    to_dict => "ds_to_dict",
);

constructors!(PyDataArray:
    /// A DataArray from a pandas Series (a MultiIndex becomes several dimensions, filled
    /// with NaN where combinations are missing); units from `.attrs["units"]`.
    from_series => "da_from_series",
    /// A DataArray from a dict as `to_dict` makes (or xarray's).
    from_dict => "da_from_dict",
);

constructors!(PyDataset:
    /// A Dataset from a pandas DataFrame: the index becomes the dimensions, columns the
    /// variables; units from `.attrs["units"]`.
    from_dataframe => "ds_from_dataframe",
    /// A Dataset from a dict as `to_dict` makes (or xarray's).
    from_dict => "ds_from_dict",
);

macro_rules! getters {
    ($ty:ident: $($(#[$doc:meta])* $method:ident => $func:literal),* $(,)?) => {
        #[pymethods]
        impl $ty {
            $(
                $(#[$doc])*
                #[getter]
                fn $method<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                    let py = slf.py();
                    forward(py, "annoray._methods", $func, slf.clone().into_any(), &PyTuple::empty(py), None)
                }
            )*
        }
    };
}

forwarders!(PyDataArray, "annoray._methods":
    /// The first `n` positions along each dimension (5 by default): `da.head(x=2)`.
    head => "head",
    /// The last `n` positions along each dimension (5 by default).
    tail => "tail",
    /// Every `n`-th position along each dimension.
    thin => "thin",
    /// Drop positions: `da.drop_isel(x=[0, 2])`.
    drop_isel => "drop_isel",
    /// Non-index coordinates as data variables of a dataset (or dropped, `drop=True`).
    reset_coords => "reset_coords",
    /// A one-variable dataset (`name=`), or one variable per label of `dim`.
    to_dataset => "da_to_dataset",
    /// Interpolated onto another object's dimension coordinates.
    interp_like => "interp_like",
    /// Reindexed onto another object's dimension coordinates.
    reindex_like => "reindex_like",
    /// Gaps (and missing labels) filled from `other` (outer join; units converted).
    combine_first => "combine_first",
    /// Missing values replaced: a number (only 0 beside a unit), a quantity, or an array.
    fillna => "fillna",
    /// Pad along dimensions with numpy's modes: `da.pad(x=(1, 2), mode="edge")`.
    pad => "pad",
    /// Ranks along a dimension (ties averaged, NaN kept); unitless.
    rank => "rank",
    /// Cumulative reductions: `da.cumulative("t").mean()`.
    cumulative => "cumulative",
    /// `func(values, axis=…)` over dimensions; the unit is kept unless `output_units=`.
    reduce => "reduce",
    /// The pandas index of a dimension.
    get_index => "get_index",
    /// No-op without dask: returns the array (files are read lazily anyway).
    chunk => "chunk",
    /// No-op without dask.
    persist => "persist",
    /// Write as a one-variable netCDF file.
    to_netcdf => "da_to_netcdf",
    /// Write as a one-variable Zarr store.
    to_zarr => "da_to_zarr",
    /// `func(self, *args, **kwargs)`: one block without dask.
    map_blocks => "map_blocks",
);

forwarders!(PyDataset, "annoray._methods":
    /// The first `n` positions along each dimension (5 by default).
    head => "head",
    /// The last `n` positions along each dimension (5 by default).
    tail => "tail",
    /// Every `n`-th position along each dimension.
    thin => "thin",
    /// Drop positions: `ds.drop_isel(x=[0, 2])`.
    drop_isel => "drop_isel",
    /// Drop every variable along the named dimensions.
    drop_dims => "drop_dims",
    /// Non-index coordinates as data variables (or dropped, `drop=True`).
    reset_coords => "reset_coords",
    /// The data variables whose attributes match: `ds.filter_by_attrs(units="m")`.
    filter_by_attrs => "filter_by_attrs",
    /// The data variables stacked along a new dimension `"variable"`.
    to_dataarray => "to_dataarray",
    /// Same as `to_dataarray`.
    to_array => "to_dataarray",
    /// Interpolated onto another object's dimension coordinates.
    interp_like => "interp_like",
    /// Reindexed onto another object's dimension coordinates.
    reindex_like => "reindex_like",
    /// Gaps filled from `other` variable by variable (outer join).
    combine_first => "combine_first",
    /// Missing values replaced: a number, a quantity, or per variable (a Dataset or dict).
    fillna => "fillna",
    /// Pad along dimensions (every variable and coordinate).
    pad => "pad",
    /// Ranks along a dimension, for the variables that have it.
    rank => "rank",
    /// `func(values, axis=…)` over dimensions, variable by variable.
    reduce => "reduce",
    /// The pandas index of a dimension.
    get_index => "get_index",
    /// No-op without dask.
    chunk => "chunk",
    /// No-op without dask.
    persist => "persist",
    /// `func(self, *args, **kwargs)`: one block without dask.
    map_blocks => "map_blocks",
);

getters!(PyDataArray:
    /// Bytes of the data (not read for lazy data).
    nbytes => "nbytes",
    /// Dimension → pandas index, for dimensions with a coordinate.
    indexes => "indexes",
);

getters!(PyDataset:
    /// Bytes of all variables (not read for lazy data).
    nbytes => "nbytes",
    /// Dimension → pandas index, for dimensions with a coordinate.
    indexes => "indexes",
);

#[pymethods]
impl PyDataArray {
    /// `None`: annoray does not use dask.
    #[getter]
    fn chunks(&self) -> Option<()> {
        None
    }
}

#[pymethods]
impl PyDataset {
    /// `{}`: annoray does not use dask.
    #[getter]
    fn chunks<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        PyDict::new(py)
    }
}

#[pymethods]
impl PyDataArray {
    /// Vectorised string methods on text data: `da.str.upper()`, `da.str.contains("x")`, ….
    #[getter]
    fn str<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py()
            .import("annoray._strings")?
            .getattr("StringAccessor")?
            .call1((slf,))
    }
}

forwarders!(PyDataArray, "annoray._query":
    /// Positions along dimensions where boolean expressions hold:
    /// `da.query(x="x > 2")` (units-aware; `Q(value, unit)` for quantities).
    query => "query",
);

forwarders!(PyDataset, "annoray._query":
    /// Positions along dimensions where boolean expressions over variables hold:
    /// `ds.query(x="a > 2 & lab == 'q'")` (units-aware; `Q(value, unit)` for quantities).
    query => "query",
);

forwarders!(PyDataArray, "annoray._methods":
    /// Positions that sort the values along a numpy axis (NaN last); no unit.
    argsort => "argsort",
);
