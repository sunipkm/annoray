//! Methods implemented in Python and exposed here: pandas and dict conversions
//! (`annoray._pandas`) and compositions of native operations (`annoray._methods`).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use crate::native::{PyDataArray, PyDataset, PyGroupBy};
use crate::tree::PyDataTree;

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
    ($ty:ident: $($rest:tt)*) => {
        getters!($ty, "annoray._methods": $($rest)*);
    };
    ($ty:ident, $module:literal: $($(#[$doc:meta])* $method:ident => $func:literal),* $(,)?) => {
        #[pymethods]
        impl $ty {
            $(
                $(#[$doc])*
                #[getter]
                fn $method<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                    let py = slf.py();
                    forward(py, $module, $func, slf.clone().into_any(), &PyTuple::empty(py), None)
                }
            )*
        }
    };
}

forwarders!(PyDataArray, "annoray._methods":
    /// A copy with the values at positions replaced: `da.put(value, time=3)`,
    /// `da.put(q, {"x": [0, 2], "y": slice(1, None)})` (keys as for `isel`). The original
    /// is unchanged (annoray's values are immutable). `value` is a DataArray (broadcast by
    /// dimension name; its dimension coordinates must match the selection's), a number or
    /// quantity, or an array shaped like the selection. It is converted into the array's
    /// unit (a bare number only as 0 or NaN beside a unit); the element type never changes
    /// (an integer array takes only exact integers that fit). A position may not repeat.
    put => "da_put",
    /// `put` with labels (keys as for `sel`, with `method=` and `tolerance=`):
    /// `da.put_sel(Quantity(0, "degC"), time="2001-07-01")`.
    put_sel => "da_put_sel",
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

macro_rules! common_extras {
    ($ty:ident) => {
        forwarders!($ty, "annoray._methods":
            /// Pickled as a plain-data state (values, units, attributes, encoding); lazy
            /// data is read.
            __reduce__ => "reduce_pickle",
            /// A copy.
            __copy__ => "shallow_copy",
            /// A copy (data is never shared between annoray objects).
            __deepcopy__ => "deepcopy",
            /// No-op: a file is released when the data read from it is dropped.
            close => "close",
            __enter__ => "enter",
            __exit__ => "exit_",
            /// The text repr, for notebooks.
            _repr_html_ => "repr_html",
        );
        getters!($ty:
            /// The real part (of real data: the data itself).
            real => "real",
            /// The imaginary part (of real data: zeros).
            imag => "imag",
            /// Dimension → pandas index, as `indexes`.
            xindexes => "indexes",
        );
        forwarders!($ty, "annoray._methods":
            /// The complex conjugate (of real data: the data itself).
            conj => "conj",
            /// The complex conjugate (of real data: the data itself).
            conjugate => "conj",
        );
    };
}

common_extras!(PyDataArray);
common_extras!(PyDataset);

forwarders!(PyDataset, "annoray._methods":
    /// A netCDF CDL-like summary, written to `buf` (standard output by default).
    info => "info",
);

getters!(PyDataset:
    /// Every variable (data variables and coordinates), by name.
    variables => "variables",
);

forwarders!(PyGroupBy, "annoray._methods":
    /// `func(values, axis=…)` over each group (the grouped dimension by default).
    reduce => "groupby_reduce",
    /// Quantiles of each group (over the grouped dimension by default).
    quantile => "groupby_quantile",
);

forwarders!(PyDataArray, "annoray._methods":
    /// Drop positions whose label repeats (`keep="first"`, `"last"` or `False`).
    drop_duplicates => "drop_duplicates",
);

forwarders!(PyDataset, "annoray._methods":
    /// Drop positions whose label repeats (`keep="first"`, `"last"` or `False`).
    drop_duplicates => "drop_duplicates",
);

macro_rules! index_methods {
    ($ty:ident) => {
        forwarders!($ty, "annoray._methods":
            /// Make a coordinate the index of a dimension: `ds.set_index(x="lab")`.
            set_index => "set_index",
            /// Remove dimension indexes (`drop=True`).
            reset_index => "reset_index",
            /// Not supported (selection is by dimension coordinates); see `set_index`.
            set_xindex => "set_xindex",
        );
    };
}

index_methods!(PyDataArray);
index_methods!(PyDataset);

forwarders!(PyDataset, "annoray._methods":
    /// Not supported (the result needs a MultiIndex).
    to_stacked_array => "to_stacked_array",
);

getters!(PyDataTree, "annoray._tree":
    /// Child name → node.
    children => "children",
    /// This node's dataset with the coordinates it inherits (a copy).
    ds => "ds",
    /// Same as `ds`.
    dataset => "ds",
    /// The data variables (with inherited coordinates).
    data_vars => "data_vars",
    /// The coordinates, inherited ones included.
    coords => "coords",
    /// The dimensions, inherited ones included.
    dims => "dims",
    /// Dimension → length.
    sizes => "sizes",
    /// This node's attributes, as a dict whose changes apply to the node.
    attrs => "attrs",
    /// This node's dataset encoding (`unlimited_dims`), as a live dict.
    encoding => "encoding",
    /// This node and every node below it, breadth first.
    subtree => "subtree",
    /// Every node below this one, breadth first.
    descendants => "descendants",
    /// The nodes without children below (or at) this one.
    leaves => "leaves",
    /// The ancestors, nearest first.
    parents => "parents",
    /// This node and its ancestors, nearest first.
    lineage => "lineage",
    /// The root first, this node last.
    ancestors => "ancestors",
    /// The paths of `subtree`.
    groups => "groups",
    /// The deepest level in the tree.
    depth => "depth",
    /// The number of ancestors.
    level => "level",
    /// The number of nodes at this level of the tree.
    width => "width",
    /// Whether this node has no children.
    is_leaf => "is_leaf",
    /// Whether this node has variables of its own.
    has_data => "has_data",
    /// Whether this node has attributes.
    has_attrs => "has_attrs",
    /// No variables and no children.
    is_empty => "is_empty",
);

forwarders!(PyDataTree, "annoray._tree":
    /// This node's path relative to another node of the tree.
    relative_to => "relative_to",
    /// Whether another node is in the same tree.
    same_tree => "same_tree",
    /// The deepest node both are below.
    find_common_ancestor => "find_common_ancestor",
    /// The same structure (child names at every level).
    isomorphic => "isomorphic",
    /// Isomorphic, with equal datasets at every node.
    equals => "equals",
    /// Isomorphic, with identical datasets at every node, and the same name.
    identical => "identical",
    /// The nodes whose relative path matches a glob pattern, as a new tree.
    r#match => "match",
    /// The nodes for which a function holds, as a new tree.
    filter => "filter",
    /// The nodes whose relative paths exist in another tree.
    filter_like => "filter_like",
    /// A copy without the named children.
    drop_nodes => "drop_nodes",
    /// A copy without the nodes whose subtree holds no data.
    prune => "prune",
    /// `func(self, *args, **kwargs)`.
    pipe => "pipe",
    /// A copy with variables or children set on this node.
    assign => "assign",
    /// Read every lazily loaded value, in place; returns this node.
    load => "load",
    /// A copy with every value read into memory.
    compute => "compute",
    /// No-op without dask.
    chunk => "chunk",
    /// No-op without dask.
    persist => "chunk",
    /// This node's subtree as a new tree.
    copy => "copy",
    /// A copy.
    __copy__ => "copy",
    /// A copy.
    __deepcopy__ => "copy",
    /// Path → each node's own dataset.
    to_dict => "to_dict",
    /// `func(dataset)` for every node, as a tree of the results.
    map_over_datasets => "map_over_datasets",
    /// The text repr, for notebooks.
    _repr_html_ => "repr_html",
    /// Pickled as its datasets by path.
    __reduce__ => "reduce_pickle",
    /// Data variable names, then child names.
    keys => "keys",
    /// The values of `keys()`.
    values => "values",
    /// `(key, value)` pairs of `keys()`.
    items => "items",
    /// The node or variable at a key, or a default.
    get => "get",
    _plain_attrs => "_plain_attrs",
    _edit_attrs => "_edit_attrs",
    _plain_encoding => "_plain_encoding",
    _edit_encoding => "_edit_encoding",
    /// `Dataset.mean` applied to every node's dataset, as a new tree of the same shape.
    mean => "node_mean",
    /// `Dataset.sum` applied to every node's dataset, as a new tree of the same shape.
    sum => "node_sum",
    /// `Dataset.prod` applied to every node's dataset, as a new tree of the same shape.
    prod => "node_prod",
    /// `Dataset.std` applied to every node's dataset, as a new tree of the same shape.
    std => "node_std",
    /// `Dataset.var` applied to every node's dataset, as a new tree of the same shape.
    var => "node_var",
    /// `Dataset.min` applied to every node's dataset, as a new tree of the same shape.
    min => "node_min",
    /// `Dataset.max` applied to every node's dataset, as a new tree of the same shape.
    max => "node_max",
    /// `Dataset.median` applied to every node's dataset, as a new tree of the same shape.
    median => "node_median",
    /// `Dataset.count` applied to every node's dataset, as a new tree of the same shape.
    count => "node_count",
    /// `Dataset.any` applied to every node's dataset, as a new tree of the same shape.
    any => "node_any",
    /// `Dataset.all` applied to every node's dataset, as a new tree of the same shape.
    all => "node_all",
    /// `Dataset.isel` applied to every node's dataset, as a new tree of the same shape.
    isel => "node_isel",
    /// `Dataset.sel` applied to every node's dataset, as a new tree of the same shape.
    sel => "node_sel",
    /// `Dataset.squeeze` applied to every node's dataset, as a new tree of the same shape.
    squeeze => "node_squeeze",
    /// `Dataset.astype` applied to every node's dataset, as a new tree of the same shape.
    astype => "node_astype",
    /// `Dataset.fillna` applied to every node's dataset, as a new tree of the same shape.
    fillna => "node_fillna",
    /// `Dataset.dropna` applied to every node's dataset, as a new tree of the same shape.
    dropna => "node_dropna",
    /// `Dataset.interp` applied to every node's dataset, as a new tree of the same shape.
    interp => "node_interp",
    /// `Dataset.reindex` applied to every node's dataset, as a new tree of the same shape.
    reindex => "node_reindex",
    /// `Dataset.round` applied to every node's dataset, as a new tree of the same shape.
    round => "node_round",
    /// `Dataset.clip` applied to every node's dataset, as a new tree of the same shape.
    clip => "node_clip",
    /// `Dataset.quantile` applied to every node's dataset, as a new tree of the same shape.
    quantile => "node_quantile",
    /// `Dataset.diff` applied to every node's dataset, as a new tree of the same shape.
    diff => "node_diff",
    /// `Dataset.shift` applied to every node's dataset, as a new tree of the same shape.
    shift => "node_shift",
    /// `Dataset.roll` applied to every node's dataset, as a new tree of the same shape.
    roll => "node_roll",
    /// `Dataset.cumsum` applied to every node's dataset, as a new tree of the same shape.
    cumsum => "node_cumsum",
    /// `Dataset.cumprod` applied to every node's dataset, as a new tree of the same shape.
    cumprod => "node_cumprod",
    /// `Dataset.isnull` applied to every node's dataset, as a new tree of the same shape.
    isnull => "node_isnull",
    /// `Dataset.notnull` applied to every node's dataset, as a new tree of the same shape.
    notnull => "node_notnull",
    /// `Dataset.where` applied to every node's dataset, as a new tree of the same shape.
    r#where => "node_where",
    /// `Dataset.drop_vars` applied to every node's dataset, as a new tree of the same shape.
    drop_vars => "node_drop_vars",
    /// `Dataset.rename_vars` applied to every node's dataset, as a new tree of the same shape.
    rename_vars => "node_rename_vars",
    /// `Dataset.head` applied to every node's dataset, as a new tree of the same shape.
    head => "node_head",
    /// `Dataset.tail` applied to every node's dataset, as a new tree of the same shape.
    tail => "node_tail",
    /// `Dataset.thin` applied to every node's dataset, as a new tree of the same shape.
    thin => "node_thin",
);

fn tree_call<'py, A>(
    slf: &Bound<'py, PyDataTree>,
    func: &str,
    args: A,
) -> PyResult<Bound<'py, PyAny>>
where
    A: IntoPyObject<'py, Target = PyTuple, Output = Bound<'py, PyTuple>>,
    PyErr: From<A::Error>,
{
    let py = slf.py();
    let rest = args.into_pyobject(py)?;
    forward(
        py,
        "annoray._tree",
        func,
        slf.clone().into_any(),
        &rest,
        None,
    )
}

#[pymethods]
impl PyDataTree {
    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "getitem", (key,))
    }
    fn __setitem__(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        tree_call(slf, "setitem", (key, value)).map(|_| ())
    }
    fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        tree_call(slf, "delitem", (key,)).map(|_| ())
    }
    fn __contains__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        tree_call(slf, "contains", (key,))?.extract()
    }
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "keys", ())?.try_iter().map(Bound::into_any)
    }
    fn __len__(slf: &Bound<'_, Self>) -> PyResult<usize> {
        tree_call(slf, "length", ())?.extract()
    }
    #[setter(attrs)]
    fn set_attrs_py(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        tree_call(slf, "set_attrs", (value,)).map(|_| ())
    }
    fn __add__<'py>(slf: &Bound<'py, Self>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "add"))
    }
    fn __radd__<'py>(slf: &Bound<'py, Self>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "add", true))
    }
    fn __sub__<'py>(slf: &Bound<'py, Self>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "sub"))
    }
    fn __rsub__<'py>(slf: &Bound<'py, Self>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "sub", true))
    }
    fn __mul__<'py>(slf: &Bound<'py, Self>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "mul"))
    }
    fn __rmul__<'py>(slf: &Bound<'py, Self>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "mul", true))
    }
    fn __truediv__<'py>(
        slf: &Bound<'py, Self>,
        o: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "truediv"))
    }
    fn __rtruediv__<'py>(
        slf: &Bound<'py, Self>,
        o: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "truediv", true))
    }
    fn __pow__<'py>(
        slf: &Bound<'py, Self>,
        o: &Bound<'py, PyAny>,
        _m: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "pow"))
    }
    fn __and__<'py>(slf: &Bound<'py, Self>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "and"))
    }
    fn __or__<'py>(slf: &Bound<'py, Self>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "or"))
    }
    fn __xor__<'py>(slf: &Bound<'py, Self>, o: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "binary", (o, "xor"))
    }
    fn __neg__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "unary", ("neg",))
    }
    fn __pos__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "unary", ("pos",))
    }
    fn __abs__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "unary", ("abs",))
    }
    fn __invert__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        tree_call(slf, "unary", ("invert",))
    }
    fn __richcmp__<'py>(
        slf: &Bound<'py, Self>,
        o: &Bound<'py, PyAny>,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::basic::CompareOp as C;
        let name = match op {
            C::Lt => "lt",
            C::Le => "le",
            C::Gt => "gt",
            C::Ge => "ge",
            C::Eq => "eq",
            C::Ne => "ne",
        };
        tree_call(slf, "binary", (o, name))
    }
}
