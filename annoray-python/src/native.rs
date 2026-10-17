//! Native, units-aware `DataArray` and `Dataset` (and their group-by and rolling helpers),
//! backed by annoray.
//!
//! Objects are immutable: every operation returns a new object, so the values a numpy view
//! shows can never change under it. Heavy work runs with the GIL released.

use std::path::PathBuf;

use annoray::io::{
    DecodeOptions, MfOptions, OpenOptions, ToNetcdf, WriteOptions, open_dataset_with,
};
use annoray::{
    BinOp, BinOptions, ConcatOptions, DataArray, Dataset, Extrapolate, How, InterpMethod,
    InterpOptions, Join, LabelIndexer, MergeOptions, Name, ReduceOptions, Reduction, Repr,
    RollingOptions, TypedArray, Unit, UnitPolicy, Variable, VariableBase,
};
use numpy::PyArrayDyn;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::convert::{
    LabelSel, OrPy, PosSel, attr_from_py, attrs_to_py, label_from_py, label_sel_from_py,
    method_from_py, numpy_dtype_name, parse_unit, pos_from_py, quantity_parts, typed_from_py,
    typed_to_py, value_error,
};
use crate::mutate::{Lending, Source, live_dict};

/// A facade error as a Python exception.
pub(crate) fn xr_error(e: annoray::Error) -> PyErr {
    match e {
        annoray::Error::Core(c) => crate::convert::core_error(c),
        annoray::Error::Unit(u) => crate::convert::unit_error(&u),
        annoray::Error::Io(io) => pyo3::exceptions::PyOSError::new_err(io.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

// ----- building variables from Python -----

/// Dimension names: a string, a sequence of strings, or `None` (`dim_0`, `dim_1`, …).
fn dims_from_py(dims: Option<&Bound<'_, PyAny>>, ndim: usize) -> PyResult<Vec<String>> {
    match dims {
        None => Ok((0..ndim).map(|i| format!("dim_{i}")).collect()),
        Some(d) if d.is_instance_of::<PyString>() => Ok(vec![d.extract()?]),
        Some(d) => d.extract(),
    }
}

/// Values, attributes and unit as a variable. The unit comes from `units`, a pint-like
/// quantity, or a `"units"` attribute; an attribute that does not parse stays an attribute
/// (with a warning), as when reading files.
pub(crate) fn variable_from_py(
    py: Python<'_>,
    dims: Option<&Bound<'_, PyAny>>,
    data: &Bound<'_, PyAny>,
    attrs: Option<&Bound<'_, PyDict>>,
    units: Option<&str>,
) -> PyResult<Variable> {
    let (data, q_units) = match quantity_parts(data)? {
        Some((m, u)) => (m, Some(u)),
        None => (data.clone(), None),
    };
    let typed = typed_from_py(&data)?;
    let dims = dims_from_py(dims, typed.ndim())?;
    let mut var = Variable::from_typed(dims, typed).py_err()?;
    let mut unit = match units.map(str::to_owned).or(q_units) {
        Some(u) => Some(parse_unit(&u)?),
        None => None,
    };
    if let Some(attrs) = attrs {
        for (k, v) in attrs.iter() {
            let key: String = k.extract()?;
            if key == "units" && v.is_instance_of::<PyString>() {
                let text: String = v.extract()?;
                if unit.is_none() {
                    match text.parse::<Unit>() {
                        Ok(u) => {
                            unit = Some(u);
                            continue;
                        }
                        Err(e) => {
                            let msg = format!("units {text:?} kept as an attribute: {e}");
                            pyo3::PyErr::warn(
                                py,
                                &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                                &std::ffi::CString::new(msg).map_err(value_error)?,
                                1,
                            )?;
                        }
                    }
                } else {
                    continue;
                }
            }
            var.attrs_mut().insert(key.into(), attr_from_py(&v)?);
        }
    }
    var.set_unit(unit);
    Ok(var)
}

/// A variable given as `(dims, data[, attrs])`, a `DataArray`, or bare data (for a coordinate
/// named like its dimension).
pub(crate) fn spec_from_py(py: Python<'_>, name: &str, v: &Bound<'_, PyAny>) -> PyResult<Variable> {
    if let Ok(da) = v.cast::<PyDataArray>() {
        return Ok(da.borrow().inner.variable().to_owned());
    }
    if let Ok(t) = v.cast::<PyTuple>() {
        let dims = t.get_item(0)?;
        let data = t.get_item(1)?;
        let attrs = if t.len() > 2 {
            Some(t.get_item(2)?)
        } else {
            None
        };
        let attrs = attrs.as_ref().map(|a| a.cast::<PyDict>()).transpose()?;
        return variable_from_py(py, Some(&dims), &data, attrs, None);
    }
    let name_dims = PyTuple::new(py, [name])?.into_any();
    let probe = typed_from_py(v)?;
    let dims = match probe.ndim() {
        0 => PyTuple::empty(py).into_any(),
        1 => name_dims,
        _ => {
            return Err(PyValueError::new_err(format!(
                "give {name:?} as (dims, data): it is not 1-D"
            )));
        }
    };
    variable_from_py(py, Some(&dims), v, None, None)
}

pub(crate) fn dict_iter<'py>(
    d: Option<&Bound<'py, PyAny>>,
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    match d {
        None => Ok(Vec::new()),
        Some(d) if d.is_none() => Ok(Vec::new()),
        Some(d) => d
            .cast::<PyDict>()?
            .iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v)))
            .collect(),
    }
}

pub(crate) fn set_attrs(attrs: &mut annoray::Attrs, d: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    for (k, v) in dict_iter(d)? {
        attrs.insert(k.into(), attr_from_py(&v)?);
    }
    Ok(())
}

// ----- values to Python -----

/// A variable's values as numpy: a read-only view of the data when it is in memory and
/// numeric (no copy), otherwise converted (dates, strings) or read (lazy data).
fn values_of<'py>(owner: Bound<'py, PyAny>, var: &Variable) -> PyResult<Bound<'py, PyAny>> {
    let py = owner.py();
    if var.is_lazy() {
        let t = py.detach(|| var.to_owned().into_typed()).py_err()?;
        return typed_to_py(py, t);
    }
    let typed = var.typed().py_err()?;
    crate::mutate::mark_lent(&owner);
    macro_rules! view {
        ($a:expr) => {{
            // SAFETY: the data belongs to `owner`, a Python object that the array keeps alive
            // as its base. Data buffers are never written in place, and `owner` is marked as
            // having lent (`mark_lent` above), so a variable it later replaces or removes is
            // retired, not freed, while it lives (see `mutate`). The array is made read-only
            // below so Python cannot write through it.
            #[allow(unsafe_code)]
            let arr = unsafe { PyArrayDyn::borrow_from_array(&$a, owner) };
            arr.call_method("setflags", (), Some(&[("write", false)].into_py_dict(py)?))?;
            Ok(arr.into_any())
        }};
    }
    match &typed {
        TypedArray::F32(a) => view!(a),
        TypedArray::F64(a) => view!(a),
        TypedArray::I8(a) => view!(a),
        TypedArray::I16(a) => view!(a),
        TypedArray::I32(a) => view!(a),
        TypedArray::I64(a) => view!(a),
        TypedArray::U8(a) => view!(a),
        TypedArray::U16(a) => view!(a),
        TypedArray::U32(a) => view!(a),
        TypedArray::U64(a) => view!(a),
        TypedArray::Bool(a) => view!(a),
        _ => typed_to_py(py, typed.to_owned()),
    }
}

use pyo3::types::IntoPyDict;

fn unit_str<S: Repr>(v: &VariableBase<S>) -> Option<String> {
    v.unit().map(ToString::to_string)
}

fn sizes_dict<'py>(
    py: Python<'py>,
    sizes: impl IntoIterator<Item = (Name, usize)>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, n) in sizes {
        d.set_item(k.as_str(), n)?;
    }
    Ok(d)
}

pub(crate) fn new_da(py: Python<'_>, inner: DataArray) -> PyResult<Py<PyDataArray>> {
    Py::new(
        py,
        PyDataArray {
            inner,
            lending: Lending::default(),
            parent: None,
        },
    )
}

/// A data array copied from variable `name` of `owner` (a Dataset, or a DataArray for a
/// coordinate); writes to its attributes reach the source (see `mutate`).
pub(crate) fn new_child_da(
    owner: &Bound<'_, PyAny>,
    name: &str,
    inner: DataArray,
) -> PyResult<Py<PyDataArray>> {
    Py::new(
        owner.py(),
        PyDataArray {
            inner,
            lending: Lending::default(),
            parent: Source::of(owner, name),
        },
    )
}

pub(crate) fn new_ds(py: Python<'_>, inner: Dataset) -> PyResult<Py<PyDataset>> {
    Py::new(
        py,
        PyDataset {
            inner,
            lending: Lending::default(),
        },
    )
}

pub(crate) fn new_dataset(py: Python<'_>, inner: Dataset) -> PyResult<Py<PyDataset>> {
    new_ds(py, inner)
}

// ----- selection -----

pub(crate) type IselArgs<'py> = Vec<(String, Bound<'py, PyAny>)>;

pub(crate) fn merge_kwargs<'py>(
    indexers: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<IselArgs<'py>> {
    let mut out = dict_iter(indexers)?;
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            out.push((k.extract()?, v));
        }
    }
    Ok(out)
}

/// Something selectable: data arrays and datasets.
trait Select: Sized + Send + Sync {
    fn take_(&self, dim: &str, idx: &[isize]) -> annoray::Result<Self>;
    fn isel_(&self, ix: Vec<(String, annoray::Indexer)>) -> annoray::Result<Self>;
    fn sel_(&self, ix: Vec<(String, LabelIndexer)>) -> annoray::Result<Self>;
    fn sel_take_(&self, dim: &str, labels: Vec<LabelIndexer>) -> annoray::Result<Self>;
    fn isel_vec_(&self, ix: &[(&str, &DataArray)]) -> annoray::Result<Self>;
    fn sel_vec_(
        &self,
        ix: &[(&str, &DataArray)],
        method: annoray::Method,
        tolerance: Option<&annoray::Label>,
    ) -> annoray::Result<Self>;
}

impl Select for DataArray {
    fn take_(&self, dim: &str, idx: &[isize]) -> annoray::Result<Self> {
        self.take(dim, idx)
    }
    fn isel_vec_(&self, ix: &[(&str, &DataArray)]) -> annoray::Result<Self> {
        self.isel_vectorized(ix)
    }
    fn sel_vec_(
        &self,
        ix: &[(&str, &DataArray)],
        method: annoray::Method,
        tolerance: Option<&annoray::Label>,
    ) -> annoray::Result<Self> {
        self.sel_vectorized(ix, method, tolerance)
    }
    fn isel_(&self, ix: Vec<(String, annoray::Indexer)>) -> annoray::Result<Self> {
        Ok(self.isel(ix)?.to_owned())
    }
    fn sel_(&self, ix: Vec<(String, LabelIndexer)>) -> annoray::Result<Self> {
        Ok(self.sel(ix)?.to_owned())
    }
    fn sel_take_(&self, dim: &str, labels: Vec<LabelIndexer>) -> annoray::Result<Self> {
        self.sel_take(dim, labels)
    }
}

impl Select for Dataset {
    fn take_(&self, dim: &str, idx: &[isize]) -> annoray::Result<Self> {
        self.take(dim, idx)
    }
    fn isel_vec_(&self, ix: &[(&str, &DataArray)]) -> annoray::Result<Self> {
        self.isel_vectorized(ix)
    }
    fn sel_vec_(
        &self,
        ix: &[(&str, &DataArray)],
        method: annoray::Method,
        tolerance: Option<&annoray::Label>,
    ) -> annoray::Result<Self> {
        self.sel_vectorized(ix, method, tolerance)
    }
    fn isel_(&self, ix: Vec<(String, annoray::Indexer)>) -> annoray::Result<Self> {
        Ok(self.isel(ix)?.to_owned())
    }
    fn sel_(&self, ix: Vec<(String, LabelIndexer)>) -> annoray::Result<Self> {
        Ok(self.sel(ix)?.to_owned())
    }
    fn sel_take_(&self, dim: &str, labels: Vec<LabelIndexer>) -> annoray::Result<Self> {
        self.sel_take(dim, labels)
    }
}

/// An indexer that selects pointwise: an annoray `DataArray` (an xarray one is converted).
fn vector_indexer(v: &Bound<'_, PyAny>) -> PyResult<Option<DataArray>> {
    if let Ok(da) = v.cast::<PyDataArray>() {
        return Ok(Some(da.borrow().inner.clone()));
    }
    let module = v.get_type().getattr("__module__")?;
    if module.extract::<String>()?.starts_with("xarray")
        && v.hasattr("dims")?
        && v.hasattr("variable")?
    {
        let converted = v
            .py()
            .import("annoray")?
            .call_method1("from_xarray", (v,))?;
        return Ok(Some(
            converted.cast::<PyDataArray>()?.borrow().inner.clone(),
        ));
    }
    Ok(None)
}

/// Positional selection: integer lists and masks first (copying those rows), then indices
/// and slices.
fn do_isel<T: Select + Clone>(py: Python<'_>, obj: &T, args: IselArgs<'_>) -> PyResult<T> {
    let mut takes = Vec::new();
    let mut basic = Vec::new();
    let mut vectors: Vec<(String, DataArray)> = Vec::new();
    for (dim, v) in args {
        if let Some(da) = vector_indexer(&v)? {
            vectors.push((dim, da));
            continue;
        }
        match pos_from_py(&v)? {
            PosSel::Basic(ix) => basic.push((dim, ix)),
            PosSel::Take(idx) => takes.push((dim, idx)),
        }
    }
    py.detach(|| {
        let mut taken: Option<T> = None;
        for (dim, idx) in &takes {
            let next = taken.as_ref().unwrap_or(obj).take_(dim, idx)?;
            taken = Some(next);
        }
        let out = taken.as_ref().unwrap_or(obj).isel_(basic)?;
        if vectors.is_empty() {
            return Ok(out);
        }
        let refs: Vec<(&str, &DataArray)> = vectors.iter().map(|(d, a)| (d.as_str(), a)).collect();
        out.isel_vec_(&refs)
    })
    .py_err()
}

fn do_sel<T: Select + Clone>(
    py: Python<'_>,
    obj: &T,
    args: IselArgs<'_>,
    method: Option<&str>,
    tolerance: Option<&Bound<'_, PyAny>>,
) -> PyResult<T> {
    let method = method_from_py(method)?;
    let tolerance = tolerance.map(label_from_py).transpose()?;
    let mut many = Vec::new();
    let mut one = Vec::new();
    let mut vectors: Vec<(String, DataArray)> = Vec::new();
    for (dim, v) in args {
        if let Some(da) = vector_indexer(&v)? {
            vectors.push((dim, da));
            continue;
        }
        match label_sel_from_py(&v, method, tolerance.as_ref())? {
            LabelSel::One(ix) => one.push((dim, *ix)),
            LabelSel::Many(ix) => many.push((dim, ix)),
        }
    }
    py.detach(|| {
        let mut taken: Option<T> = None;
        for (dim, labels) in many {
            let next = taken.as_ref().unwrap_or(obj).sel_take_(&dim, labels)?;
            taken = Some(next);
        }
        let out = taken.as_ref().unwrap_or(obj).sel_(one)?;
        if vectors.is_empty() {
            return Ok(out);
        }
        let refs: Vec<(&str, &DataArray)> = vectors.iter().map(|(d, a)| (d.as_str(), a)).collect();
        out.sel_vec_(&refs, method, tolerance.as_ref())
    })
    .py_err()
}

/// `args` without the names that are not dimensions: an error (`missing_dims="raise"`), a
/// warning (`"warn"`) or silently (`"ignore"`).
fn known_dims<'py>(
    py: Python<'py>,
    dims: &[Name],
    args: IselArgs<'py>,
    missing_dims: &str,
) -> PyResult<IselArgs<'py>> {
    let (known, unknown): (Vec<_>, Vec<_>) = args
        .into_iter()
        .partition(|(d, _)| dims.iter().any(|x| x.as_str() == d.as_str()));
    if let Some((d, _)) = unknown.first() {
        match missing_dims {
            "raise" => {
                return Err(PyValueError::new_err(format!(
                    "{d:?} is not a dimension; dimensions: {dims:?}"
                )));
            }
            "warn" => {
                let warnings = py.import("warnings")?;
                warnings.call_method1("warn", (format!("{d:?} is not a dimension; ignored"),))?;
            }
            "ignore" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "missing_dims must be \"raise\", \"warn\" or \"ignore\", not {other:?}"
                )));
            }
        }
    }
    Ok(known)
}

/// The coordinates of `before` that indexing made 0-d.
fn scalarized(
    before: &annoray::Coords<annoray::Owned>,
    after: &annoray::Coords<annoray::Owned>,
) -> Vec<String> {
    after
        .iter()
        .filter(|(k, c)| c.ndim() == 0 && before.get(*k).is_some_and(|b| b.ndim() > 0))
        .map(|(k, _)| k.to_string())
        .collect()
}

fn drop_scalarized_da(before: &DataArray, after: DataArray) -> PyResult<DataArray> {
    let gone = scalarized(before.coords(), after.coords());
    let refs: Vec<&str> = gone.iter().map(String::as_str).collect();
    after.drop_coords(&refs).py_err()
}

fn drop_scalarized_ds(before: &Dataset, mut after: Dataset) -> PyResult<Dataset> {
    for k in scalarized(before.coords(), after.coords()) {
        after.remove(&k).py_err()?;
    }
    Ok(after)
}

// ----- reductions -----

pub(crate) fn reduce_dims(dim: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Name>>> {
    match dim {
        None => Ok(None),
        Some(d) if d.is_none() => Ok(None),
        Some(d) if d.is_instance_of::<PyString>() => Ok(Some(vec![d.extract::<String>()?.into()])),
        Some(d) => Ok(Some(
            d.extract::<Vec<String>>()?
                .into_iter()
                .map(Into::into)
                .collect(),
        )),
    }
}

fn reduce_opts(
    skipna: Option<bool>,
    ddof: usize,
    min_count: Option<usize>,
    keep_attrs: bool,
) -> ReduceOptions {
    let mut o = ReduceOptions::default().ddof(ddof).keep_attrs(keep_attrs);
    if let Some(s) = skipna {
        o = o.skipna(s);
    }
    if let Some(m) = min_count {
        o = o.min_count(m);
    }
    o
}

/// numpy's `np.sum(da, …)` passes `out=None` (and `dtype=None` for some functions); other
/// values are not supported.
fn numpy_only_defaults(
    out: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if out.is_some_and(|o| !o.is_none()) {
        return Err(PyTypeError::new_err(
            "out= is not supported (results are new arrays)",
        ));
    }
    if dtype.is_some_and(|d| !d.is_none()) {
        return Err(PyTypeError::new_err(
            "dtype= is not supported; use .astype() on the result",
        ));
    }
    Ok(())
}

/// Dimension names of numpy axes (`0`, `-1`, `(0, 1)`) of an array with `dims`.
pub(crate) fn axis_names(dims: &[Name], axis: &Bound<'_, PyAny>) -> PyResult<Vec<Name>> {
    let axes: Vec<isize> = match axis.extract::<isize>() {
        Ok(a) => vec![a],
        Err(_) => axis.extract()?,
    };
    let n = isize::try_from(dims.len()).unwrap_or(isize::MAX);
    axes.into_iter()
        .map(|a| {
            let i = if a < 0 { a.saturating_add(n) } else { a };
            usize::try_from(i)
                .ok()
                .and_then(|i| dims.get(i))
                .cloned()
                .ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err(format!(
                        "axis {a} is out of bounds for an array of {n} dimensions"
                    ))
                })
        })
        .collect()
}

/// The reduction methods (`sum`, `mean`, …) of a class with `reduce_with`.
macro_rules! reductions {
    ($ty:ident) => {
        reductions!(@impl $ty;
            sum Sum, prod Prod, mean Mean, median Median, std Std, var Var, min Min, max Max,
            count Count, argmin ArgMin, argmax ArgMax, any Any, all All);
    };
    (@impl $ty:ident; $($m:ident $op:ident),*) => {
        #[pymethods]
        impl $ty {
            $(
                #[doc = concat!("`", stringify!($m), "` over `dim` (all dimensions by default).")]
                #[allow(clippy::too_many_arguments)]
                #[pyo3(signature = (
                    dim=None, *, axis=None, skipna=None, ddof=0, min_count=None,
                    keep_attrs=None, out=None, dtype=None
                ))]
                fn $m(
                    &self,
                    py: Python<'_>,
                    dim: Option<&Bound<'_, PyAny>>,
                    axis: Option<&Bound<'_, PyAny>>,
                    skipna: Option<bool>,
                    ddof: usize,
                    min_count: Option<usize>,
                    keep_attrs: Option<bool>,
                    out: Option<&Bound<'_, PyAny>>,
                    dtype: Option<&Bound<'_, PyAny>>,
                ) -> PyResult<Py<PyAny>> {
                    numpy_only_defaults(out, dtype)?;
                    let opts = reduce_opts(skipna, ddof, min_count, keep_attrs.unwrap_or(true));
                    let dims = match axis {
                        Some(a) if !a.is_none() => {
                            if dim.is_some_and(|d| !d.is_none()) {
                                return Err(PyValueError::new_err(
                                    "give dim or axis, not both",
                                ));
                            }
                            Some(self.dims_of_axis(a)?)
                        }
                        _ => reduce_dims(dim)?,
                    };
                    self.reduce_with(py, Reduction::$op, dims, &opts)
                }
            )*
        }
    };
}

/// `+ - * /` with units checked (and the reflected forms), for a class with `binary_with`.
macro_rules! arithmetic {
    ($ty:ident) => {
        #[pymethods]
        impl $ty {
            fn __add__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                self.binary_with(py, BinOp::Add, o, false)
            }
            fn __sub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                self.binary_with(py, BinOp::Sub, o, false)
            }
            fn __mul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                self.binary_with(py, BinOp::Mul, o, false)
            }
            fn __truediv__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                self.binary_with(py, BinOp::Div, o, false)
            }
            fn __radd__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                self.binary_with(py, BinOp::Add, o, true)
            }
            fn __rsub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                self.binary_with(py, BinOp::Sub, o, true)
            }
            fn __rmul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                self.binary_with(py, BinOp::Mul, o, true)
            }
            fn __rtruediv__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                self.binary_with(py, BinOp::Div, o, true)
            }
        }
    };
}

/// The other operand of arithmetic: a data array, or a scalar (possibly with units) as a
/// 0-d data array.
pub(crate) enum Operand<'py> {
    Array(PyRef<'py, PyDataArray>),
    Owned(Box<DataArray>),
}

impl Operand<'_> {
    pub(crate) fn get(&self) -> &DataArray {
        match self {
            Self::Array(a) => &a.inner,
            Self::Owned(a) => a,
        }
    }
}

pub(crate) fn operand<'py>(
    py: Python<'py>,
    o: &Bound<'py, PyAny>,
) -> PyResult<Option<Operand<'py>>> {
    operand_like(py, o, None)
}

/// Whether `o` is a plain Python `float` or `int` (not a numpy scalar, not a bool): numpy's
/// "weak" scalars, which take the other operand's type.
fn is_weak_scalar(o: &Bound<'_, PyAny>) -> bool {
    let t = o.get_type();
    t.is(o.py().get_type::<pyo3::types::PyFloat>()) || t.is(o.py().get_type::<pyo3::types::PyInt>())
}

/// [`operand`], with a plain Python number (or a quantity's) taken as `f32` beside `f32`
/// data (`like`), as numpy's weak scalars: `f32 * 2.0` stays `f32`.
pub(crate) fn operand_like<'py>(
    py: Python<'py>,
    o: &Bound<'py, PyAny>,
    like: Option<annoray::DType>,
) -> PyResult<Option<Operand<'py>>> {
    let weak = like == Some(annoray::DType::F32)
        && (is_weak_scalar(o)
            || o.is_instance_of::<crate::units::PyQuantity>()
            || quantity_parts(o)?.is_some_and(|(m, _)| is_weak_scalar(&m)));
    let out = operand_strong(py, o)?;
    match out {
        Some(Operand::Owned(a)) if weak && a.dtype() == annoray::DType::F64 => Ok(Some(
            Operand::Owned(Box::new(a.astype(annoray::DType::F32).py_err()?)),
        )),
        other => Ok(other),
    }
}

fn operand_strong<'py>(py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Option<Operand<'py>>> {
    if let Ok(da) = o.cast::<PyDataArray>() {
        return Ok(Some(Operand::Array(da.borrow())));
    }
    // Scalars only: arrays would broadcast by position, which labelled data never does.
    let (mag, unit) = match quantity_parts(o)? {
        Some((m, u)) => (m, Some(u)),
        None => (o.clone(), None),
    };
    let np = py.import("numpy")?;
    if np.call_method1("ndim", (&mag,))?.extract::<usize>()? != 0 {
        return Ok(None);
    }
    let units = unit.as_deref();
    let var = variable_from_py(py, Some(&PyTuple::empty(py).into_any()), &mag, None, units)?;
    Ok(Some(Operand::Owned(Box::new(DataArray::new(var)))))
}

// ----- DataArray -----

/// A labelled, units-aware N-dimensional array (xarray's `DataArray`).
#[pyclass(name = "DataArray", module = "annoray", weakref)]
pub(crate) struct PyDataArray {
    pub(crate) inner: DataArray,
    pub(crate) lending: Lending,
    pub(crate) parent: Option<Source>,
}

impl PyDataArray {
    fn dims_of_axis(&self, axis: &Bound<'_, PyAny>) -> PyResult<Vec<Name>> {
        axis_names(self.inner.dims(), axis)
    }

    fn reduce_with(
        &self,
        py: Python<'_>,
        op: Reduction,
        dims: Option<Vec<Name>>,
        opts: &ReduceOptions,
    ) -> PyResult<Py<PyAny>> {
        let out = py
            .detach(|| match dims {
                None => self.inner.reduce(op, .., opts),
                Some(d) => self.inner.reduce(op, d, opts),
            })
            .py_err()?;
        Ok(new_da(py, out)?.into_any())
    }

    pub(crate) fn binary_with(
        &self,
        py: Python<'_>,
        op: BinOp,
        o: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        let Some(other) = operand_like(py, o, Some(self.inner.dtype()))? else {
            return Ok(py.NotImplemented());
        };
        let other = other.get();
        let out = py
            .detach(|| {
                if reflected {
                    other.binary(op, &self.inner, UnitPolicy::Strict)
                } else {
                    self.inner.binary(op, other, UnitPolicy::Strict)
                }
            })
            .py_err()?;
        // A scalar operand keeps the array's name and dimension order.
        let out = if reflected && other.ndim() == 0 {
            match self.inner.name() {
                Some(n) => out.with_name(n.clone()),
                None => out,
            }
        } else {
            out
        };
        Ok(new_da(py, out)?.into_any())
    }

    pub(crate) fn coord_array(&self, name: &str) -> PyResult<DataArray> {
        let c = self.inner.coord(name).py_err()?;
        let mut out = DataArray::new(c.to_owned()).with_name(name);
        for (k, v) in self.inner.coords() {
            if k.as_str() != name && v.dims().iter().all(|d| c.dims().contains(d)) {
                out = out.with_coord(k.clone(), v.to_owned()).py_err()?;
            }
        }
        Ok(out)
    }
}

reductions!(PyDataArray);
arithmetic!(PyDataArray);

#[pymethods]
impl PyDataArray {
    /// `DataArray[Length]`: the dimension, for annotations (checked by `annoray.dims.checked`).
    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, pyo3::types::PyType>,
        item: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cls.py()
            .import("types")?
            .getattr("GenericAlias")?
            .call1((cls, item))
    }

    /// `DataArray(data, dims=None, coords=None, name=None, attrs=None, units=None)`.
    ///
    /// `data` may be a numpy array (any numeric, bool, datetime64, timedelta64, string or
    /// cftime dtype), anything numpy accepts, or a pint-like quantity. The unit comes from
    /// `units`, the quantity, or `attrs["units"]`.
    #[new]
    #[pyo3(signature = (data, dims=None, coords=None, name=None, attrs=None, units=None))]
    fn py_new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        dims: Option<&Bound<'_, PyAny>>,
        coords: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
        attrs: Option<&Bound<'_, PyDict>>,
        units: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let units = units
            .map(|u| crate::units::unit_from_py(u).map(|u| u.to_string()))
            .transpose()?;
        let var = variable_from_py(py, dims, data, attrs, units.as_deref())?;
        let mut inner = DataArray::new(var);
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        for (k, v) in dict_iter(coords)? {
            let c = spec_from_py(py, &k, &v)?;
            inner = inner.with_coord(k, c).py_err()?;
        }
        Ok(Self {
            inner,
            lending: Lending::default(),
            parent: None,
        })
    }

    /// The array's name (`None` if unnamed); assignable.
    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name().map(ToString::to_string)
    }

    #[setter]
    fn set_name(&mut self, name: Option<String>) {
        self.inner.set_name(name.map(Name::from));
    }

    /// The dimension of the data's unit (`None` for unitless data).
    #[getter]
    fn dimension(&self) -> Option<crate::units::PyDimension> {
        self.inner
            .unit()
            .map(|u| crate::units::PyDimension { inner: u.dim() })
    }

    /// The unit as an `annoray.Unit` (`None` for unitless data); `units` gives it as text.
    #[getter]
    fn unit(&self) -> Option<crate::units::PyUnit> {
        self.inner
            .unit()
            .map(|u| crate::units::PyUnit { inner: u.clone() })
    }

    /// Dimension names.
    #[getter]
    fn dims<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.dims().iter().map(Name::as_str))
    }

    /// The length of each dimension, in order.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape())
    }

    /// Dimension name → length.
    #[getter]
    fn sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        sizes_dict(py, self.inner.sizes().map(|(k, n)| (k.clone(), n)))
    }

    /// The number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// The number of elements.
    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape()
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of unsized object"))
    }

    /// The numpy dtype of `.values`.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np_dtype = py.import("numpy")?.getattr("dtype")?;
        if self.inner.dtype() == annoray::DType::Str {
            // Fixed-width text, as `.values` gives (lazy text is read for its width).
            let v = self.inner.variable();
            let width = match v.typed() {
                Ok(TypedArray::Str(a)) => crate::convert::text_width(&a.view()),
                _ => match v.to_owned().into_typed().py_err()? {
                    TypedArray::Str(a) => crate::convert::text_width(&a.view()),
                    _ => 1,
                },
            };
            return np_dtype.call1((format!("U{width}"),));
        }
        np_dtype.call1((numpy_dtype_name(self.inner.dtype()),))
    }

    /// The unit, or `None` for unitless data.
    #[getter]
    fn units(&self) -> Option<String> {
        unit_str(self.inner.variable())
    }

    /// The attributes, as a dict whose changes are applied to this array (and, for an
    /// array taken from a dataset, to that dataset's variable).
    #[getter]
    fn attrs<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let d = attrs_to_py(slf.py(), slf.borrow().inner.attrs())?;
        live_dict(slf.as_any(), "AttrsDict", d.into_any())
    }

    #[setter(attrs)]
    fn set_attrs_py(slf: &Bound<'_, Self>, attrs: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut a = annoray::Attrs::new();
        set_attrs(&mut a, Some(attrs))?;
        Self::replace_attrs(slf, a)
    }

    /// Coordinates, as data arrays, in a dict whose assignments (`da.coords["x"] = …`) and
    /// deletions apply to this array.
    #[getter]
    fn coords<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        live_dict(
            slf.as_any(),
            "CoordsDict",
            Self::plain_coords(slf)?.into_any(),
        )
    }

    /// The values as numpy: a read-only view (no copy) for numbers in memory.
    #[getter]
    fn values<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        values_of(slf.clone().into_any(), slf.borrow().inner.variable())
    }

    /// Same as `.values`.
    #[getter]
    fn data<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Self::values(slf)
    }

    /// Same as `.values`.
    fn to_numpy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Self::values(slf)
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        slf: &Bound<'py, Self>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = Self::values(slf)?;
        let np = slf.py().import("numpy")?;
        let kw = PyDict::new(slf.py());
        kw.set_item("dtype", dtype)?;
        kw.set_item("copy", copy)?;
        np.getattr("array")?.call((v,), Some(&kw))
    }

    /// Whether the values are still on disk (read on demand).
    #[getter]
    fn is_lazy(&self) -> bool {
        self.inner.variable().is_lazy()
    }

    /// A copy with the values read into memory.
    fn load(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        let (name, mut var, coords) = self.inner.to_owned().into_parts();
        py.detach(|| var.load()).py_err()?;
        let mut out = DataArray::new(var);
        if let Some(n) = name {
            out = out.with_name(n);
        }
        for (k, c) in coords {
            out = out.with_coord(k, c).py_err()?;
        }
        new_da(py, out)
    }

    /// Same as `load`.
    fn compute(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        self.load(py)
    }

    /// `put` with positions already resolved: `at` pairs a dimension with one position (an
    /// int; the dimension leaves the selection) or a list of positions. `value` is a data
    /// array, a scalar or quantity, or an array (or quantity of one) shaped like the
    /// selection, whose dimensions it then takes in order.
    fn _put(
        &self,
        py: Python<'_>,
        at: Vec<(String, Bound<'_, PyAny>)>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyDataArray>> {
        let mut pos: Vec<(Name, annoray::Positions)> = Vec::with_capacity(at.len());
        for (d, p) in at {
            let p = match p.extract::<usize>() {
                Ok(i) => annoray::Positions::At(i),
                Err(_) => annoray::Positions::Each(p.extract::<Vec<usize>>()?),
            };
            pos.push((Name::from(d.as_str()), p));
        }
        let held;
        let owned;
        let value: &DataArray = if let Ok(da) = value.cast::<PyDataArray>() {
            held = da.borrow();
            &held.inner
        } else {
            let (mag, unit) = match quantity_parts(value)? {
                Some((m, u)) => (m, Some(u)),
                None => (value.clone(), None),
            };
            let np = py.import("numpy")?;
            let ndim = np.call_method1("ndim", (&mag,))?.extract::<usize>()?;
            let dims: Vec<String> = if ndim == 0 {
                Vec::new()
            } else {
                // An unlabelled array takes the selection's dimensions, in order.
                self.inner
                    .dims()
                    .iter()
                    .filter(|d| {
                        !pos.iter()
                            .any(|(n, p)| n == *d && matches!(p, annoray::Positions::At(_)))
                    })
                    .map(ToString::to_string)
                    .collect()
            };
            if ndim != 0 && ndim != dims.len() {
                return Err(PyValueError::new_err(format!(
                    "an unlabelled value must have the selection's shape (dimensions {dims:?}); \
                     got {ndim} dimension(s): give a DataArray to broadcast by name"
                )));
            }
            let dims = PyTuple::new(py, dims)?;
            let var = variable_from_py(py, Some(dims.as_any()), &mag, None, unit.as_deref())?;
            owned = DataArray::new(var);
            &owned
        };
        let out = py.detach(|| self.inner.put(&pos, value)).py_err()?;
        new_da(py, out)
    }

    /// The data converted to another unit (as float64).
    fn to(&self, py: Python<'_>, units: &Bound<'_, PyAny>) -> PyResult<Py<PyDataArray>> {
        let u = crate::units::unit_from_py(units)?;
        new_da(py, py.detach(|| self.inner.to_unit(&u)).py_err()?)
    }

    /// Reorder dimensions: `da.transpose("x", ...)`; no names reverses them.
    #[pyo3(signature = (*dims))]
    fn transpose(&self, py: Python<'_>, dims: Vec<Bound<'_, PyAny>>) -> PyResult<Py<PyDataArray>> {
        let order = transpose_order(py, &dims)?;
        let refs: Vec<&str> = order.iter().map(String::as_str).collect();
        new_da(py, py.detach(|| self.inner.transpose(&refs)).py_err()?)
    }

    /// The transpose (dimensions reversed).
    #[getter]
    #[allow(non_snake_case)]
    fn T(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.transpose(&[])).py_err()?)
    }

    /// A copy with another name (`rename("v")`), or with dimensions and coordinates renamed
    /// (`rename({"x": "lon"})`, `rename(x="lon")`).
    #[pyo3(signature = (new_name_or_name_dict=None, **names))]
    fn rename(
        &self,
        py: Python<'_>,
        new_name_or_name_dict: Option<&Bound<'_, PyAny>>,
        names: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        if let Some(n) = new_name_or_name_dict.filter(|n| n.is_instance_of::<PyString>()) {
            return new_da(py, self.inner.clone().with_name(n.extract::<String>()?));
        }
        // `rename()` / `rename(None)` clear the name, as in xarray.
        if new_name_or_name_dict.is_none_or(|n| n.is_none()) && names.is_none_or(|n| n.is_empty()) {
            let mut out = self.inner.clone();
            out.set_name(None);
            return new_da(py, out);
        }
        let mut pairs: Vec<(String, String)> = Vec::new();
        for d in [
            new_name_or_name_dict.filter(|d| !d.is_none()),
            names.map(|n| n.as_any()),
        ]
        .into_iter()
        .flatten()
        {
            let d = d.cast::<PyDict>()?;
            for (k, v) in d.iter() {
                pairs.push((k.extract()?, v.extract()?));
            }
        }
        let refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        new_da(py, py.detach(|| self.inner.rename_dims(&refs)).py_err()?)
    }

    /// Select by position: `da.isel(time=0, x=slice(1, 3))`, lists and masks too.
    /// `drop=True` removes the coordinates a scalar index makes 0-d; `missing_dims`
    /// (`"raise"`, `"warn"`, `"ignore"`) treats names that are not dimensions.
    #[pyo3(signature = (indexers=None, drop=false, missing_dims="raise", **kwargs))]
    fn isel(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        drop: bool,
        missing_dims: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let args = known_dims(
            py,
            self.inner.dims(),
            merge_kwargs(indexers, kwargs)?,
            missing_dims,
        )?;
        let out = do_isel(py, &self.inner, args)?;
        new_da(
            py,
            if drop {
                drop_scalarized_da(&self.inner, out)?
            } else {
                out
            },
        )
    }

    /// Select by label: `da.sel(time="2020-01", x=slice(0, 10))`; slices are inclusive.
    /// `method` is `"nearest"`, `"pad"`/`"ffill"` or `"backfill"`/`"bfill"`; `drop=True`
    /// removes the coordinates a scalar label makes 0-d.
    #[pyo3(signature = (indexers=None, method=None, tolerance=None, drop=false, **kwargs))]
    fn sel(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        method: Option<&str>,
        tolerance: Option<&Bound<'_, PyAny>>,
        drop: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let args = merge_kwargs(indexers, kwargs)?;
        let out = do_sel(py, &self.inner, args, method, tolerance)?;
        new_da(
            py,
            if drop {
                drop_scalarized_da(&self.inner, out)?
            } else {
                out
            },
        )
    }

    /// Coordinates as attributes: `da.time`; registered accessors first.
    fn __getattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(acc) = accessor(slf.as_any(), "DataArray", name)? {
            return Ok(acc);
        }
        let coord = {
            let me = slf.borrow();
            if me.inner.coords().contains_key(name) {
                Some(me.coord_array(name)?)
            } else {
                None
            }
        };
        if let Some(c) = coord {
            return Ok(new_child_da(slf.as_any(), name, c)?.into_any());
        }
        Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "'DataArray' object has no attribute {name:?}"
        )))
    }

    /// `da[0, 1:3]` by position, `da[{"x": 0}]` by name, `da["x"]` for a coordinate.
    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyDataArray>> {
        let py = slf.py();
        if key.is_instance_of::<PyString>() {
            let name: String = key.extract()?;
            let c = slf.borrow().coord_array(&name)?;
            return new_child_da(slf.as_any(), &name, c);
        }
        let me = slf.borrow();
        let this = &*me;
        if key.is_instance_of::<PyDict>() {
            return this.isel(py, Some(key), false, "raise", None);
        }
        let items: Vec<Bound<'_, PyAny>> = match key.cast::<PyTuple>() {
            Ok(t) => t.iter().collect(),
            Err(_) => vec![key.clone()],
        };
        if items.iter().any(|k| k.is(py.Ellipsis())) {
            let ix = items
                .iter()
                .map(|k| match pos_from_py(k)? {
                    PosSel::Basic(i) => Ok(i),
                    PosSel::Take(_) => Err(PyTypeError::new_err(
                        "lists and masks cannot be combined with ... ; use isel",
                    )),
                })
                .collect::<PyResult<Vec<_>>>()?;
            let out = this.inner.isel_pos(ix).py_err()?.to_owned();
            return new_da(py, out);
        }
        if items.len() > this.inner.ndim() {
            return Err(pyo3::exceptions::PyIndexError::new_err("too many indices"));
        }
        let args = this
            .inner
            .dims()
            .iter()
            .zip(items)
            .map(|(d, k)| (d.to_string(), k))
            .collect();
        new_da(py, do_isel(py, &this.inner, args)?)
    }

    /// Whether each value is missing (NaN, NaT): a boolean array, unit dropped.
    fn isnull(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        new_da(py, self.inner.isnull().py_err()?)
    }

    /// Whether each value is present (not NaN or NaT): a boolean array, unit dropped.
    fn notnull(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        new_da(py, self.inner.notnull().py_err()?)
    }

    /// Missing values (NaN) replaced by the number `value` (`fillna` checks units first).
    fn _fillna_number(&self, py: Python<'_>, value: f64) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.fillna(value)).py_err()?)
    }

    /// Keep values where `cond` (booleans, or a callable taking this array) is true;
    /// elsewhere NaN, or `other`: a data array, a quantity (converted into this array's
    /// unit), or a number (only NaN, or 0 in a ratio-scale unit, beside data with a unit).
    #[pyo3(signature = (cond, other=None))]
    fn r#where(
        slf: &Bound<'_, Self>,
        cond: &Bound<'_, PyAny>,
        other: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataArray>> {
        crate::utility::where_method(slf, cond, other)
    }

    /// Drop positions along `dim` with missing values (`how="any"` or `"all"`).
    #[pyo3(signature = (dim, how="any"))]
    fn dropna(&self, py: Python<'_>, dim: &str, how: &str) -> PyResult<Py<PyDataArray>> {
        new_da(py, self.inner.dropna(dim, how_from(how)?).py_err()?)
    }

    /// Conform to new labels along dimensions, filling missing ones with NaN (NaT);
    /// `method` (`"nearest"`, `"pad"`, `"backfill"`), `tolerance` and `fill_value` as xarray.
    #[pyo3(signature = (indexers=None, method=None, tolerance=None, fill_value=None, **kwargs))]
    fn reindex(
        slf: &Bound<'_, Self>,
        indexers: Option<&Bound<'_, PyAny>>,
        method: Option<&Bound<'_, PyAny>>,
        tolerance: Option<&Bound<'_, PyAny>>,
        fill_value: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if method.is_some() || tolerance.is_some() || fill_value.is_some() {
            let kw = PyDict::new(py);
            kw.set_item("indexers", indexers)?;
            kw.set_item("method", method)?;
            kw.set_item("tolerance", tolerance)?;
            kw.set_item("fill_value", fill_value)?;
            if let Some(k) = kwargs {
                kw.update(k.as_mapping())?;
            }
            return Ok(py
                .import("annoray._methods")?
                .getattr("reindex_method")?
                .call((slf,), Some(&kw))?
                .unbind());
        }
        let mut out = slf.borrow().inner.clone();
        for (dim, v) in merge_kwargs(indexers, kwargs)? {
            let labels = labels_var(py, &dim, &v)?;
            out = out.reindex(&dim, &labels).py_err()?;
        }
        Ok(new_da(py, out)?.into_any())
    }

    /// Interpolate along dimensions: `da.interp(x=[0.5, 1.5], method="linear")`.
    /// `extrapolate` is `None` (NaN outside the coordinates), `"edge"`, `"extend"`, a number
    /// `k` (a power law `y_edge * (x / x_edge)**k`), or a `(below, above)` pair of those.
    #[pyo3(signature = (coords=None, method="linear", extrapolate=None, **kwargs))]
    fn interp(
        &self,
        py: Python<'_>,
        coords: Option<&Bound<'_, PyAny>>,
        method: &str,
        extrapolate: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let m = interp_options(method, extrapolate)?;
        let mut out = self.inner.clone();
        for (dim, v) in merge_kwargs(coords, kwargs)? {
            let at = labels_var(py, &dim, &v)?;
            out = py.detach(|| out.interp(&dim, &at, m)).py_err()?;
        }
        new_da(py, out)
    }

    /// Group by a coordinate (or `"time.month"`, …).
    fn groupby(slf: &Bound<'_, Self>, group: String) -> PyResult<PyGroupBy> {
        slf.borrow().inner.groupby(&group).py_err()?;
        Ok(PyGroupBy {
            parent: Parent::Da(slf.clone().unbind()),
            split: Split::Labels(group),
        })
    }

    /// Group by bins of a numeric coordinate.
    #[pyo3(signature = (group, bins, right=true, include_lowest=false))]
    fn groupby_bins(
        slf: &Bound<'_, Self>,
        group: String,
        bins: Vec<f64>,
        right: bool,
        include_lowest: bool,
    ) -> PyResult<PyGroupBy> {
        let opts = BinOptions::default()
            .right(right)
            .include_lowest(include_lowest);
        slf.borrow()
            .inner
            .groupby_bins(&group, &bins, &opts)
            .py_err()?;
        Ok(PyGroupBy {
            parent: Parent::Da(slf.clone().unbind()),
            split: Split::Bins { group, bins, opts },
        })
    }

    /// Resample a time dimension: `da.resample(time="1D").mean()`.
    #[pyo3(signature = (indexer=None, **kwargs))]
    fn resample(
        slf: &Bound<'_, Self>,
        indexer: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyGroupBy> {
        let (dim, freq) = one_pair(merge_kwargs(indexer, kwargs)?)?;
        let freq: String = freq.extract()?;
        slf.borrow().inner.resample(&dim, &freq).py_err()?;
        Ok(PyGroupBy {
            parent: Parent::Da(slf.clone().unbind()),
            split: Split::Resample { dim, freq },
        })
    }

    /// A moving window: `da.rolling(time=3, center=True).mean()`.
    #[pyo3(signature = (dim=None, center=false, min_periods=None, **kwargs))]
    fn rolling(
        slf: &Bound<'_, Self>,
        dim: Option<&Bound<'_, PyAny>>,
        center: bool,
        min_periods: Option<usize>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let pairs = merge_kwargs(dim, kwargs)?;
        if pairs.len() > 1 {
            return multi_rolling(slf.as_any(), pairs, center, min_periods);
        }
        let (dim, window) = one_pair(pairs)?;
        let window: usize = window.extract()?;
        let opts = rolling_opts(center, min_periods);
        slf.borrow().inner.rolling(&dim, window, &opts).py_err()?;
        let r = PyRolling {
            parent: Parent::Da(slf.clone().unbind()),
            dim,
            window,
            opts,
        };
        Ok(Py::new(slf.py(), r)?.into_any())
    }

    /// The values as an `astropy.units.Quantity` in this array's unit (coordinates and
    /// attributes are dropped; needs astropy).
    fn to_astropy<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::units::to_astropy(slf.as_any())
    }

    /// Plotting with matplotlib: `da.plot()` by rank, or `da.plot.pcolormesh(...)` etc.
    /// (`annoray.plot`).
    #[getter]
    fn plot<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py()
            .import("annoray.plot")?
            .getattr("DataArrayPlotAccessor")?
            .call1((slf,))
    }

    /// Date and time components: `da.dt.year`, `da.dt.dayofyear`, ….
    #[getter]
    fn dt(slf: &Bound<'_, Self>) -> PyDt {
        PyDt {
            parent: slf.clone().unbind(),
        }
    }

    /// A one-variable dataset (`to_dataset` also splits along a dimension).
    #[pyo3(signature = (name=None))]
    fn _to_dataset_named(&self, py: Python<'_>, name: Option<String>) -> PyResult<Py<PyDataset>> {
        let name = name
            .or_else(|| self.inner.name().map(ToString::to_string))
            .ok_or_else(|| PyValueError::new_err("unnamed DataArray: pass name="))?;
        let mut ds = Dataset::new();
        ds.add_array(name, self.inner.clone()).py_err()?;
        new_ds(py, ds)
    }

    /// As an `xarray.DataArray` (units as the `"units"` attribute).
    fn to_xarray<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        let xr = py.import("xarray")?;
        let coords = PyDict::new(py);
        for (k, c) in this.inner.coords() {
            coords.set_item(k.as_str(), xr_variable(py, &xr, slf.clone().into_any(), c)?)?;
        }
        let kw = PyDict::new(py);
        kw.set_item("coords", coords)?;
        kw.set_item("name", this.name())?;
        let var = xr_variable(py, &xr, slf.clone().into_any(), this.inner.variable())?;
        xr.getattr("DataArray")?.call((var,), Some(&kw))
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

// ----- Dataset -----

/// A units-aware collection of labelled arrays sharing dimensions (xarray's `Dataset`).
#[pyclass(name = "Dataset", module = "annoray", weakref)]
pub(crate) struct PyDataset {
    pub(crate) inner: Dataset,
    pub(crate) lending: Lending,
}

impl PyDataset {
    #[allow(clippy::unused_self)]
    fn dims_of_axis(&self, _axis: &Bound<'_, PyAny>) -> PyResult<Vec<Name>> {
        Err(PyTypeError::new_err(
            "axis= is ambiguous for a Dataset (its variables order dimensions differently); \
             use dim=",
        ))
    }

    fn reduce_with(
        &self,
        py: Python<'_>,
        op: Reduction,
        dims: Option<Vec<Name>>,
        opts: &ReduceOptions,
    ) -> PyResult<Py<PyAny>> {
        let out = py
            .detach(|| match dims {
                None => self.inner.reduce(op, .., opts),
                Some(d) => self.inner.reduce(op, d, opts),
            })
            .py_err()?;
        Ok(new_ds(py, out)?.into_any())
    }

    pub(crate) fn array(&self, name: &str) -> PyResult<DataArray> {
        if !self.inner.contains(name) {
            return Err(PyKeyError::new_err(name.to_owned()));
        }
        Ok(self.inner.get(name).py_err()?.to_owned())
    }
}

reductions!(PyDataset);

#[pymethods]
impl PyDataset {
    /// `Dataset(data_vars=None, coords=None, attrs=None)`; variables as `(dims, data[,
    /// attrs])` tuples or `DataArray`s.
    #[new]
    #[pyo3(signature = (data_vars=None, coords=None, attrs=None))]
    fn py_new(
        py: Python<'_>,
        data_vars: Option<&Bound<'_, PyAny>>,
        coords: Option<&Bound<'_, PyAny>>,
        attrs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut inner = Dataset::new();
        for (k, v) in dict_iter(coords)? {
            inner
                .add_coord(k.as_str(), spec_from_py(py, &k, &v)?)
                .py_err()?;
        }
        for (k, v) in dict_iter(data_vars)? {
            match v.cast::<PyDataArray>() {
                Ok(da) => {
                    let mut a = da.borrow().inner.clone();
                    // Coordinates already given win (as in xarray with compat="override"-like
                    // identical coordinates).
                    a = a.with_name(k.as_str());
                    inner.add_array(k.as_str(), a).py_err()?;
                }
                Err(_) => inner
                    .add_var(k.as_str(), spec_from_py(py, &k, &v)?)
                    .py_err()?,
            }
        }
        set_attrs(inner.attrs_mut(), attrs)?;
        Ok(Self {
            inner,
            lending: Lending::default(),
        })
    }

    /// Data variables, as data arrays.
    /// Plots of variables against each other: `ds.plot.scatter(x=..., y=...)`, `quiver`,
    /// `streamplot` (`annoray.plot`).
    #[getter]
    fn plot<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py()
            .import("annoray.plot")?
            .getattr("DatasetPlotAccessor")?
            .call1((slf,))
    }

    /// The data variables, as a live dict: `ds.data_vars["v"] = ...` adds or replaces one.
    #[getter]
    fn data_vars<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let names: Vec<String> = slf.borrow().keys();
        Self::children(slf, names)
    }

    /// Coordinates, as data arrays, in a dict whose assignments (`ds.coords["x"] = …`) and
    /// deletions apply to this dataset.
    #[getter]
    fn coords<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        live_dict(
            slf.as_any(),
            "CoordsDict",
            Self::plain_coords(slf)?.into_any(),
        )
    }

    /// Dimension names.
    #[getter]
    fn dims<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.dims().iter().map(Name::as_str))
    }

    /// Dimension name → length.
    #[getter]
    fn sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        sizes_dict(py, self.inner.sizes())
    }

    /// The global attributes, as a dict whose changes are applied to this dataset.
    #[getter]
    fn attrs<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let d = attrs_to_py(slf.py(), slf.borrow().inner.attrs())?;
        live_dict(slf.as_any(), "AttrsDict", d.into_any())
    }

    #[setter(attrs)]
    fn set_attrs_py(&mut self, attrs: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut a = annoray::Attrs::new();
        set_attrs(&mut a, Some(attrs))?;
        *self.inner.attrs_mut() = a;
        Ok(())
    }

    /// Names of the data variables.
    pub(crate) fn keys(&self) -> Vec<String> {
        self.inner
            .data_vars()
            .keys()
            .map(ToString::to_string)
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.data_vars().len()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyList::new(py, self.keys())?
            .try_iter()
            .map(Bound::into_any)
    }

    /// Variables and coordinates as attributes: `ds.temp`; registered accessors first.
    fn __getattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
        if let Some(acc) = accessor(slf.as_any(), "Dataset", name)? {
            return Ok(acc);
        }
        let found = {
            let me = slf.borrow();
            if me.inner.contains(name) {
                Some(me.array(name)?)
            } else {
                None
            }
        };
        if let Some(a) = found {
            return Ok(new_child_da(slf.as_any(), name, a)?.into_any());
        }
        Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "'Dataset' object has no attribute {name:?}"
        )))
    }

    /// `ds["temp"]` for a variable (or coordinate); `ds[["a", "b"]]` for a subset.
    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if let Ok(name) = key.extract::<String>() {
            let a = slf.borrow().array(&name)?;
            return Ok(new_child_da(slf.as_any(), &name, a)?.into_any());
        }
        let me = slf.borrow();
        let this = &*me;
        let names: Vec<String> = key.extract()?;
        let mut out = Dataset::new();
        for (k, c) in this.inner.coords() {
            out.add_coord(k.clone(), c.to_owned()).py_err()?;
        }
        for n in names {
            let v = this
                .inner
                .data_vars()
                .get(n.as_str())
                .ok_or_else(|| PyKeyError::new_err(n.clone()))?;
            out.add_var(n, v.to_owned()).py_err()?;
        }
        *out.attrs_mut() = this.inner.attrs().clone();
        Ok(new_ds(py, out)?.into_any())
    }

    /// A copy with variables added or replaced: `ds.assign(t2=ds["t"] * 2)`.
    #[pyo3(signature = (variables=None, **kwargs))]
    fn assign(
        &self,
        py: Python<'_>,
        variables: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let mut out = self.inner.clone();
        for (k, v) in merge_kwargs(variables, kwargs)? {
            if out.contains(&k) {
                out.remove(&k).py_err()?;
            }
            match v.cast::<PyDataArray>() {
                Ok(da) => {
                    out.add_array(k.as_str(), da.borrow().inner.clone().with_name(k.as_str()))
                }
                Err(_) => out.add_var(k.as_str(), spec_from_py(py, &k, &v)?),
            }
            .py_err()?;
        }
        new_ds(py, out)
    }

    /// Reorder dimensions of every variable: `ds.transpose("time", ...)`.
    #[pyo3(signature = (*dims))]
    fn transpose(&self, py: Python<'_>, dims: Vec<Bound<'_, PyAny>>) -> PyResult<Py<PyDataset>> {
        let order = transpose_order(py, &dims)?;
        let refs: Vec<&str> = order.iter().map(String::as_str).collect();
        new_ds(py, py.detach(|| self.inner.transpose(&refs)).py_err()?)
    }

    /// A copy without the named variables; `errors="ignore"` skips names that are not
    /// variables.
    #[pyo3(signature = (names, *, errors="raise"))]
    fn drop_vars(
        &self,
        py: Python<'_>,
        names: &Bound<'_, PyAny>,
        errors: &str,
    ) -> PyResult<Py<PyDataset>> {
        let names: Vec<String> = match names.extract::<String>() {
            Ok(n) => vec![n],
            Err(_) => names.extract()?,
        };
        let mut out = self.inner.clone();
        for n in names {
            if errors == "ignore" && !out.contains(&n) {
                continue;
            }
            out.remove(&n).py_err()?;
        }
        new_ds(py, out)
    }

    /// A copy with data variables turned into coordinates.
    fn set_coords(&self, py: Python<'_>, names: &Bound<'_, PyAny>) -> PyResult<Py<PyDataset>> {
        let names: Vec<String> = match names.extract::<String>() {
            Ok(n) => vec![n],
            Err(_) => names.extract()?,
        };
        let mut out = self.inner.clone();
        for n in names {
            out.set_coord(&n).py_err()?;
        }
        new_ds(py, out)
    }

    /// A copy with every variable read into memory.
    fn load(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        let out = py.detach(|| self.inner.clone().loaded()).py_err()?;
        new_ds(py, out)
    }

    /// Same as `load`.
    fn compute(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        self.load(py)
    }

    /// Select by position along named dimensions: `ds.isel(time=0, lat=slice(10, 20))`;
    /// integers drop the dimension, slices and integer arrays keep it. `drop=True` removes
    /// coordinates made 0-d; `missing_dims` is `"raise"`, `"warn"` or `"ignore"`.
    #[pyo3(signature = (indexers=None, drop=false, missing_dims="raise", **kwargs))]
    fn isel(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        drop: bool,
        missing_dims: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let dims = self.inner.dims();
        let args = known_dims(py, &dims, merge_kwargs(indexers, kwargs)?, missing_dims)?;
        let out = do_isel(py, &self.inner, args)?;
        new_ds(
            py,
            if drop {
                drop_scalarized_ds(&self.inner, out)?
            } else {
                out
            },
        )
    }

    /// Select by label: `ds.sel(time="2020-01", lat=slice(30, 60))`; slices are inclusive.
    /// `method` is `"nearest"`, `"pad"`/`"ffill"` or `"backfill"`/`"bfill"`, within
    /// `tolerance` if given; quantity labels are converted into the coordinate's unit.
    #[pyo3(signature = (indexers=None, method=None, tolerance=None, drop=false, **kwargs))]
    fn sel(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        method: Option<&str>,
        tolerance: Option<&Bound<'_, PyAny>>,
        drop: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let args = merge_kwargs(indexers, kwargs)?;
        let out = do_sel(py, &self.inner, args, method, tolerance)?;
        new_ds(
            py,
            if drop {
                drop_scalarized_ds(&self.inner, out)?
            } else {
                out
            },
        )
    }

    fn _fillna_number(&self, py: Python<'_>, value: f64) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.fillna(value)).py_err()?)
    }

    /// The dataset without the positions along `dim` where any (`how="any"`) or every
    /// (`how="all"`) value of the data variables is missing.
    #[pyo3(signature = (dim, how="any"))]
    fn dropna(&self, py: Python<'_>, dim: &str, how: &str) -> PyResult<Py<PyDataset>> {
        new_ds(py, self.inner.dropna(dim, how_from(how)?).py_err()?)
    }

    /// Conform to new coordinate labels: positions without a match get `fill_value` (NaN by
    /// default; `""` for text). `method` (`"nearest"`, `"pad"`, `"backfill"`) and
    /// `tolerance` match inexact labels.
    #[pyo3(signature = (indexers=None, method=None, tolerance=None, fill_value=None, **kwargs))]
    fn reindex(
        slf: &Bound<'_, Self>,
        indexers: Option<&Bound<'_, PyAny>>,
        method: Option<&Bound<'_, PyAny>>,
        tolerance: Option<&Bound<'_, PyAny>>,
        fill_value: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if method.is_some() || tolerance.is_some() || fill_value.is_some() {
            let kw = PyDict::new(py);
            kw.set_item("indexers", indexers)?;
            kw.set_item("method", method)?;
            kw.set_item("tolerance", tolerance)?;
            kw.set_item("fill_value", fill_value)?;
            if let Some(k) = kwargs {
                kw.update(k.as_mapping())?;
            }
            return Ok(py
                .import("annoray._methods")?
                .getattr("reindex_method")?
                .call((slf,), Some(&kw))?
                .unbind());
        }
        let mut out = slf.borrow().inner.clone();
        for (dim, v) in merge_kwargs(indexers, kwargs)? {
            let labels = labels_var(py, &dim, &v)?;
            out = out.reindex(&dim, &labels).py_err()?;
        }
        Ok(new_ds(py, out)?.into_any())
    }

    /// Interpolate along dimensions to new coordinate values (`method` `"linear"`,
    /// `"nearest"`, or log-scale `"log_coord"`, `"log_value"`, `"loglog"`); no
    /// extrapolation unless `extrapolate` asks for it (`"edge"`, `"extend"`, or a number for
    /// a power law). Text variables along the dimension are dropped.
    #[pyo3(signature = (coords=None, method="linear", extrapolate=None, **kwargs))]
    fn interp(
        &self,
        py: Python<'_>,
        coords: Option<&Bound<'_, PyAny>>,
        method: &str,
        extrapolate: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let m = interp_options(method, extrapolate)?;
        let mut out = self.inner.clone();
        for (dim, v) in merge_kwargs(coords, kwargs)? {
            let at = labels_var(py, &dim, &v)?;
            out = py.detach(|| out.interp(&dim, &at, m)).py_err()?;
        }
        new_ds(py, out)
    }

    /// Group by the values of a coordinate or variable (`"time.month"` for a datetime
    /// component): reductions give one value per group, arithmetic with a per-group object
    /// broadcasts back (`ds.groupby("time.month") - clim`).
    fn groupby(slf: &Bound<'_, Self>, group: String) -> PyResult<PyGroupBy> {
        slf.borrow().inner.groupby(&group).py_err()?;
        Ok(PyGroupBy {
            parent: Parent::Ds(slf.clone().unbind()),
            split: Split::Labels(group),
        })
    }

    /// Group into bins of a coordinate's values: `bins` is a number of equal-width bins or
    /// their edges; `right` closes bins on the right, `include_lowest` closes the first on
    /// the left.
    #[pyo3(signature = (group, bins, right=true, include_lowest=false))]
    fn groupby_bins(
        slf: &Bound<'_, Self>,
        group: String,
        bins: Vec<f64>,
        right: bool,
        include_lowest: bool,
    ) -> PyResult<PyGroupBy> {
        let opts = BinOptions::default()
            .right(right)
            .include_lowest(include_lowest);
        slf.borrow()
            .inner
            .groupby_bins(&group, &bins, &opts)
            .py_err()?;
        Ok(PyGroupBy {
            parent: Parent::Ds(slf.clone().unbind()),
            split: Split::Bins { group, bins, opts },
        })
    }

    /// Resample along a datetime dimension: `ds.resample(time="1D").mean()`. Every CF
    /// calendar is supported; times need not be sorted, and NaT belongs to no bin.
    #[pyo3(signature = (indexer=None, **kwargs))]
    fn resample(
        slf: &Bound<'_, Self>,
        indexer: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyGroupBy> {
        let (dim, freq) = one_pair(merge_kwargs(indexer, kwargs)?)?;
        let freq: String = freq.extract()?;
        slf.borrow().inner.resample(&dim, &freq).py_err()?;
        Ok(PyGroupBy {
            parent: Parent::Ds(slf.clone().unbind()),
            split: Split::Resample { dim, freq },
        })
    }

    /// A moving window along one or more dimensions: `ds.rolling(time=24).mean()`;
    /// `center` labels windows at their centre, `min_periods` sets the number of valid values
    /// a window needs.
    #[pyo3(signature = (dim=None, center=false, min_periods=None, **kwargs))]
    fn rolling(
        slf: &Bound<'_, Self>,
        dim: Option<&Bound<'_, PyAny>>,
        center: bool,
        min_periods: Option<usize>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let pairs = merge_kwargs(dim, kwargs)?;
        if pairs.len() > 1 {
            return multi_rolling(slf.as_any(), pairs, center, min_periods);
        }
        let (dim, window) = one_pair(pairs)?;
        let window: usize = window.extract()?;
        let opts = rolling_opts(center, min_periods);
        slf.borrow().inner.rolling(&dim, window, &opts).py_err()?;
        let r = PyRolling {
            parent: Parent::Ds(slf.clone().unbind()),
            dim,
            window,
            opts,
        };
        Ok(Py::new(slf.py(), r)?.into_any())
    }

    /// Write to netCDF (`format` as in xarray; netCDF-4 by default), with the GIL released.
    /// Lazy variables (opened from files, or computed from them) are read, computed and
    /// written chunk by chunk, so the data never has to fit in memory. `encoding` takes
    /// xarray's per-variable settings (`zlib`, `complevel`, `shuffle`, `chunksizes`,
    /// `compression`, `dtype`, `_FillValue`, `scale_factor`, `add_offset`, `units`,
    /// `calendar`); settings not given keep the variable's own.
    /// `mode="a"` adds or replaces variables in an existing file (rewritten through a
    /// temporary file, so a failed write leaves it as it was).
    #[pyo3(signature = (path, format="NETCDF4", encoding=None, unlimited_dims=None, mode="w"))]
    fn to_netcdf(
        slf: &Bound<'_, Self>,
        path: PathBuf,
        format: &str,
        encoding: Option<&Bound<'_, PyDict>>,
        unlimited_dims: Option<Vec<String>>,
        mode: &str,
    ) -> PyResult<()> {
        let py = slf.py();
        match mode {
            "w" => {}
            "a" => {
                let kw = PyDict::new(py);
                kw.set_item("format", format)?;
                if let Some(e) = encoding {
                    kw.set_item("encoding", e)?;
                }
                if let Some(u) = &unlimited_dims {
                    kw.set_item("unlimited_dims", u.clone())?;
                }
                py.import("annoray._io")?
                    .getattr("append_netcdf")?
                    .call((slf, path), Some(&kw))?;
                return Ok(());
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "mode must be \"w\" or \"a\", not {other:?}"
                )));
            }
        }
        let this = slf.borrow();
        let opts = WriteOptions::format(crate::raw::format_from_name(format)?);
        // A view: the encoding given is applied without copying any data.
        let mut ds = this.inner.view();
        if let Some(u) = unlimited_dims {
            ds.encoding_mut().unlimited_dims = u.into_iter().map(Into::into).collect();
        }
        for (k, v) in dict_iter(encoding.map(|e| e.as_any()))? {
            let enc = ds.encoding_of_mut(&k).py_err()?;
            crate::raw::update_encoding(enc, v.cast::<PyDict>()?)?;
        }
        py.detach(|| ds.to_netcdf_with(&path, &opts))
            .map_err(xr_error)
    }

    /// As an `xarray.Dataset` (units as `"units"` attributes).
    fn to_xarray<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = &slf.borrow().inner;
        let xr = py.import("xarray")?;
        let owner = slf.clone().into_any();
        let vars = PyDict::new(py);
        for (k, v) in this.data_vars() {
            vars.set_item(k.as_str(), xr_variable(py, &xr, owner.clone(), v)?)?;
        }
        let coords = PyDict::new(py);
        for (k, v) in this.coords() {
            coords.set_item(k.as_str(), xr_variable(py, &xr, owner.clone(), v)?)?;
        }
        let ds = xr
            .getattr("Dataset")?
            .call1((vars, coords, attrs_to_py(py, this.attrs())?))?;
        if !this.encoding().unlimited_dims.is_empty() {
            let u: Vec<&str> = this
                .encoding()
                .unlimited_dims
                .iter()
                .map(Name::as_str)
                .collect();
            ds.getattr("encoding")?
                .set_item("unlimited_dims", PyList::new(py, u)?)?;
        }
        Ok(ds)
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

/// An `xarray.Variable` sharing our values (read-only views for numbers).
fn xr_variable<'py>(
    py: Python<'py>,
    xr: &Bound<'py, PyModule>,
    owner: Bound<'py, PyAny>,
    v: &Variable,
) -> PyResult<Bound<'py, PyAny>> {
    let attrs = attrs_to_py(py, v.attrs())?;
    if let Some(u) = unit_str(v) {
        attrs.set_item("units", u)?;
    }
    let dims = PyTuple::new(py, v.dims().iter().map(Name::as_str))?;
    xr.getattr("Variable")?
        .call1((dims, values_of(owner, v)?, attrs))
}

// ----- helpers shared by the classes -----

/// Dimension names for `transpose`, with Python's `...` as `"..."`.
fn transpose_order(py: Python<'_>, dims: &[Bound<'_, PyAny>]) -> PyResult<Vec<String>> {
    dims.iter()
        .map(|d| {
            if d.is(py.Ellipsis()) {
                Ok("...".to_owned())
            } else {
                d.extract()
            }
        })
        .collect()
}

fn how_from(how: &str) -> PyResult<How> {
    match how {
        "any" => Ok(How::Any),
        "all" => Ok(How::All),
        other => Err(PyValueError::new_err(format!(
            "how must be \"any\" or \"all\", not {other:?}"
        ))),
    }
}

fn interp_method(m: &str) -> PyResult<InterpMethod> {
    match m {
        "linear" => Ok(InterpMethod::Linear),
        "nearest" => Ok(InterpMethod::Nearest),
        "log_coord" => Ok(InterpMethod::LogCoord),
        "log_value" => Ok(InterpMethod::LogValue),
        "loglog" | "log_log" => Ok(InterpMethod::LogLog),
        other => Err(PyValueError::new_err(format!(
            "method must be \"linear\", \"nearest\", \"log_coord\", \"log_value\" or \"loglog\", \
             not {other:?}"
        ))),
    }
}

fn extrapolation(e: Option<&Bound<'_, PyAny>>) -> PyResult<Extrapolate> {
    let Some(e) = e.filter(|e| !e.is_none()) else {
        return Ok(Extrapolate::Nan);
    };
    if !e.is_instance_of::<PyString>() {
        // A number: a power law with that exponent from the edge point.
        return Ok(Extrapolate::PowerLaw(e.extract::<f64>()?));
    }
    match e.extract::<String>()?.as_str() {
        "nan" => Ok(Extrapolate::Nan),
        "edge" => Ok(Extrapolate::Edge),
        "extend" => Ok(Extrapolate::Extend),
        other => Err(PyValueError::new_err(format!(
            "extrapolate must be None, \"edge\", \"extend\" or a power-law exponent (or a \
             (below, above) pair), not {other:?}"
        ))),
    }
}

fn interp_options(method: &str, extrapolate: Option<&Bound<'_, PyAny>>) -> PyResult<InterpOptions> {
    let opts = InterpOptions::new(interp_method(method)?);
    Ok(match extrapolate {
        Some(e) if e.is_instance_of::<PyTuple>() => {
            let (below, above): (Bound<'_, PyAny>, Bound<'_, PyAny>) = e.extract()?;
            opts.below(extrapolation(Some(&below))?)
                .above(extrapolation(Some(&above))?)
        }
        e => opts.extrapolate(extrapolation(e)?),
    })
}

fn rolling_opts(center: bool, min_periods: Option<usize>) -> RollingOptions {
    let mut o = RollingOptions::default();
    o.center = center;
    o.min_periods = min_periods;
    o
}

/// The single `dim=value` pair of `resample` and `rolling`.
/// The accessor registered as `name` for `kind` (`annoray.register_*_accessor`), if any.
fn accessor(obj: &Bound<'_, PyAny>, kind: &str, name: &str) -> PyResult<Option<Py<PyAny>>> {
    if name.starts_with("__") {
        return Ok(None);
    }
    let r = obj
        .py()
        .import("annoray._accessors")?
        .getattr("lookup")?
        .call1((obj, kind, name))?;
    Ok((!r.is_none()).then(|| r.unbind()))
}

/// Windows along several dimensions (`annoray._rolling.MultiRolling`).
fn multi_rolling(
    obj: &Bound<'_, PyAny>,
    pairs: IselArgs<'_>,
    center: bool,
    min_periods: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let py = obj.py();
    let windows = PyDict::new(py);
    for (d, w) in pairs {
        windows.set_item(d, w.extract::<usize>()?)?;
    }
    Ok(py
        .import("annoray._rolling")?
        .getattr("MultiRolling")?
        .call1((obj, windows, center, min_periods))?
        .unbind())
}

fn one_pair(args: IselArgs<'_>) -> PyResult<(String, Bound<'_, PyAny>)> {
    let mut it = args.into_iter();
    match (it.next(), it.next()) {
        (Some(kv), None) => Ok(kv),
        _ => Err(PyValueError::new_err(
            "give exactly one dimension, e.g. time=...",
        )),
    }
}

/// New labels (or interpolation points) along `dim`, as a 1-D variable.
fn labels_var(py: Python<'_>, dim: &str, v: &Bound<'_, PyAny>) -> PyResult<Variable> {
    if let Ok(da) = v.cast::<PyDataArray>() {
        return Ok(da.borrow().inner.variable().to_owned());
    }
    let dims = PyTuple::new(py, [dim])?.into_any();
    variable_from_py(py, Some(&dims), v, None, None)
}

// ----- group-by and resample -----

enum Parent {
    Da(Py<PyDataArray>),
    Ds(Py<PyDataset>),
}

enum Split {
    Labels(String),
    Bins {
        group: String,
        bins: Vec<f64>,
        opts: BinOptions,
    },
    Resample {
        dim: String,
        freq: String,
    },
}

/// Groups of a data array or dataset (from `groupby`, `groupby_bins` or `resample`).
#[pyclass(frozen, name = "GroupBy", module = "annoray")]
pub(crate) struct PyGroupBy {
    parent: Parent,
    split: Split,
}

/// Run `$body` with the group-by of the parent bound to `$g` and wrap its result.
macro_rules! with_groups {
    ($self:ident, $py:ident, |$g:ident| $body:expr) => {
        match &$self.parent {
            Parent::Da(p) => {
                let obj = &p.borrow($py).inner;
                let $g = match &$self.split {
                    Split::Labels(n) => obj.groupby(n),
                    Split::Bins { group, bins, opts } => obj.groupby_bins(group, bins, opts),
                    Split::Resample { dim, freq } => obj.resample(dim, freq),
                }
                .py_err()?;
                let out = $py.detach(|| $body).py_err()?;
                Ok(new_da($py, out)?.into_any())
            }
            Parent::Ds(p) => {
                let obj = &p.borrow($py).inner;
                let $g = match &$self.split {
                    Split::Labels(n) => obj.groupby(n),
                    Split::Bins { group, bins, opts } => obj.groupby_bins(group, bins, opts),
                    Split::Resample { dim, freq } => obj.resample(dim, freq),
                }
                .py_err()?;
                let out = $py.detach(|| $body).py_err()?;
                Ok(new_ds($py, out)?.into_any())
            }
        }
    };
}

impl PyGroupBy {
    #[allow(clippy::unused_self)]
    fn dims_of_axis(&self, _axis: &Bound<'_, PyAny>) -> PyResult<Vec<Name>> {
        Err(PyTypeError::new_err(
            "axis= is not supported for groups; use dim=",
        ))
    }

    fn reduce_with(
        &self,
        py: Python<'_>,
        op: Reduction,
        dims: Option<Vec<Name>>,
        opts: &ReduceOptions,
    ) -> PyResult<Py<PyAny>> {
        if dims.is_some() {
            return Err(PyValueError::new_err(
                "group reductions run over the grouped dimension; dim is not supported yet",
            ));
        }
        with_groups!(self, py, |g| g.reduce(op, opts))
    }

    pub(crate) fn binary_with(
        &self,
        py: Python<'_>,
        op: BinOp,
        o: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        let Parent::Da(p) = &self.parent else {
            return Err(PyTypeError::new_err("group arithmetic needs a DataArray"));
        };
        let Ok(other) = o.cast::<PyDataArray>() else {
            return Ok(py.NotImplemented());
        };
        if reflected {
            return Ok(py.NotImplemented());
        }
        let obj = &p.borrow(py).inner;
        let other = &other.borrow().inner;
        let g = match &self.split {
            Split::Labels(n) => obj.groupby(n),
            Split::Bins { group, bins, opts } => obj.groupby_bins(group, bins, opts),
            Split::Resample { dim, freq } => obj.resample(dim, freq),
        }
        .py_err()?;
        let out = py
            .detach(|| g.binary(op, other, UnitPolicy::Strict))
            .py_err()?;
        Ok(new_da(py, out)?.into_any())
    }
}

reductions!(PyGroupBy);
arithmetic!(PyGroupBy);

#[pymethods]
impl PyGroupBy {
    /// The first element of each group.
    fn first(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        with_groups!(self, py, |g| g.first())
    }

    /// The last element of each group.
    fn last(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        with_groups!(self, py, |g| g.last())
    }

    /// Number of groups.
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        match &self.parent {
            Parent::Da(p) => {
                let obj = &p.borrow(py).inner;
                Ok(match &self.split {
                    Split::Labels(n) => obj.groupby(n),
                    Split::Bins { group, bins, opts } => obj.groupby_bins(group, bins, opts),
                    Split::Resample { dim, freq } => obj.resample(dim, freq),
                }
                .py_err()?
                .len())
            }
            Parent::Ds(p) => {
                let obj = &p.borrow(py).inner;
                Ok(match &self.split {
                    Split::Labels(n) => obj.groupby(n),
                    Split::Bins { group, bins, opts } => obj.groupby_bins(group, bins, opts),
                    Split::Resample { dim, freq } => obj.resample(dim, freq),
                }
                .py_err()?
                .len())
            }
        }
    }

    /// `(label, group)` pairs, as xarray iterates.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pairs = PyList::empty(py);
        macro_rules! fill {
            ($obj:expr, $wrap:ident) => {{
                let g = match &self.split {
                    Split::Labels(n) => $obj.groupby(n),
                    Split::Bins { group, bins, opts } => $obj.groupby_bins(group, bins, opts),
                    Split::Resample { dim, freq } => $obj.resample(dim, freq),
                }
                .py_err()?;
                let labels = typed_to_py(py, g.labels().to_owned().into_typed().py_err()?)?
                    .call_method0("tolist")?;
                for i in 0..g.len() {
                    let part = $wrap(py, g.group(i).py_err()?)?;
                    pairs.append((labels.get_item(i)?, part))?;
                }
            }};
        }
        match &self.parent {
            Parent::Da(p) => {
                let held = p.borrow(py);
                fill!(held.inner, new_da)
            }
            Parent::Ds(p) => {
                let held = p.borrow(py);
                fill!(held.inner, new_ds)
            }
        }
        pairs.try_iter().map(Bound::into_any)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let what = match &self.split {
            Split::Labels(n) => format!("groupby({n:?})"),
            Split::Bins { group, .. } => format!("groupby_bins({group:?})"),
            Split::Resample { dim, freq } => format!("resample({dim}={freq:?})"),
        };
        Ok(format!(
            "<annoray.GroupBy {what}, {} groups>",
            self.__len__(py)?
        ))
    }
}

// ----- rolling -----

/// Moving windows along one dimension.
#[pyclass(frozen, name = "Rolling", module = "annoray")]
pub(crate) struct PyRolling {
    parent: Parent,
    dim: String,
    window: usize,
    opts: RollingOptions,
}

macro_rules! rolling_reductions {
    ($($m:ident $op:ident),*) => {
        #[pymethods]
        impl PyRolling {
            $(
                #[doc = concat!("Moving `", stringify!($m), "`.")]
                fn $m(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                    self.reduce(py, Reduction::$op)
                }
            )*
        }
    };
}

rolling_reductions!(sum Sum, prod Prod, mean Mean, median Median, std Std, var Var,
    min Min, max Max, count Count, argmin ArgMin, argmax ArgMax, any Any, all All);

#[pymethods]
impl PyRolling {
    /// Reduce each window with the named operation (`"mean"`, …).
    #[pyo3(name = "reduce")]
    fn reduce_named(&self, py: Python<'_>, op: &str) -> PyResult<Py<PyAny>> {
        self.reduce(py, reduction_from(op)?)
    }

    /// The windows as a new trailing dimension (NaN-padded).
    fn construct(&self, py: Python<'_>, window_dim: &str) -> PyResult<Py<PyAny>> {
        let Parent::Da(p) = &self.parent else {
            return Err(PyTypeError::new_err("construct needs a DataArray"));
        };
        let held = p.borrow(py);
        let r = held
            .inner
            .rolling(&self.dim, self.window, &self.opts)
            .py_err()?;
        let out = py.detach(|| r.construct(window_dim)).py_err()?;
        Ok(new_da(py, out)?.into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "<annoray.Rolling {}={} center={}>",
            self.dim, self.window, self.opts.center
        )
    }
}

impl PyRolling {
    fn reduce(&self, py: Python<'_>, op: Reduction) -> PyResult<Py<PyAny>> {
        match &self.parent {
            Parent::Da(p) => {
                let held = p.borrow(py);
                let r = held
                    .inner
                    .rolling(&self.dim, self.window, &self.opts)
                    .py_err()?;
                Ok(new_da(py, py.detach(|| r.reduce(op)).py_err()?)?.into_any())
            }
            Parent::Ds(p) => {
                let held = p.borrow(py);
                let r = held
                    .inner
                    .rolling(&self.dim, self.window, &self.opts)
                    .py_err()?;
                Ok(new_ds(py, py.detach(|| r.reduce(op)).py_err()?)?.into_any())
            }
        }
    }
}

pub(crate) fn reduction_from(op: &str) -> PyResult<Reduction> {
    Ok(match op {
        "sum" => Reduction::Sum,
        "prod" => Reduction::Prod,
        "mean" => Reduction::Mean,
        "median" => Reduction::Median,
        "std" => Reduction::Std,
        "var" => Reduction::Var,
        "min" => Reduction::Min,
        "max" => Reduction::Max,
        "count" => Reduction::Count,
        "argmin" => Reduction::ArgMin,
        "argmax" => Reduction::ArgMax,
        "any" => Reduction::Any,
        "all" => Reduction::All,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown reduction {other:?}"
            )));
        }
    })
}

// ----- .dt -----

/// Date and time components of a data array (`da.dt.year`, …).
#[pyclass(frozen, name = "DatetimeAccessor", module = "annoray")]
pub(crate) struct PyDt {
    parent: Py<PyDataArray>,
}

#[pymethods]
impl PyDt {
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let da = &self.parent.borrow(py).inner;
        let var = match da.variable().dt(name) {
            Ok(v) => v,
            // Methods and duration fields written in Python (`annoray._dt`).
            Err(_) => {
                return Ok(py
                    .import("annoray._dt")?
                    .getattr("extra")?
                    .call1((self.parent.bind(py), name))?
                    .unbind());
            }
        };
        let mut out = DataArray::new(var);
        if let Some(n) = da.name() {
            out = out.with_name(n.clone());
        }
        for (k, c) in da.coords() {
            out = out.with_coord(k.clone(), c.to_owned()).py_err()?;
        }
        Ok(new_da(py, out)?.into_any())
    }
}

// ----- module functions -----

/// Open a netCDF-3, netCDF-4 or HDF5 file as a native `Dataset`. Data is memory-mapped
/// and read on demand; CF conventions are decoded (times, packing, fill values, units).
#[pyfunction]
#[pyo3(signature = (path, *, group=None, mask_and_scale=true, decode_times=true,
    decode_units=true, decode_coords=true, concat_characters=true, lazy=true))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub(crate) fn open_dataset(
    py: Python<'_>,
    path: PathBuf,
    group: Option<String>,
    mask_and_scale: bool,
    decode_times: bool,
    decode_units: bool,
    decode_coords: bool,
    concat_characters: bool,
    lazy: bool,
) -> PyResult<Py<PyDataset>> {
    let opts = open_options(
        group,
        mask_and_scale,
        decode_times,
        decode_units,
        decode_coords,
        concat_characters,
        lazy,
    );
    let opened = py
        .detach(|| open_dataset_with(&path, &opts))
        .map_err(xr_error)?;
    warn_all(py, &opened.warnings)?;
    new_ds(py, opened.dataset)
}

#[allow(clippy::fn_params_excessive_bools)]
pub(crate) fn open_options(
    group: Option<String>,
    mask_and_scale: bool,
    decode_times: bool,
    decode_units: bool,
    decode_coords: bool,
    concat_characters: bool,
    lazy: bool,
) -> OpenOptions {
    let mut d = DecodeOptions::default();
    d.mask_and_scale = mask_and_scale;
    d.decode_times = decode_times;
    d.decode_units = decode_units;
    d.decode_coords = decode_coords;
    d.concat_characters = concat_characters;
    let mut o = OpenOptions::default().decode(d).lazy(lazy);
    if let Some(g) = group {
        o = o.group(g);
    }
    o
}

pub(crate) fn warn_all(py: Python<'_>, warnings: &[String]) -> PyResult<()> {
    for w in warnings {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            &std::ffi::CString::new(w.as_str()).map_err(value_error)?,
            1,
        )?;
    }
    Ok(())
}

/// CF-decode a dataset opened (or built) without decoding (xarray's `decode_cf`): times,
/// packing, fill values, units and coordinates, by the same rules as `open_dataset`.
#[pyfunction]
#[pyo3(signature = (obj, *, mask_and_scale=true, decode_times=true, decode_units=true,
    decode_coords=true, concat_characters=true))]
#[allow(clippy::fn_params_excessive_bools)]
pub(crate) fn decode_cf(
    py: Python<'_>,
    obj: &Bound<'_, PyDataset>,
    mask_and_scale: bool,
    decode_times: bool,
    decode_units: bool,
    decode_coords: bool,
    concat_characters: bool,
) -> PyResult<Py<PyDataset>> {
    let mut d = DecodeOptions::default();
    d.mask_and_scale = mask_and_scale;
    d.decode_times = decode_times;
    d.decode_units = decode_units;
    d.decode_coords = decode_coords;
    d.concat_characters = concat_characters;
    let ds = obj.borrow();
    let ds = &ds.inner;
    let decoded = py
        .detach(|| {
            let raw = annoray::cf::encode(
                ds,
                &annoray::cf::EncodeOptions::format(annoray::io::Format::Netcdf4),
            )?;
            annoray::cf::decode(&raw, &d)
        })
        .map_err(|e| xr_error(e.into()))?;
    warn_all(py, &decoded.warnings)?;
    new_ds(py, decoded.dataset)
}

/// Open several files as one dataset: a glob pattern or a list of paths, combined by their
/// coordinates (default) or concatenated along `concat_dim` in the order given.
#[pyfunction]
#[pyo3(signature = (paths, *, concat_dim=None, lazy=true))]
pub(crate) fn open_mfdataset(
    py: Python<'_>,
    paths: &Bound<'_, PyAny>,
    concat_dim: Option<String>,
    lazy: bool,
) -> PyResult<Py<PyDataset>> {
    let paths: Vec<PathBuf> = match paths.extract::<String>() {
        Ok(pattern) => annoray::io::glob(&pattern).map_err(xr_error)?,
        Err(_) => paths.extract()?,
    };
    let mut opts = MfOptions::default().open(OpenOptions::default().lazy(lazy));
    if let Some(d) = concat_dim {
        opts = opts.concat_dim(d);
    }
    let out = py
        .detach(|| annoray::io::open_mfdataset(&paths, &opts))
        .map_err(xr_error)?;
    warn_all(py, &out.warnings)?;
    new_ds(py, out.dataset)
}

/// Set the streaming budget (default 64 MiB): how much on-disk data reductions of lazy data
/// (`mean`, `sum`, `std`, `var`, `min`, `max`) read at a time, and how many bytes of chunks
/// reading and writing decode or compress at once. Memory use when streaming stays near a
/// few times this, whatever the size of the data.
#[pyfunction]
pub(crate) fn set_stream_bytes(bytes: usize) {
    annoray::set_stream_bytes(bytes);
}

/// The current streaming budget, in bytes.
#[pyfunction]
pub(crate) fn stream_bytes() -> usize {
    annoray::stream_bytes()
}

/// `periods` regular dates from `start` at a pandas frequency, as `datetime64[ns]` (xarray's
/// `date_range`): fixed steps (`"D"`, `"6h"`, …) or period starts (`"MS"`, `"QS"`, `"YS"`).
#[pyfunction]
pub(crate) fn date_range<'py>(
    py: Python<'py>,
    start: &str,
    periods: usize,
    freq: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let dates = annoray::date_range(start, periods, freq).py_err()?;
    typed_to_py(py, TypedArray::DateTime(dates.into_dyn()))
}

fn concat_which(s: &str) -> PyResult<annoray::Concat> {
    Ok(match s {
        "auto" => annoray::Concat::Auto,
        "minimal" => annoray::Concat::Minimal,
        "different" => annoray::Concat::Different,
        "all" => annoray::Concat::All,
        other => {
            return Err(PyValueError::new_err(format!(
                "expected \"minimal\", \"different\", \"all\" or \"auto\", not {other:?}"
            )));
        }
    })
}

fn compat_from(s: &str) -> PyResult<annoray::Compat> {
    Ok(match s {
        "override" => annoray::Compat::Override,
        "equals" => annoray::Compat::Equals,
        "identical" => annoray::Compat::Identical,
        "no_conflicts" => annoray::Compat::NoConflicts,
        other => {
            return Err(PyValueError::new_err(format!(
                "compat must be \"override\", \"equals\", \"identical\" or \"no_conflicts\", \
                 not {other:?}"
            )));
        }
    })
}

/// Regular dates in a CF calendar (as CF dates): `periods` of them, or up to `end`.
#[pyfunction]
#[pyo3(signature = (start, end, periods, freq, calendar))]
pub(crate) fn _date_range_in<'py>(
    py: Python<'py>,
    start: &str,
    end: Option<&str>,
    periods: Option<usize>,
    freq: &str,
    calendar: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let cal = calendar
        .parse::<annoray::units::Calendar>()
        .map_err(|e| crate::convert::unit_error(&e))?;
    let dates = annoray::date_range_in(start, end, periods, freq, cal).py_err()?;
    typed_to_py(py, TypedArray::CfDate(dates.into_dyn()))
}

/// Concatenate data arrays (or datasets) along a dimension (new or existing). `dim` may be
/// a name, or a pandas Index / DataArray whose name is the dimension and whose values
/// label it. Defaults are xarray's new ones (exact join, minimal variables, `override`).
#[pyfunction]
#[pyo3(signature = (objs, dim, *, data_vars="auto", coords="minimal", compat="override", join="exact"))]
pub(crate) fn concat(
    py: Python<'_>,
    objs: Vec<Bound<'_, PyAny>>,
    dim: &Bound<'_, PyAny>,
    data_vars: &str,
    coords: &str,
    compat: &str,
    join: &str,
) -> PyResult<Py<PyAny>> {
    let opts = ConcatOptions::default()
        .data_vars(concat_which(data_vars)?)
        .coords(concat_which(coords)?)
        .compat(compat_from(compat)?)
        .join(join_from(join)?);
    // A labelled dimension: concatenate, then label it.
    if !dim.is_instance_of::<PyString>() {
        let name: String = dim.getattr("name")?.extract().map_err(|_| {
            PyTypeError::new_err("dim must be a name, or a named Index or DataArray")
        })?;
        let values = py.import("numpy")?.call_method1("asarray", (dim,))?;
        let out = concat(
            py,
            objs,
            &PyString::new(py, &name).into_any(),
            data_vars,
            coords,
            compat,
            join,
        )?;
        let kw = PyDict::new(py);
        kw.set_item(&name, (name.as_str(), values))?;
        return Ok(out
            .bind(py)
            .call_method("assign_coords", (), Some(&kw))?
            .unbind());
    }
    let dim: String = dim.extract()?;
    let dim = dim.as_str();
    if objs.iter().all(|o| o.is_instance_of::<PyDataArray>()) {
        let arrays: Vec<Bound<'_, PyDataArray>> = objs
            .iter()
            .map(|o| o.cast::<PyDataArray>().cloned())
            .collect::<Result<_, _>>()?;
        let held: Vec<PyRef<'_, PyDataArray>> = arrays.iter().map(Bound::borrow).collect();
        let refs: Vec<&DataArray> = held.iter().map(|a| &a.inner).collect();
        let out = py.detach(|| annoray::concat(&refs, dim, &opts)).py_err()?;
        return Ok(new_da(py, out)?.into_any());
    }
    let sets: Vec<Bound<'_, PyDataset>> = objs
        .iter()
        .map(|o| o.cast::<PyDataset>().cloned())
        .collect::<Result<_, _>>()?;
    let held: Vec<PyRef<'_, PyDataset>> = sets.iter().map(Bound::borrow).collect();
    let refs: Vec<&Dataset> = held.iter().map(|a| &a.inner).collect();
    let out = py
        .detach(|| annoray::concat_datasets(&refs, dim, &opts))
        .py_err()?;
    Ok(new_ds(py, out)?.into_any())
}

/// Merge datasets (or named data arrays) into one dataset; `compat` decides variables of the
/// same name (`"override"`, `"equals"`, `"identical"`, `"no_conflicts"`).
#[pyfunction]
#[pyo3(signature = (objs, *, join="exact", compat="override"))]
pub(crate) fn merge(
    py: Python<'_>,
    objs: Vec<Bound<'_, PyAny>>,
    join: &str,
    compat: &str,
) -> PyResult<Py<PyDataset>> {
    let mut sets = Vec::with_capacity(objs.len());
    for o in &objs {
        if let Ok(da) = o.cast::<PyDataArray>() {
            let a = &da.borrow().inner;
            let name = a
                .name()
                .ok_or_else(|| PyValueError::new_err("unnamed DataArray cannot be merged"))?;
            let mut ds = Dataset::new();
            ds.add_array(name.clone(), a.clone()).py_err()?;
            sets.push(ds);
        } else {
            sets.push(o.cast::<PyDataset>()?.borrow().inner.clone());
        }
    }
    let refs: Vec<&Dataset> = sets.iter().collect();
    let mut opts = MergeOptions::default();
    opts.join = join_from(join)?;
    opts.compat = compat_from(compat)?;
    new_ds(py, py.detach(|| annoray::merge(&refs, &opts)).py_err()?)
}

/// Combine datasets that tile one dimension, ordered by its coordinate (xarray's
/// `combine_by_coords`); datasets that do not differ along any dimension are merged.
#[pyfunction]
#[pyo3(signature = (data_objects, **kwargs))]
pub(crate) fn combine_by_coords(
    py: Python<'_>,
    data_objects: Vec<Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyDataset>> {
    let _ = kwargs;
    let mut sets = Vec::with_capacity(data_objects.len());
    for o in &data_objects {
        if let Ok(da) = o.cast::<PyDataArray>() {
            let a = da.borrow().inner.clone();
            let name = a
                .name()
                .cloned()
                .ok_or_else(|| PyValueError::new_err("unnamed DataArray cannot be combined"))?;
            let mut ds = Dataset::new();
            ds.add_array(name, a).py_err()?;
            sets.push(ds);
        } else {
            sets.push(o.cast::<PyDataset>()?.borrow().inner.clone());
        }
    }
    let refs: Vec<&Dataset> = sets.iter().collect();
    let opts = ConcatOptions::default();
    new_ds(
        py,
        py.detach(|| annoray::combine_by_coords(&refs, &opts))
            .py_err()?,
    )
}

/// Align data arrays (or datasets) on their shared coordinates.
#[pyfunction]
#[pyo3(signature = (*objs, join="inner"))]
pub(crate) fn align<'py>(
    py: Python<'py>,
    objs: Vec<Bound<'py, PyAny>>,
    join: &str,
) -> PyResult<Bound<'py, PyTuple>> {
    let j = join_from(join)?;
    if objs.iter().all(|o| o.is_instance_of::<PyDataArray>()) {
        let arrays: Vec<Bound<'_, PyDataArray>> = objs
            .iter()
            .map(|o| o.cast::<PyDataArray>().cloned())
            .collect::<Result<_, _>>()?;
        let held: Vec<PyRef<'_, PyDataArray>> = arrays.iter().map(Bound::borrow).collect();
        let refs: Vec<&DataArray> = held.iter().map(|a| &a.inner).collect();
        let out = py.detach(|| annoray::align(&refs, j)).py_err()?;
        let items = out
            .into_iter()
            .map(|a| new_da(py, a))
            .collect::<PyResult<Vec<_>>>()?;
        return PyTuple::new(py, items);
    }
    let sets: Vec<Bound<'_, PyDataset>> = objs
        .iter()
        .map(|o| o.cast::<PyDataset>().cloned())
        .collect::<Result<_, _>>()?;
    let held: Vec<PyRef<'_, PyDataset>> = sets.iter().map(Bound::borrow).collect();
    let refs: Vec<&Dataset> = held.iter().map(|a| &a.inner).collect();
    let out = py.detach(|| annoray::align_datasets(&refs, j)).py_err()?;
    let items = out
        .into_iter()
        .map(|a| new_ds(py, a))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, items)
}

fn join_from(join: &str) -> PyResult<Join> {
    Ok(match join {
        "inner" => Join::Inner,
        "outer" => Join::Outer,
        "left" => Join::Left,
        "right" => Join::Right,
        "exact" => Join::Exact,
        other => return Err(PyValueError::new_err(format!("unknown join {other:?}"))),
    })
}

/// A variable of an xarray object, with its `"units"` attribute as our unit.
fn var_from_xarray(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<Variable> {
    let attrs = v.getattr("attrs")?;
    let attrs = attrs.cast::<PyDict>()?;
    let dims = v.getattr("dims")?;
    let values = v.getattr("values")?;
    variable_from_py(py, Some(&dims), &values, Some(attrs), None)
}

/// Convert an `xarray.DataArray` or `xarray.Dataset` (units from `"units"` attributes).
#[pyfunction]
pub(crate) fn from_xarray(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let xr = py.import("xarray")?;
    if obj.is_instance(&xr.getattr("DataArray")?)? {
        let mut out = DataArray::new(var_from_xarray(py, &obj.getattr("variable")?)?);
        if let Ok(n) = obj.getattr("name")?.extract::<String>() {
            out = out.with_name(n);
        }
        for (k, c) in obj
            .getattr("coords")?
            .call_method0("items")?
            .try_iter()?
            .map(|kv| {
                let kv = kv?;
                Ok::<_, PyErr>((kv.get_item(0)?.extract::<String>()?, kv.get_item(1)?))
            })
            .collect::<PyResult<Vec<_>>>()?
        {
            out = out
                .with_coord(k, var_from_xarray(py, &c.getattr("variable")?)?)
                .py_err()?;
        }
        return Ok(new_da(py, out)?.into_any());
    }
    if obj.is_instance(&xr.getattr("Dataset")?)? {
        let mut ds = Dataset::new();
        let coords: Vec<String> = obj
            .getattr("coords")?
            .try_iter()?
            .map(|k| k?.extract())
            .collect::<PyResult<_>>()?;
        for k in &coords {
            let v = obj.getattr("variables")?.get_item(k)?;
            ds.add_coord(k.as_str(), var_from_xarray(py, &v)?)
                .py_err()?;
        }
        let names: Vec<String> = obj
            .getattr("data_vars")?
            .try_iter()?
            .map(|k| k?.extract())
            .collect::<PyResult<_>>()?;
        for k in &names {
            let v = obj.getattr("variables")?.get_item(k)?;
            ds.add_var(k.as_str(), var_from_xarray(py, &v)?).py_err()?;
        }
        set_attrs(ds.attrs_mut(), Some(&obj.getattr("attrs")?))?;
        if let Ok(u) = obj
            .getattr("encoding")?
            .get_item("unlimited_dims")
            .and_then(|u| u.extract::<Vec<String>>())
        {
            ds.encoding_mut().unlimited_dims = u.into_iter().map(Into::into).collect();
        }
        return Ok(new_ds(py, ds)?.into_any());
    }
    if obj.is_instance(&xr.getattr("DataTree")?)? {
        return Ok(crate::tree::tree_from_xarray(py, obj)?.into_any());
    }
    Err(PyTypeError::new_err(
        "expected an xarray DataArray, Dataset or DataTree",
    ))
}

/// `func(group, *args, **kwargs)` as a closure for the core's group `map`, its Python error
/// kept in `err`.
fn group_call<'a, T: Clone + 'a>(
    py: Python<'a>,
    func: &'a Bound<'a, PyAny>,
    args: &'a Bound<'a, PyTuple>,
    kwargs: Option<&'a Bound<'a, PyDict>>,
    err: &'a std::cell::RefCell<Option<PyErr>>,
    wrap: impl Fn(Python<'a>, T) -> PyResult<Bound<'a, PyAny>> + 'a,
    unwrap: impl Fn(&Bound<'a, PyAny>) -> PyResult<T> + 'a,
) -> impl FnMut(T) -> annoray::Result<T> + 'a {
    move |x: T| {
        let call = || -> PyResult<T> {
            let mut all = vec![wrap(py, x)?];
            all.extend(args.iter());
            let r = func.call(PyTuple::new(py, all)?, kwargs)?;
            unwrap(&r)
        };
        call().map_err(|e| {
            let msg = e.to_string();
            *err.borrow_mut() = Some(e);
            annoray::CoreError::InvalidReduction(msg)
        })
    }
}

#[pymethods]
impl PyGroupBy {
    /// `func(group, *args, **kwargs)` for every group, combined: results that keep the
    /// grouped dimension go back to the original order; others are stacked along the group
    /// dimension (xarray's `GroupBy.map`).
    #[pyo3(signature = (func, args=None, shortcut=None, **kwargs))]
    fn map(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: Option<&Bound<'_, PyTuple>>,
        shortcut: Option<bool>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = shortcut;
        let empty = PyTuple::empty(py);
        let args = args.unwrap_or(&empty);
        let err = std::cell::RefCell::new(None);
        macro_rules! run {
            ($obj:expr, $wrap:ident, $cls:ty, $what:literal) => {{
                let g = match &self.split {
                    Split::Labels(n) => $obj.groupby(n),
                    Split::Bins { group, bins, opts } => $obj.groupby_bins(group, bins, opts),
                    Split::Resample { dim, freq } => $obj.resample(dim, freq),
                }
                .py_err()?;
                let f = group_call(
                    py,
                    func,
                    args,
                    kwargs,
                    &err,
                    |py, x| Ok($wrap(py, x)?.into_bound(py).into_any()),
                    |r| {
                        Ok(r.cast::<$cls>()
                            .map_err(|_| PyTypeError::new_err($what))?
                            .borrow()
                            .inner
                            .clone())
                    },
                );
                let out = g.map(f);
                if let Some(e) = err.borrow_mut().take() {
                    return Err(e);
                }
                Ok($wrap(py, out.py_err()?)?.into_any())
            }};
        }
        match &self.parent {
            Parent::Da(p) => {
                let held = p.borrow(py);
                run!(
                    held.inner,
                    new_da,
                    PyDataArray,
                    "GroupBy.map: func must return a DataArray"
                )
            }
            Parent::Ds(p) => {
                let held = p.borrow(py);
                run!(
                    held.inner,
                    new_ds,
                    PyDataset,
                    "GroupBy.map: func must return a Dataset"
                )
            }
        }
    }

    /// The dimension being grouped along.
    #[getter]
    fn _dim(&self, py: Python<'_>) -> PyResult<String> {
        macro_rules! dim {
            ($obj:expr) => {{
                let g = match &self.split {
                    Split::Labels(n) => $obj.groupby(n),
                    Split::Bins { group, bins, opts } => $obj.groupby_bins(group, bins, opts),
                    Split::Resample { dim, freq } => $obj.resample(dim, freq),
                }
                .py_err()?;
                Ok(g.dim().to_string())
            }};
        }
        match &self.parent {
            Parent::Da(p) => {
                let held = p.borrow(py);
                dim!(held.inner)
            }
            Parent::Ds(p) => {
                let held = p.borrow(py);
                dim!(held.inner)
            }
        }
    }
}
