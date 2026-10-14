//! Native, units-aware `DataArray` and `Dataset` (and their group-by and rolling helpers),
//! backed by xarray-rs.
//!
//! Objects are immutable: every operation returns a new object, so the values a numpy view
//! shows can never change under it. Heavy work runs with the GIL released.

use std::path::PathBuf;

use numpy::PyArrayDyn;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use xarray_rs::io::{
    DecodeOptions, MfOptions, OpenOptions, ToNetcdf, WriteOptions, open_dataset_with,
};
use xarray_rs::{
    BinOp, BinOptions, ConcatOptions, DataArray, Dataset, How, InterpMethod, Join, LabelIndexer,
    MergeOptions, Name, ReduceOptions, Reduction, Repr, RollingOptions, TypedArray, Unit,
    UnitPolicy, Variable, VariableBase,
};

use crate::convert::{
    LabelSel, OrPy, PosSel, attr_from_py, attrs_to_py, label_from_py, label_sel_from_py,
    method_from_py, numpy_dtype_name, parse_unit, pos_from_py, quantity_parts, typed_from_py,
    typed_to_py, value_error,
};

/// A facade error as a Python exception.
pub(crate) fn xr_error(e: xarray_rs::Error) -> PyErr {
    match e {
        xarray_rs::Error::Core(c) => crate::convert::core_error(c),
        xarray_rs::Error::Unit(u) => crate::convert::unit_error(&u),
        xarray_rs::Error::Io(io) => pyo3::exceptions::PyOSError::new_err(io.to_string()),
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
fn variable_from_py(
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
fn spec_from_py(py: Python<'_>, name: &str, v: &Bound<'_, PyAny>) -> PyResult<Variable> {
    if let Ok(da) = v.cast::<PyDataArray>() {
        return Ok(da.get().inner.variable().to_owned());
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

fn dict_iter<'py>(d: Option<&Bound<'py, PyAny>>) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
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

fn set_attrs(attrs: &mut xarray_rs::Attrs, d: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
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
    macro_rules! view {
        ($a:expr) => {{
            // SAFETY: the data belongs to `owner`, an immutable (frozen) Python object that
            // the array keeps alive as its base; nothing ever mutates or moves the buffer,
            // and the array is made read-only below so Python cannot write through it.
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

fn new_da(py: Python<'_>, inner: DataArray) -> PyResult<Py<PyDataArray>> {
    Py::new(py, PyDataArray { inner })
}

fn new_ds(py: Python<'_>, inner: Dataset) -> PyResult<Py<PyDataset>> {
    Py::new(py, PyDataset { inner })
}

// ----- selection -----

type IselArgs<'py> = Vec<(String, Bound<'py, PyAny>)>;

fn merge_kwargs<'py>(
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
    fn take_(&self, dim: &str, idx: &[isize]) -> xarray_rs::Result<Self>;
    fn isel_(&self, ix: Vec<(String, xarray_rs::Indexer)>) -> xarray_rs::Result<Self>;
    fn sel_(&self, ix: Vec<(String, LabelIndexer)>) -> xarray_rs::Result<Self>;
    fn sel_take_(&self, dim: &str, labels: Vec<LabelIndexer>) -> xarray_rs::Result<Self>;
}

impl Select for DataArray {
    fn take_(&self, dim: &str, idx: &[isize]) -> xarray_rs::Result<Self> {
        self.take(dim, idx)
    }
    fn isel_(&self, ix: Vec<(String, xarray_rs::Indexer)>) -> xarray_rs::Result<Self> {
        Ok(self.isel(ix)?.to_owned())
    }
    fn sel_(&self, ix: Vec<(String, LabelIndexer)>) -> xarray_rs::Result<Self> {
        Ok(self.sel(ix)?.to_owned())
    }
    fn sel_take_(&self, dim: &str, labels: Vec<LabelIndexer>) -> xarray_rs::Result<Self> {
        self.sel_take(dim, labels)
    }
}

impl Select for Dataset {
    fn take_(&self, dim: &str, idx: &[isize]) -> xarray_rs::Result<Self> {
        self.take(dim, idx)
    }
    fn isel_(&self, ix: Vec<(String, xarray_rs::Indexer)>) -> xarray_rs::Result<Self> {
        Ok(self.isel(ix)?.to_owned())
    }
    fn sel_(&self, ix: Vec<(String, LabelIndexer)>) -> xarray_rs::Result<Self> {
        Ok(self.sel(ix)?.to_owned())
    }
    fn sel_take_(&self, dim: &str, labels: Vec<LabelIndexer>) -> xarray_rs::Result<Self> {
        self.sel_take(dim, labels)
    }
}

/// Positional selection: integer lists and masks first (copying those rows), then indices
/// and slices.
fn do_isel<T: Select + Clone>(py: Python<'_>, obj: &T, args: IselArgs<'_>) -> PyResult<T> {
    let mut takes = Vec::new();
    let mut basic = Vec::new();
    for (dim, v) in args {
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
        taken.as_ref().unwrap_or(obj).isel_(basic)
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
    for (dim, v) in args {
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
        taken.as_ref().unwrap_or(obj).sel_(one)
    })
    .py_err()
}

// ----- reductions -----

fn reduce_dims(dim: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Name>>> {
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
                #[pyo3(signature = (dim=None, *, skipna=None, ddof=0, min_count=None, keep_attrs=false))]
                fn $m(
                    &self,
                    py: Python<'_>,
                    dim: Option<&Bound<'_, PyAny>>,
                    skipna: Option<bool>,
                    ddof: usize,
                    min_count: Option<usize>,
                    keep_attrs: bool,
                ) -> PyResult<Py<PyAny>> {
                    let opts = reduce_opts(skipna, ddof, min_count, keep_attrs);
                    self.reduce_with(py, Reduction::$op, reduce_dims(dim)?, &opts)
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
enum Operand<'a> {
    Array(&'a DataArray),
    Owned(Box<DataArray>),
}

impl Operand<'_> {
    fn get(&self) -> &DataArray {
        match self {
            Self::Array(a) => a,
            Self::Owned(a) => a,
        }
    }
}

fn operand<'a>(py: Python<'_>, o: &'a Bound<'_, PyAny>) -> PyResult<Option<Operand<'a>>> {
    if let Ok(da) = o.cast::<PyDataArray>() {
        return Ok(Some(Operand::Array(&da.get().inner)));
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
#[pyclass(frozen, name = "DataArray", module = "rarray")]
pub(crate) struct PyDataArray {
    pub(crate) inner: DataArray,
}

impl PyDataArray {
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

    fn binary_with(
        &self,
        py: Python<'_>,
        op: BinOp,
        o: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        let Some(other) = operand(py, o)? else {
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

    fn coord_array(&self, name: &str) -> PyResult<DataArray> {
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
        units: Option<&str>,
    ) -> PyResult<Self> {
        let var = variable_from_py(py, dims, data, attrs, units)?;
        let mut inner = DataArray::new(var);
        if let Some(n) = name {
            inner = inner.with_name(n);
        }
        for (k, v) in dict_iter(coords)? {
            let c = spec_from_py(py, &k, &v)?;
            inner = inner.with_coord(k, c).py_err()?;
        }
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name().map(ToString::to_string)
    }

    /// Dimension names.
    #[getter]
    fn dims<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.dims().iter().map(Name::as_str))
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape())
    }

    /// Dimension name → length.
    #[getter]
    fn sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        sizes_dict(py, self.inner.sizes().map(|(k, n)| (k.clone(), n)))
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.shape().first().copied().unwrap_or(0)
    }

    /// The numpy dtype of `.values`.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("numpy")?
            .getattr("dtype")?
            .call1((numpy_dtype_name(self.inner.dtype()),))
    }

    /// The unit, or `None` for unitless data.
    #[getter]
    fn units(&self) -> Option<String> {
        unit_str(self.inner.variable())
    }

    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        attrs_to_py(py, self.inner.attrs())
    }

    /// Coordinates, as data arrays.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for k in self.inner.coords().keys() {
            d.set_item(k.as_str(), new_da(py, self.coord_array(k)?)?)?;
        }
        Ok(d)
    }

    /// The values as numpy: a read-only view (no copy) for numbers in memory.
    #[getter]
    fn values<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        values_of(slf.clone().into_any(), slf.get().inner.variable())
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

    /// The data converted to another unit (as float64).
    fn to(&self, py: Python<'_>, units: &str) -> PyResult<Py<PyDataArray>> {
        let u = parse_unit(units)?;
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

    /// A copy with another name.
    fn rename(&self, py: Python<'_>, name: String) -> PyResult<Py<PyDataArray>> {
        new_da(py, self.inner.clone().with_name(name))
    }

    /// Select by position: `da.isel(time=0, x=slice(1, 3))`, lists and masks too.
    #[pyo3(signature = (indexers=None, **kwargs))]
    fn isel(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        new_da(
            py,
            do_isel(py, &self.inner, merge_kwargs(indexers, kwargs)?)?,
        )
    }

    /// Select by label: `da.sel(time="2020-01", x=slice(0, 10))`; slices are inclusive.
    /// `method` is `"nearest"`, `"pad"`/`"ffill"` or `"backfill"`/`"bfill"`.
    #[pyo3(signature = (indexers=None, method=None, tolerance=None, **kwargs))]
    fn sel(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        method: Option<&str>,
        tolerance: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let args = merge_kwargs(indexers, kwargs)?;
        new_da(py, do_sel(py, &self.inner, args, method, tolerance)?)
    }

    /// Coordinates as attributes: `da.time`.
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyDataArray>> {
        if self.inner.coords().contains_key(name) {
            return new_da(py, self.coord_array(name)?);
        }
        Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "'DataArray' object has no attribute {name:?}"
        )))
    }

    /// `da[0, 1:3]` by position, `da[{"x": 0}]` by name, `da["x"]` for a coordinate.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyDataArray>> {
        if key.is_instance_of::<PyString>() {
            return new_da(py, self.coord_array(&key.extract::<String>()?)?);
        }
        if key.is_instance_of::<PyDict>() {
            return self.isel(py, Some(key), None);
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
            let out = self.inner.isel_pos(ix).py_err()?.to_owned();
            return new_da(py, out);
        }
        if items.len() > self.inner.ndim() {
            return Err(pyo3::exceptions::PyIndexError::new_err("too many indices"));
        }
        let args = self
            .inner
            .dims()
            .iter()
            .zip(items)
            .map(|(d, k)| (d.to_string(), k))
            .collect();
        new_da(py, do_isel(py, &self.inner, args)?)
    }

    fn isnull(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        new_da(py, self.inner.isnull().py_err()?)
    }

    fn notnull(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        new_da(py, self.inner.notnull().py_err()?)
    }

    /// Replace missing values (NaN) with `value`.
    fn fillna(&self, py: Python<'_>, value: f64) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.fillna(value)).py_err()?)
    }

    /// Keep values where `cond` is true; elsewhere NaN (or `other`).
    #[pyo3(signature = (cond, other=None))]
    fn r#where(
        &self,
        py: Python<'_>,
        cond: &Bound<'_, PyDataArray>,
        other: Option<f64>,
    ) -> PyResult<Py<PyDataArray>> {
        let c = &cond.get().inner;
        let out = py
            .detach(|| match other {
                None => self.inner.where_(c),
                Some(o) => self.inner.where_or(c, o),
            })
            .py_err()?;
        new_da(py, out)
    }

    /// Drop positions along `dim` with missing values (`how="any"` or `"all"`).
    #[pyo3(signature = (dim, how="any"))]
    fn dropna(&self, py: Python<'_>, dim: &str, how: &str) -> PyResult<Py<PyDataArray>> {
        new_da(py, self.inner.dropna(dim, how_from(how)?).py_err()?)
    }

    /// Conform to new labels along dimensions, filling missing ones with NaN (NaT).
    #[pyo3(signature = (indexers=None, **kwargs))]
    fn reindex(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let mut out = self.inner.clone();
        for (dim, v) in merge_kwargs(indexers, kwargs)? {
            let labels = labels_var(py, &dim, &v)?;
            out = out.reindex(&dim, &labels).py_err()?;
        }
        new_da(py, out)
    }

    /// Interpolate along dimensions: `da.interp(x=[0.5, 1.5], method="linear")`.
    #[pyo3(signature = (coords=None, method="linear", **kwargs))]
    fn interp(
        &self,
        py: Python<'_>,
        coords: Option<&Bound<'_, PyAny>>,
        method: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let m = interp_method(method)?;
        let mut out = self.inner.clone();
        for (dim, v) in merge_kwargs(coords, kwargs)? {
            let at = labels_var(py, &dim, &v)?;
            out = py.detach(|| out.interp(&dim, &at, m)).py_err()?;
        }
        new_da(py, out)
    }

    /// Group by a coordinate (or `"time.month"`, …).
    fn groupby(slf: &Bound<'_, Self>, group: String) -> PyResult<PyGroupBy> {
        slf.get().inner.groupby(&group).py_err()?;
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
        slf.get()
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
        slf.get().inner.resample(&dim, &freq).py_err()?;
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
    ) -> PyResult<PyRolling> {
        let (dim, window) = one_pair(merge_kwargs(dim, kwargs)?)?;
        let window: usize = window.extract()?;
        let opts = rolling_opts(center, min_periods);
        slf.get().inner.rolling(&dim, window, &opts).py_err()?;
        Ok(PyRolling {
            parent: Parent::Da(slf.clone().unbind()),
            dim,
            window,
            opts,
        })
    }

    /// Date and time components: `da.dt.year`, `da.dt.dayofyear`, ….
    #[getter]
    fn dt(slf: &Bound<'_, Self>) -> PyDt {
        PyDt {
            parent: slf.clone().unbind(),
        }
    }

    /// A one-variable dataset.
    #[pyo3(signature = (name=None))]
    fn to_dataset(&self, py: Python<'_>, name: Option<String>) -> PyResult<Py<PyDataset>> {
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
        let this = slf.get();
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
        self.inner.to_string().replacen("<xarray.", "<rarray.", 1)
    }
}

// ----- Dataset -----

/// A units-aware collection of labelled arrays sharing dimensions (xarray's `Dataset`).
#[pyclass(frozen, name = "Dataset", module = "rarray")]
pub(crate) struct PyDataset {
    pub(crate) inner: Dataset,
}

impl PyDataset {
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

    fn array(&self, name: &str) -> PyResult<DataArray> {
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
                    let mut a = da.get().inner.clone();
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
        Ok(Self { inner })
    }

    /// Data variables, as data arrays.
    #[getter]
    fn data_vars<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for k in self.inner.data_vars().keys() {
            d.set_item(k.as_str(), new_da(py, self.array(k)?)?)?;
        }
        Ok(d)
    }

    /// Coordinates, as data arrays.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for k in self.inner.coords().keys() {
            d.set_item(k.as_str(), new_da(py, self.array(k)?)?)?;
        }
        Ok(d)
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

    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        attrs_to_py(py, self.inner.attrs())
    }

    /// Names of the data variables.
    fn keys(&self) -> Vec<String> {
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

    /// Variables and coordinates as attributes: `ds.temp`.
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyDataArray>> {
        if self.inner.contains(name) {
            return new_da(py, self.array(name)?);
        }
        Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "'Dataset' object has no attribute {name:?}"
        )))
    }

    /// `ds["temp"]` for a variable (or coordinate); `ds[["a", "b"]]` for a subset.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(name) = key.extract::<String>() {
            return Ok(new_da(py, self.array(&name)?)?.into_any());
        }
        let names: Vec<String> = key.extract()?;
        let mut out = Dataset::new();
        for (k, c) in self.inner.coords() {
            out.add_coord(k.clone(), c.to_owned()).py_err()?;
        }
        for n in names {
            let v = self
                .inner
                .data_vars()
                .get(n.as_str())
                .ok_or_else(|| PyKeyError::new_err(n.clone()))?;
            out.add_var(n, v.to_owned()).py_err()?;
        }
        *out.attrs_mut() = self.inner.attrs().clone();
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
                Ok(da) => out.add_array(k.as_str(), da.get().inner.clone().with_name(k.as_str())),
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

    /// A copy without the named variables.
    fn drop_vars(&self, py: Python<'_>, names: &Bound<'_, PyAny>) -> PyResult<Py<PyDataset>> {
        let names: Vec<String> = match names.extract::<String>() {
            Ok(n) => vec![n],
            Err(_) => names.extract()?,
        };
        let mut out = self.inner.clone();
        for n in names {
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

    #[pyo3(signature = (indexers=None, **kwargs))]
    fn isel(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        new_ds(
            py,
            do_isel(py, &self.inner, merge_kwargs(indexers, kwargs)?)?,
        )
    }

    #[pyo3(signature = (indexers=None, method=None, tolerance=None, **kwargs))]
    fn sel(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        method: Option<&str>,
        tolerance: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let args = merge_kwargs(indexers, kwargs)?;
        new_ds(py, do_sel(py, &self.inner, args, method, tolerance)?)
    }

    fn fillna(&self, py: Python<'_>, value: f64) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.fillna(value)).py_err()?)
    }

    #[pyo3(signature = (dim, how="any"))]
    fn dropna(&self, py: Python<'_>, dim: &str, how: &str) -> PyResult<Py<PyDataset>> {
        new_ds(py, self.inner.dropna(dim, how_from(how)?).py_err()?)
    }

    #[pyo3(signature = (indexers=None, **kwargs))]
    fn reindex(
        &self,
        py: Python<'_>,
        indexers: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let mut out = self.inner.clone();
        for (dim, v) in merge_kwargs(indexers, kwargs)? {
            let labels = labels_var(py, &dim, &v)?;
            out = out.reindex(&dim, &labels).py_err()?;
        }
        new_ds(py, out)
    }

    #[pyo3(signature = (coords=None, method="linear", **kwargs))]
    fn interp(
        &self,
        py: Python<'_>,
        coords: Option<&Bound<'_, PyAny>>,
        method: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let m = interp_method(method)?;
        let mut out = self.inner.clone();
        for (dim, v) in merge_kwargs(coords, kwargs)? {
            let at = labels_var(py, &dim, &v)?;
            out = py.detach(|| out.interp(&dim, &at, m)).py_err()?;
        }
        new_ds(py, out)
    }

    fn groupby(slf: &Bound<'_, Self>, group: String) -> PyResult<PyGroupBy> {
        slf.get().inner.groupby(&group).py_err()?;
        Ok(PyGroupBy {
            parent: Parent::Ds(slf.clone().unbind()),
            split: Split::Labels(group),
        })
    }

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
        slf.get()
            .inner
            .groupby_bins(&group, &bins, &opts)
            .py_err()?;
        Ok(PyGroupBy {
            parent: Parent::Ds(slf.clone().unbind()),
            split: Split::Bins { group, bins, opts },
        })
    }

    #[pyo3(signature = (indexer=None, **kwargs))]
    fn resample(
        slf: &Bound<'_, Self>,
        indexer: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyGroupBy> {
        let (dim, freq) = one_pair(merge_kwargs(indexer, kwargs)?)?;
        let freq: String = freq.extract()?;
        slf.get().inner.resample(&dim, &freq).py_err()?;
        Ok(PyGroupBy {
            parent: Parent::Ds(slf.clone().unbind()),
            split: Split::Resample { dim, freq },
        })
    }

    #[pyo3(signature = (dim=None, center=false, min_periods=None, **kwargs))]
    fn rolling(
        slf: &Bound<'_, Self>,
        dim: Option<&Bound<'_, PyAny>>,
        center: bool,
        min_periods: Option<usize>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRolling> {
        let (dim, window) = one_pair(merge_kwargs(dim, kwargs)?)?;
        let window: usize = window.extract()?;
        let opts = rolling_opts(center, min_periods);
        slf.get().inner.rolling(&dim, window, &opts).py_err()?;
        Ok(PyRolling {
            parent: Parent::Ds(slf.clone().unbind()),
            dim,
            window,
            opts,
        })
    }

    /// Write to netCDF (`format` as in xarray; netCDF-4 by default), with the GIL released.
    #[pyo3(signature = (path, format="NETCDF4", unlimited_dims=None))]
    fn to_netcdf(
        &self,
        py: Python<'_>,
        path: PathBuf,
        format: &str,
        unlimited_dims: Option<Vec<String>>,
    ) -> PyResult<()> {
        let opts = WriteOptions::format(crate::raw::format_from_name(format)?);
        let write = |ds: &Dataset| py.detach(|| ds.to_netcdf_with(&path, &opts));
        match unlimited_dims {
            None => write(&self.inner),
            Some(u) => {
                let mut ds = self.inner.clone();
                ds.encoding_mut().unlimited_dims = u.into_iter().map(Into::into).collect();
                write(&ds)
            }
        }
        .map_err(xr_error)
    }

    /// As an `xarray.Dataset` (units as `"units"` attributes).
    fn to_xarray<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = &slf.get().inner;
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
        self.inner.to_string().replacen("<xarray.", "<rarray.", 1)
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
        other => Err(PyValueError::new_err(format!(
            "method must be \"linear\" or \"nearest\", not {other:?}"
        ))),
    }
}

fn rolling_opts(center: bool, min_periods: Option<usize>) -> RollingOptions {
    let mut o = RollingOptions::default();
    o.center = center;
    o.min_periods = min_periods;
    o
}

/// The single `dim=value` pair of `resample` and `rolling`.
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
        return Ok(da.get().inner.variable().to_owned());
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
#[pyclass(frozen, name = "GroupBy", module = "rarray")]
pub(crate) struct PyGroupBy {
    parent: Parent,
    split: Split,
}

/// Run `$body` with the group-by of the parent bound to `$g` and wrap its result.
macro_rules! with_groups {
    ($self:ident, $py:ident, |$g:ident| $body:expr) => {
        match &$self.parent {
            Parent::Da(p) => {
                let obj = &p.get().inner;
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
                let obj = &p.get().inner;
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

    fn binary_with(
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
        let obj = &p.get().inner;
        let other = &other.get().inner;
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
    fn __len__(&self) -> PyResult<usize> {
        match &self.parent {
            Parent::Da(p) => {
                let obj = &p.get().inner;
                Ok(match &self.split {
                    Split::Labels(n) => obj.groupby(n),
                    Split::Bins { group, bins, opts } => obj.groupby_bins(group, bins, opts),
                    Split::Resample { dim, freq } => obj.resample(dim, freq),
                }
                .py_err()?
                .len())
            }
            Parent::Ds(p) => {
                let obj = &p.get().inner;
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
            Parent::Da(p) => fill!(p.get().inner, new_da),
            Parent::Ds(p) => fill!(p.get().inner, new_ds),
        }
        pairs.try_iter().map(Bound::into_any)
    }

    fn __repr__(&self) -> PyResult<String> {
        let what = match &self.split {
            Split::Labels(n) => format!("groupby({n:?})"),
            Split::Bins { group, .. } => format!("groupby_bins({group:?})"),
            Split::Resample { dim, freq } => format!("resample({dim}={freq:?})"),
        };
        Ok(format!(
            "<rarray.GroupBy {what}, {} groups>",
            self.__len__()?
        ))
    }
}

// ----- rolling -----

/// Moving windows along one dimension.
#[pyclass(frozen, name = "Rolling", module = "rarray")]
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
        let r = p
            .get()
            .inner
            .rolling(&self.dim, self.window, &self.opts)
            .py_err()?;
        let out = py.detach(|| r.construct(window_dim)).py_err()?;
        Ok(new_da(py, out)?.into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "<rarray.Rolling {}={} center={}>",
            self.dim, self.window, self.opts.center
        )
    }
}

impl PyRolling {
    fn reduce(&self, py: Python<'_>, op: Reduction) -> PyResult<Py<PyAny>> {
        match &self.parent {
            Parent::Da(p) => {
                let r = p
                    .get()
                    .inner
                    .rolling(&self.dim, self.window, &self.opts)
                    .py_err()?;
                Ok(new_da(py, py.detach(|| r.reduce(op)).py_err()?)?.into_any())
            }
            Parent::Ds(p) => {
                let r = p
                    .get()
                    .inner
                    .rolling(&self.dim, self.window, &self.opts)
                    .py_err()?;
                Ok(new_ds(py, py.detach(|| r.reduce(op)).py_err()?)?.into_any())
            }
        }
    }
}

fn reduction_from(op: &str) -> PyResult<Reduction> {
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
#[pyclass(frozen, name = "DatetimeAccessor", module = "rarray")]
pub(crate) struct PyDt {
    parent: Py<PyDataArray>,
}

#[pymethods]
impl PyDt {
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyDataArray>> {
        let da = &self.parent.get().inner;
        let var = da
            .variable()
            .dt(name)
            .map_err(|e| pyo3::exceptions::PyAttributeError::new_err(e.to_string()))?;
        let mut out = DataArray::new(var);
        if let Some(n) = da.name() {
            out = out.with_name(n.clone());
        }
        for (k, c) in da.coords() {
            out = out.with_coord(k.clone(), c.to_owned()).py_err()?;
        }
        new_da(py, out)
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
fn open_options(
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

fn warn_all(py: Python<'_>, warnings: &[String]) -> PyResult<()> {
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
        Ok(pattern) => xarray_rs::io::glob(&pattern).map_err(xr_error)?,
        Err(_) => paths.extract()?,
    };
    let mut opts = MfOptions::default().open(OpenOptions::default().lazy(lazy));
    if let Some(d) = concat_dim {
        opts = opts.concat_dim(d);
    }
    let out = py
        .detach(|| xarray_rs::io::open_mfdataset(&paths, &opts))
        .map_err(xr_error)?;
    warn_all(py, &out.warnings)?;
    new_ds(py, out.dataset)
}

/// Concatenate data arrays (or datasets) along a dimension.
#[pyfunction]
pub(crate) fn concat(
    py: Python<'_>,
    objs: Vec<Bound<'_, PyAny>>,
    dim: &str,
) -> PyResult<Py<PyAny>> {
    let opts = ConcatOptions::default();
    if objs.iter().all(|o| o.is_instance_of::<PyDataArray>()) {
        let arrays: Vec<Bound<'_, PyDataArray>> = objs
            .iter()
            .map(|o| o.cast::<PyDataArray>().cloned())
            .collect::<Result<_, _>>()?;
        let refs: Vec<&DataArray> = arrays.iter().map(|a| &a.get().inner).collect();
        let out = py
            .detach(|| xarray_rs::concat(&refs, dim, &opts))
            .py_err()?;
        return Ok(new_da(py, out)?.into_any());
    }
    let sets: Vec<Bound<'_, PyDataset>> = objs
        .iter()
        .map(|o| o.cast::<PyDataset>().cloned())
        .collect::<Result<_, _>>()?;
    let refs: Vec<&Dataset> = sets.iter().map(|a| &a.get().inner).collect();
    let out = py
        .detach(|| xarray_rs::concat_datasets(&refs, dim, &opts))
        .py_err()?;
    Ok(new_ds(py, out)?.into_any())
}

/// Merge datasets (or named data arrays) into one dataset.
#[pyfunction]
#[pyo3(signature = (objs, *, join="exact"))]
pub(crate) fn merge(
    py: Python<'_>,
    objs: Vec<Bound<'_, PyAny>>,
    join: &str,
) -> PyResult<Py<PyDataset>> {
    let mut sets = Vec::with_capacity(objs.len());
    for o in &objs {
        if let Ok(da) = o.cast::<PyDataArray>() {
            let a = &da.get().inner;
            let name = a
                .name()
                .ok_or_else(|| PyValueError::new_err("unnamed DataArray cannot be merged"))?;
            let mut ds = Dataset::new();
            ds.add_array(name.clone(), a.clone()).py_err()?;
            sets.push(ds);
        } else {
            sets.push(o.cast::<PyDataset>()?.get().inner.clone());
        }
    }
    let refs: Vec<&Dataset> = sets.iter().collect();
    let mut opts = MergeOptions::default();
    opts.join = join_from(join)?;
    new_ds(py, py.detach(|| xarray_rs::merge(&refs, &opts)).py_err()?)
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
        let refs: Vec<&DataArray> = arrays.iter().map(|a| &a.get().inner).collect();
        let out = py.detach(|| xarray_rs::align(&refs, j)).py_err()?;
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
    let refs: Vec<&Dataset> = sets.iter().map(|a| &a.get().inner).collect();
    let out = py.detach(|| xarray_rs::align_datasets(&refs, j)).py_err()?;
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
    Err(PyTypeError::new_err(
        "expected an xarray DataArray or Dataset",
    ))
}
