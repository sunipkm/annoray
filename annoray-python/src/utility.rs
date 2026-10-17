//! Conveniences of xarray's DataArray and Dataset: `copy`, `astype`, `assign_attrs`, `pipe`,
//! Python scalars (`float(da)`, `da.item()`), dataset renames and `expand_dims`, the
//! mapping methods (`items`, `values`, `get`), `.loc`, and the module functions `where`,
//! `full_like` (`zeros_like` and `ones_like` are written in Python on top of it).

use annoray::units::{Quantity, Unit, UnitKind};
use annoray::{DType, DataArray, Dataset, Name, Variable};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::convert::{OrPy, UnitError, attr_from_py, parse_unit, quantity_parts};
use crate::native::{
    PyDataArray, PyDataset, dict_iter, merge_kwargs, new_child_da, new_da, new_ds, operand,
    variable_from_py,
};
use crate::units::PyQuantity;

/// A numpy dtype (`"float32"`, `np.int16`, `int`, …) as ours.
pub(crate) fn dtype_arg(py: Python<'_>, d: &Bound<'_, PyAny>) -> PyResult<DType> {
    let name: String = py
        .import("numpy")?
        .getattr("dtype")?
        .call1((d,))?
        .getattr("name")?
        .extract()?;
    crate::raw::dtype_from_name(&name)
        .ok_or_else(|| PyTypeError::new_err(format!("dtype {name} is not supported")))
}

/// Names given as one string or a sequence of strings.
fn names_arg(names: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    match names.extract::<String>() {
        Ok(n) => Ok(vec![n]),
        Err(_) => names.extract(),
    }
}

/// `old → new` pairs from a mapping and keyword arguments.
fn pairs_arg(
    mapping: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<(String, String)>> {
    merge_kwargs(mapping, kwargs)?
        .into_iter()
        .map(|(k, v)| Ok((k, v.extract()?)))
        .collect()
}

fn as_refs(pairs: &[(String, String)]) -> Vec<(&str, &str)> {
    pairs
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect()
}

/// `func(obj, *args, **kwargs)`, or with `func` a `(callable, "keyword")` pair,
/// `callable(*args, keyword=obj, **kwargs)` (xarray's `pipe`).
fn pipe_call<'py>(
    obj: &Bound<'py, PyAny>,
    func: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    if let Ok(t) = func.cast::<PyTuple>() {
        let (f, target): (Bound<'_, PyAny>, String) = t.extract()?;
        let kw = match kwargs {
            Some(k) => k.copy()?,
            None => PyDict::new(py),
        };
        if kw.contains(&target)? {
            return Err(PyValueError::new_err(format!(
                "{target} is both the pipe target and a keyword argument"
            )));
        }
        kw.set_item(target, obj)?;
        return f.call(args, Some(&kw));
    }
    let mut all = vec![obj.clone()];
    all.extend(args.iter());
    func.call(PyTuple::new(py, all)?, kwargs)
}

/// One new dimension of `expand_dims`: its name, and either a size or labels.
struct NewDim<'py> {
    name: String,
    size: usize,
    labels: Option<Bound<'py, PyAny>>,
}

/// The dimensions of `expand_dims(dim, **dim_kwargs)`: a name, names, or a mapping to sizes
/// or labels.
fn new_dims<'py>(
    dim: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Vec<NewDim<'py>>> {
    let entry = |name: String, v: Option<Bound<'py, PyAny>>| -> PyResult<NewDim<'py>> {
        match v {
            None => Ok(NewDim {
                name,
                size: 1,
                labels: None,
            }),
            Some(v) => match v.extract::<usize>() {
                Ok(n) => Ok(NewDim {
                    name,
                    size: n,
                    labels: None,
                }),
                Err(_) => Ok(NewDim {
                    name,
                    size: v.len()?,
                    labels: Some(v),
                }),
            },
        }
    };
    let mut out = Vec::new();
    match dim {
        None => {}
        Some(d) if d.is_none() => {}
        Some(d) if d.is_instance_of::<PyDict>() => {
            for (k, v) in dict_iter(Some(d))? {
                out.push(entry(k, Some(v))?);
            }
        }
        Some(d) => {
            for n in names_arg(d)? {
                out.push(entry(n, None)?);
            }
        }
    }
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            out.push(entry(k.extract()?, Some(v))?);
        }
    }
    Ok(out)
}

/// What `expand_dims` needs of a data array or dataset.
pub(crate) trait Expand: Sized {
    fn expand(&self, dim: &str, axis: isize) -> annoray::Result<Self>;
    fn repeat(&self, dim: &str, n: usize) -> annoray::Result<Self>;
    fn labelled(self, dim: &str, coord: Variable) -> annoray::Result<Self>;
}

fn zeros(n: usize) -> Vec<isize> {
    vec![0; n]
}

impl Expand for DataArray {
    fn expand(&self, dim: &str, axis: isize) -> annoray::Result<Self> {
        self.expand_dims(dim, axis)
    }
    fn repeat(&self, dim: &str, n: usize) -> annoray::Result<Self> {
        self.take(dim, &zeros(n))
    }
    fn labelled(self, dim: &str, coord: Variable) -> annoray::Result<Self> {
        self.with_coord(dim, coord)
    }
}

impl Expand for Dataset {
    fn expand(&self, dim: &str, axis: isize) -> annoray::Result<Self> {
        self.expand_dims(dim, axis)
    }
    fn repeat(&self, dim: &str, n: usize) -> annoray::Result<Self> {
        self.take(dim, &zeros(n))
    }
    fn labelled(mut self, dim: &str, coord: Variable) -> annoray::Result<Self> {
        if self.coords().contains_key(dim) {
            self.remove(dim)?;
        }
        self.add_coord(dim, coord)?;
        Ok(self)
    }
}

/// xarray's `expand_dims`: each new dimension inserted at its `axis` (0 by default, in
/// order, so `["a", "b"]` gives `("a", "b", …)`), repeated to its size, labelled if labels
/// are given.
pub(crate) fn do_expand<T: Expand + Clone>(
    py: Python<'_>,
    obj: &T,
    dim: Option<&Bound<'_, PyAny>>,
    axis: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<T> {
    let dims = new_dims(dim, kwargs)?;
    let axes: Vec<isize> = match axis {
        None => (0..dims.len())
            .map(|i| isize::try_from(i).unwrap_or(isize::MAX))
            .collect(),
        Some(a) if a.is_none() => (0..dims.len())
            .map(|i| isize::try_from(i).unwrap_or(isize::MAX))
            .collect(),
        Some(a) => match a.extract::<isize>() {
            Ok(i) => vec![i],
            Err(_) => a.extract()?,
        },
    };
    if axes.len() != dims.len() {
        return Err(PyValueError::new_err(
            "expand_dims: give one axis per new dimension",
        ));
    }
    let mut out = obj.clone();
    for (d, ax) in dims.iter().zip(axes) {
        out = out.expand(&d.name, ax).py_err()?;
        if d.size != 1 {
            out = out.repeat(&d.name, d.size).py_err()?;
        }
        if let Some(labels) = &d.labels {
            let dims_t = PyTuple::new(py, [d.name.as_str()])?;
            let coord = variable_from_py(py, Some(&dims_t.into_any()), labels, None, None)?;
            out = out.labelled(&d.name, coord).py_err()?;
        }
    }
    Ok(out)
}

/// A scalar `other` of `where`: a number, or a quantity.
pub(crate) enum WhereOther {
    Number(f64),
    Quantity(Quantity),
}

impl WhereOther {
    /// `o` as a scalar `other`, or `None` if it is not a number or quantity.
    fn parse(o: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        if let Ok(q) = o.cast::<PyQuantity>() {
            return Ok(Some(Self::Quantity(q.get().inner.clone())));
        }
        if let Some((m, u)) = quantity_parts(o)? {
            return Ok(match m.extract::<f64>() {
                Ok(v) => Some(Self::Quantity(Quantity::new(v, parse_unit(&u)?).py_err()?)),
                Err(_) => None,
            });
        }
        if o.is_instance_of::<pyo3::types::PyBool>() {
            return Ok(None);
        }
        Ok(o.extract::<f64>().ok().map(Self::Number))
    }

    /// `o` as a scalar `other` (for a Dataset, where nothing else is accepted).
    pub(crate) fn from_py(o: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::parse(o)?.ok_or_else(|| PyTypeError::new_err("other must be a number or a quantity"))
    }

    /// The value in `unit`: a quantity converted; a bare number beside data with a unit only
    /// when it is NaN, or 0 in a ratio-scale or logarithmic unit.
    fn value_in(&self, unit: Option<&Unit>) -> annoray::Result<f64> {
        match (self, unit) {
            (Self::Quantity(q), Some(u)) => Ok(q.to(u.clone())?.value()),
            (Self::Quantity(q), None) => Ok(q.as_number()?),
            (Self::Number(v), None) => Ok(*v),
            (Self::Number(v), Some(u)) => {
                let zero_ok = matches!(u.kind(), UnitKind::Linear | UnitKind::Log(_));
                if v.is_nan() || (zero_ok && *v == 0.0) {
                    Ok(*v)
                } else {
                    Err(annoray::CoreError::Unit(
                        annoray::units::UnitError::Incompatible {
                            lhs: u.to_string(),
                            rhs: Unit::one().to_string(),
                        },
                    ))
                }
            }
        }
    }

    /// `v.where(c, self)`.
    pub(crate) fn apply(&self, v: &DataArray, c: &DataArray) -> annoray::Result<DataArray> {
        v.where_or(c, self.value_in(v.unit())?)
    }
}

/// `cond`, or `cond(obj)` when it is callable (xarray accepts both).
fn condition<'py>(
    obj: &Bound<'py, PyAny>,
    cond: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if cond.is_callable() && !cond.is_instance_of::<PyDataArray>() {
        cond.call1((obj,))
    } else {
        Ok(cond.clone())
    }
}

/// A variable of `data` replacing `like`'s values: `like`'s dimensions, unit (unless `data`
/// carries one), attributes and encoding.
fn replaced_data(py: Python<'_>, like: &Variable, data: &Bound<'_, PyAny>) -> PyResult<Variable> {
    let dims = PyTuple::new(py, like.dims().iter().map(Name::as_str))?;
    let unit = match quantity_parts(data)? {
        Some(_) => None,
        None => like.unit().map(ToString::to_string),
    };
    let mut v = variable_from_py(py, Some(&dims.into_any()), data, None, unit.as_deref())?;
    if v.shape() != like.shape() {
        return Err(PyValueError::new_err(format!(
            "data has shape {:?}; the original has {:?}",
            v.shape(),
            like.shape()
        )));
    }
    v.attrs_mut()
        .extend(like.attrs().iter().map(|(k, a)| (k.clone(), a.clone())));
    Ok(v.with_encoding(like.encoding().clone()))
}

fn attrs_update(
    attrs: &mut annoray::Attrs,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    for a in args.iter() {
        for (k, v) in dict_iter(Some(&a))? {
            attrs.insert(k.into(), attr_from_py(&v)?);
        }
    }
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            attrs.insert(k.extract::<String>()?.into(), attr_from_py(&v)?);
        }
    }
    Ok(())
}

fn loc_indexer<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    obj.py()
        .import("annoray._indexing")?
        .getattr("LocIndexer")?
        .call1((obj,))
}

fn size_one(n: usize, what: &str) -> PyResult<()> {
    if n == 1 {
        Ok(())
    } else {
        Err(PyTypeError::new_err(format!(
            "only size-1 arrays can be converted to {what}; this one has {n} elements"
        )))
    }
}

impl PyDataArray {
    /// The single value as a Python scalar, without its unit.
    fn plain_item<'py>(slf: &Bound<'py, Self>, what: &str) -> PyResult<Bound<'py, PyAny>> {
        size_one(slf.borrow().inner.len(), what)?;
        slf.getattr("values")?.call_method0("item")
    }

    /// The single value as a number (a dimensionless quantity converted: 5 % is 0.05).
    fn number(slf: &Bound<'_, Self>, what: &str) -> PyResult<f64> {
        let v = Self::plain_item(slf, what)?;
        match slf.borrow().inner.unit() {
            Some(u) => Quantity::<annoray::units::Dyn>::new(v.extract::<f64>()?, u.clone())
                .and_then(|q| q.as_number())
                .map_err(|e| {
                    UnitError::new_err(format!(
                        "{e}; only dimensionless data converts to a plain number (use .item() \
                         for a Quantity, or .values)"
                    ))
                }),
            None => v.extract(),
        }
    }
}

#[pymethods]
impl PyDataArray {
    /// A copy (annoray arrays are immutable, so deep and shallow copies are the same), with
    /// its values replaced by `data` (same shape) if given.
    #[pyo3(signature = (deep=true, data=None))]
    fn copy(
        &self,
        py: Python<'_>,
        deep: bool,
        data: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataArray>> {
        let _ = deep;
        let Some(data) = data else {
            return new_da(py, self.inner.clone());
        };
        let v = replaced_data(py, self.inner.variable(), data)?;
        let (name, _, coords) = self.inner.clone().into_parts();
        let mut out = DataArray::new(v);
        for (k, c) in coords {
            out = out.with_coord(k, c).py_err()?;
        }
        if let Some(n) = name {
            out = out.with_name(n);
        }
        new_da(py, out)
    }

    /// The values converted to `dtype`. Unlike numpy, values with no counterpart (NaN as an
    /// integer, 300 as `uint8`) are errors; floats truncate toward zero as integers.
    #[pyo3(signature = (dtype, *, keep_attrs=true))]
    fn astype(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        keep_attrs: bool,
    ) -> PyResult<Py<PyDataArray>> {
        let d = dtype_arg(py, dtype)?;
        let mut out = py.detach(|| self.inner.astype(d)).py_err()?;
        if !keep_attrs {
            out.attrs_mut().clear();
        }
        new_da(py, out)
    }

    /// A copy with attributes added or replaced: `da.assign_attrs(units_note="…")`.
    #[pyo3(signature = (*args, **kwargs))]
    fn assign_attrs(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataArray>> {
        let mut out = self.inner.clone();
        attrs_update(out.attrs_mut(), args, kwargs)?;
        new_da(py, out)
    }

    /// A copy without the named coordinates; `errors="ignore"` skips names that are not
    /// coordinates.
    #[pyo3(signature = (names, *, errors="raise"))]
    fn drop_vars(
        &self,
        py: Python<'_>,
        names: &Bound<'_, PyAny>,
        errors: &str,
    ) -> PyResult<Py<PyDataArray>> {
        let mut names = names_arg(names)?;
        if errors == "ignore" {
            names.retain(|n| self.inner.coords().contains_key(n.as_str()));
        }
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        new_da(py, self.inner.drop_coords(&refs).py_err()?)
    }

    /// `func(self, *args, **kwargs)`; `func` may be `(callable, "keyword")` to pass the
    /// array as that keyword.
    #[pyo3(signature = (func, *args, **kwargs))]
    fn pipe<'py>(
        slf: &Bound<'py, Self>,
        func: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        pipe_call(slf.as_any(), func, args, kwargs)
    }

    /// An element as a Python scalar (`da.item()` for a single value, `da.item(i)` or
    /// `da.item(i, j)` by position); with a unit, a `Quantity`.
    #[pyo3(signature = (*args))]
    fn item<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let v = slf.getattr("values")?.call_method1("item", args)?;
        match slf.borrow().inner.unit() {
            Some(u) if !v.is_instance_of::<pyo3::types::PyBool>() => match v.extract::<f64>() {
                Ok(x) => Ok(Bound::new(py, PyQuantity::new(x, u.clone())?)?.into_any()),
                Err(_) => Ok(v),
            },
            _ => Ok(v),
        }
    }

    /// The single value as a float; only dimensionless data converts (`float(5 %)` is 0.05).
    fn __float__(slf: &Bound<'_, Self>) -> PyResult<f64> {
        Self::number(slf, "Python scalars")
    }

    /// The single value as an int (floats truncated); only dimensionless data converts.
    fn __int__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let builtins = slf.py().import("builtins")?;
        if slf.borrow().inner.unit().is_some() {
            let x = Self::number(slf, "Python scalars")?;
            return builtins.getattr("int")?.call1((x,));
        }
        let v = Self::plain_item(slf, "Python scalars")?;
        builtins.getattr("int")?.call1((v,))
    }

    /// The truth of the single value; ambiguous (an error) for more than one value, and
    /// for a unit with an offset or reference (0 °C is not "nothing").
    fn __bool__(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let n = slf.borrow().inner.len();
        if n != 1 {
            return Err(PyValueError::new_err(format!(
                "the truth value of an array with {n} elements is ambiguous; use .any() or \
                 .all()"
            )));
        }
        if let Some(u) = slf.borrow().inner.unit()
            && !matches!(u.kind(), UnitKind::Linear | UnitKind::Log(_))
        {
            return Err(PyValueError::new_err(format!(
                "the truth value of data in {u} is ambiguous: its zero is not \"nothing\""
            )));
        }
        Self::plain_item(slf, "bool")?.is_truthy()
    }

    /// Label-based indexing by position: `da.loc[10.0]`, `da.loc[:, "a":"c"]`,
    /// `da.loc[{"x": 10.0}]` (read only).
    #[getter]
    fn loc<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        loc_indexer(slf.as_any())
    }
}

#[pymethods]
impl PyDataset {
    /// A copy (datasets are immutable, so deep and shallow copies are the same), with the
    /// data of variables replaced from `data` (a mapping of name to values of the same
    /// shape) if given.
    #[pyo3(signature = (deep=true, data=None))]
    fn copy(
        &self,
        py: Python<'_>,
        deep: bool,
        data: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataset>> {
        let _ = deep;
        let mut out = self.inner.clone();
        for (k, v) in dict_iter(data)? {
            let old =
                self.inner.data_vars().get(k.as_str()).ok_or_else(|| {
                    PyValueError::new_err(format!("{k:?} is not a data variable"))
                })?;
            let new = replaced_data(py, old, &v)?;
            out.remove(&k).py_err()?;
            out.add_var(k.as_str(), new).py_err()?;
        }
        new_ds(py, out)
    }

    /// Every data variable converted to `dtype` (coordinates unchanged); see
    /// `DataArray.astype`.
    #[pyo3(signature = (dtype, *, keep_attrs=true))]
    fn astype(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        keep_attrs: bool,
    ) -> PyResult<Py<PyDataset>> {
        let d = dtype_arg(py, dtype)?;
        let mut out = py.detach(|| self.inner.astype(d)).py_err()?;
        if !keep_attrs {
            out.attrs_mut().clear();
        }
        new_ds(py, out)
    }

    /// A copy with dataset attributes added or replaced.
    #[pyo3(signature = (*args, **kwargs))]
    fn assign_attrs(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let mut out = self.inner.clone();
        attrs_update(out.attrs_mut(), args, kwargs)?;
        new_ds(py, out)
    }

    /// `func(self, *args, **kwargs)`; see `DataArray.pipe`.
    #[pyo3(signature = (func, *args, **kwargs))]
    fn pipe<'py>(
        slf: &Bound<'py, Self>,
        func: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        pipe_call(slf.as_any(), func, args, kwargs)
    }

    /// Variables and dimensions renamed: `ds.rename({"t": "temp"})`, `ds.rename(x="lon")`.
    #[pyo3(signature = (name_dict=None, **names))]
    fn rename(
        &self,
        py: Python<'_>,
        name_dict: Option<&Bound<'_, PyAny>>,
        names: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let pairs = pairs_arg(name_dict, names)?;
        new_ds(py, self.inner.rename(&as_refs(&pairs)).py_err()?)
    }

    /// Variables (data variables and coordinates) renamed; dimensions kept.
    #[pyo3(signature = (name_dict=None, **names))]
    fn rename_vars(
        &self,
        py: Python<'_>,
        name_dict: Option<&Bound<'_, PyAny>>,
        names: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let pairs = pairs_arg(name_dict, names)?;
        new_ds(py, self.inner.rename_vars(&as_refs(&pairs)).py_err()?)
    }

    /// Dimensions renamed; variable names kept.
    #[pyo3(signature = (dims_dict=None, **dims))]
    fn rename_dims(
        &self,
        py: Python<'_>,
        dims_dict: Option<&Bound<'_, PyAny>>,
        dims: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let pairs = pairs_arg(dims_dict, dims)?;
        new_ds(py, self.inner.rename_dims(&as_refs(&pairs)).py_err()?)
    }

    /// New dimensions in every data variable: a name, names, or `{name: size or labels}`
    /// (also as keywords), at `axis` (0 by default).
    #[pyo3(signature = (dim=None, axis=None, **dim_kwargs))]
    fn expand_dims(
        &self,
        py: Python<'_>,
        dim: Option<&Bound<'_, PyAny>>,
        axis: Option<&Bound<'_, PyAny>>,
        dim_kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        new_ds(py, do_expand(py, &self.inner, dim, axis, dim_kwargs)?)
    }

    /// This dataset merged with `other` (a dataset, or a named data array); see `merge`.
    #[pyo3(signature = (other, *, join="exact", compat="override"))]
    fn merge<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        join: &str,
        compat: &str,
    ) -> PyResult<Py<PyDataset>> {
        crate::native::merge(
            slf.py(),
            vec![slf.clone().into_any(), other.clone()],
            join,
            compat,
        )
    }

    /// `(name, DataArray)` pairs of the data variables.
    fn items<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyList>> {
        let names = slf.borrow().keys();
        let d = PyDataset::children(slf, names)?;
        PyList::new(slf.py(), d.items())
    }

    /// The data variables, as data arrays.
    fn values<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyList>> {
        let names = slf.borrow().keys();
        let d = PyDataset::children(slf, names)?;
        PyList::new(slf.py(), d.values())
    }

    /// The variable (or coordinate) `key`, or `default` if there is none.
    #[pyo3(signature = (key, default=None))]
    fn get(slf: &Bound<'_, Self>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let found = {
            let me = slf.borrow();
            if me.inner.contains(key) {
                Some(me.array(key)?)
            } else {
                None
            }
        };
        match found {
            Some(a) => Ok(new_child_da(slf.as_any(), key, a)?.into_any()),
            None => Ok(default.unwrap_or_else(|| slf.py().None())),
        }
    }

    /// Label-based indexing with a mapping: `ds.loc[{"x": 10.0}]` (read only).
    #[getter]
    fn loc<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        loc_indexer(slf.as_any())
    }
}

/// `cond ? x : y`, elementwise (xarray's `xr.where`): `x` and `y` data arrays, quantities or
/// numbers. Operands are aligned exactly and broadcast; `y` is converted into `x`'s unit.
#[pyfunction]
#[pyo3(name = "where", signature = (cond, x, y, keep_attrs=None))]
pub(crate) fn where_(
    py: Python<'_>,
    cond: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    keep_attrs: Option<bool>,
) -> PyResult<Py<PyDataArray>> {
    let c = cond
        .cast::<PyDataArray>()
        .map_err(|_| PyTypeError::new_err("cond must be a DataArray of booleans"))?;
    let missing = || PyTypeError::new_err("x and y must be DataArrays, quantities or scalars");
    // Plain numbers take the arrays' float type (numpy's weak scalars).
    let like = [x, y].iter().find_map(|o| {
        o.cast::<PyDataArray>()
            .ok()
            .map(|d| d.borrow().inner.dtype())
    });
    let xo = crate::native::operand_like(py, x, like)?.ok_or_else(missing)?;
    let yo = crate::native::operand_like(py, y, like)?.ok_or_else(missing)?;
    let cond = &c.borrow().inner;
    let (xa, ya) = (xo.get(), yo.get());
    let mut out = py.detach(|| annoray::where_(cond, xa, ya)).py_err()?;
    if keep_attrs == Some(false) {
        out.attrs_mut().clear();
    }
    new_da(py, out)
}

/// A DataArray or Dataset shaped like `other`, filled with `fill_value` (in `other`'s unit,
/// or a quantity converted into it) as `dtype` (`other`'s by default). An integer or boolean
/// fill must be exact: numpy truncates 0.5 to 0.
#[pyfunction]
#[pyo3(signature = (other, fill_value, dtype=None))]
pub(crate) fn full_like(
    py: Python<'_>,
    other: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let dt = match dtype {
        Some(d) if !d.is_none() => Some(dtype_arg(py, d)?),
        _ => None,
    };
    if let Ok(da) = other.cast::<PyDataArray>() {
        let a = &da.borrow().inner;
        let v = match fill_value.cast::<PyQuantity>() {
            Ok(q) => match a.unit() {
                Some(u) => q.get().inner.to(u.clone()).py_err()?.value(),
                None => q.get().inner.as_number().py_err()?,
            },
            Err(_) => fill_value.extract::<f64>()?,
        };
        return Ok(new_da(py, a.full_like(v, dt).py_err()?)?.into_any());
    }
    if let Ok(ds) = other.cast::<PyDataset>() {
        if fill_value.is_instance_of::<PyQuantity>() {
            return Err(PyTypeError::new_err(
                "full_like of a Dataset takes a number (in each variable's unit)",
            ));
        }
        let v: f64 = fill_value.extract()?;
        return Ok(new_ds(py, ds.borrow().inner.full_like(v, dt).py_err()?)?.into_any());
    }
    Err(PyTypeError::new_err(
        "full_like takes a DataArray or Dataset",
    ))
}

/// `DataArray.where` with `other` a data array, quantity or number (see `scalar_in`), and
/// `cond` possibly a callable.
pub(crate) fn where_method(
    slf: &Bound<'_, PyDataArray>,
    cond: &Bound<'_, PyAny>,
    other: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyDataArray>> {
    let py = slf.py();
    let me = &slf.borrow().inner;
    let cond = condition(slf.as_any(), cond)?;
    let c = cond
        .cast::<PyDataArray>()
        .map_err(|_| PyTypeError::new_err("cond must be a DataArray of booleans"))?;
    let c = &c.borrow().inner;
    let out = match other {
        None => py.detach(|| me.where_(c)),
        Some(o) if o.is_none() => py.detach(|| me.where_(c)),
        Some(o) => match WhereOther::parse(o)? {
            Some(w) => py.detach(|| w.apply(me, c)),
            None => {
                let op = operand(py, o)?.ok_or_else(|| {
                    PyTypeError::new_err("other must be a DataArray, quantity or scalar")
                })?;
                let other = op.get();
                py.detach(|| me.where_else(c, other))
            }
        },
    };
    new_da(py, out.py_err()?)
}

/// `Dataset.where`'s condition: `cond`, or `cond(ds)` when callable.
pub(crate) fn dataset_condition<'py>(
    slf: &Bound<'py, PyDataset>,
    cond: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    condition(slf.as_any(), cond)
}
