//! Dataset operators (`+ - * / ** < == & | ^ ~ -`), numpy ufuncs on datasets, `where`,
//! `isnull`, `notnull` and `map`, variable by variable as xarray.

use annoray::{BinOp, CmpOp, DataArray, Dataset, LogicOp, MathFn, UnitPolicy};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyTuple};

use crate::convert::OrPy;
use crate::native::{PyDataArray, PyDataset, new_da, new_ds, operand};

/// The other operand of a dataset operation.
enum Other<'py> {
    Dataset(PyRef<'py, PyDataset>),
    Array(Box<DataArray>),
}

fn other<'py>(py: Python<'py>, o: &Bound<'py, PyAny>) -> PyResult<Option<Other<'py>>> {
    if let Ok(ds) = o.cast::<PyDataset>() {
        return Ok(Some(Other::Dataset(ds.borrow())));
    }
    Ok(operand(py, o)?.map(|a| Other::Array(Box::new(a.get().clone()))))
}

impl PyDataset {
    /// `self op other` (or `other op self` when `reflected`) with `f` applied to each pair of
    /// variables; NotImplemented for unsupported operands.
    fn zip_op(
        &self,
        py: Python<'_>,
        o: &Bound<'_, PyAny>,
        reflected: bool,
        f: impl Fn(&DataArray, &DataArray) -> annoray::Result<DataArray> + Send + Sync,
    ) -> PyResult<Py<PyAny>> {
        let Some(o) = other(py, o)? else {
            return Ok(py.NotImplemented());
        };
        let pair = |a: &DataArray, b: &DataArray| if reflected { f(b, a) } else { f(a, b) };
        let target: Result<&Dataset, &DataArray> = match &o {
            Other::Dataset(ds) => Ok(&ds.inner),
            Other::Array(da) => Err(da),
        };
        let out = py
            .detach(|| match target {
                Ok(ds) => self
                    .inner
                    .zip_with_dataset(ds, |a, b| pair(&a.to_owned(), &b.to_owned())),
                Err(da) => self.inner.zip_with_array(da, |v, d| pair(&v.to_owned(), d)),
            })
            .py_err()?;
        Ok(new_ds(py, out)?.into_any())
    }

    fn arith(
        &self,
        py: Python<'_>,
        op: BinOp,
        o: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        self.zip_op(py, o, reflected, move |a, b| {
            a.binary(op, b, UnitPolicy::Strict)
        })
    }

    fn cmp(
        &self,
        py: Python<'_>,
        op: CmpOp,
        o: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        self.zip_op(py, o, reflected, move |a, b| a.compare(op, b))
    }

    fn logic(
        &self,
        py: Python<'_>,
        op: LogicOp,
        o: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        self.zip_op(py, o, reflected, move |a, b| a.logical(op, b))
    }

    fn unary(&self, py: Python<'_>, f: MathFn) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.math(f)).py_err()?)
    }
}

#[pymethods]
impl PyDataset {
    /// A unit-aware function by its numpy name, variable by variable (see
    /// `DataArray._math`).
    fn _math(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyDataset>> {
        let f = MathFn::from_numpy(name)
            .ok_or_else(|| PyTypeError::new_err(format!("unknown function {name:?}")))?;
        self.unary(py, f)
    }

    fn __add__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.arith(py, BinOp::Add, o, false)
    }

    fn __radd__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.arith(py, BinOp::Add, o, true)
    }

    fn __sub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.arith(py, BinOp::Sub, o, false)
    }

    fn __rsub__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.arith(py, BinOp::Sub, o, true)
    }

    fn __mul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.arith(py, BinOp::Mul, o, false)
    }

    fn __rmul__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.arith(py, BinOp::Mul, o, true)
    }

    fn __truediv__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.arith(py, BinOp::Div, o, false)
    }

    fn __rtruediv__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.arith(py, BinOp::Div, o, true)
    }
}

fn compare_op(name: &str) -> Option<CmpOp> {
    Some(match name {
        "less" => CmpOp::Lt,
        "less_equal" => CmpOp::Le,
        "greater" => CmpOp::Gt,
        "greater_equal" => CmpOp::Ge,
        "equal" => CmpOp::Eq,
        "not_equal" => CmpOp::Ne,
        _ => return None,
    })
}

fn binary_op(name: &str) -> Option<BinOp> {
    Some(match name {
        "add" => BinOp::Add,
        "subtract" => BinOp::Sub,
        "multiply" => BinOp::Mul,
        "true_divide" | "divide" => BinOp::Div,
        "maximum" => BinOp::Max,
        "minimum" => BinOp::Min,
        _ => return None,
    })
}

fn logic_op(name: &str) -> Option<LogicOp> {
    Some(match name {
        "logical_and" | "bitwise_and" => LogicOp::And,
        "logical_or" | "bitwise_or" => LogicOp::Or,
        "logical_xor" | "bitwise_xor" => LogicOp::Xor,
        _ => return None,
    })
}

#[pymethods]
impl PyDataset {
    /// Elementwise comparisons of every data variable (`ds > 0`), with a number, a quantity,
    /// a DataArray or another Dataset (its common variables).
    fn __richcmp__(
        &self,
        py: Python<'_>,
        o: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let op = match op {
            CompareOp::Lt => CmpOp::Lt,
            CompareOp::Le => CmpOp::Le,
            CompareOp::Gt => CmpOp::Gt,
            CompareOp::Ge => CmpOp::Ge,
            CompareOp::Eq => CmpOp::Eq,
            CompareOp::Ne => CmpOp::Ne,
        };
        self.cmp(py, op, o, false)
    }

    fn __and__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logic(py, LogicOp::And, o, false)
    }

    fn __rand__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logic(py, LogicOp::And, o, true)
    }

    fn __or__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logic(py, LogicOp::Or, o, false)
    }

    fn __ror__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logic(py, LogicOp::Or, o, true)
    }

    fn __xor__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logic(py, LogicOp::Xor, o, false)
    }

    fn __rxor__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logic(py, LogicOp::Xor, o, true)
    }

    fn __invert__(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.logical_not()).py_err()?)
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        self.unary(py, MathFn::Neg)
    }

    fn __pos__(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        new_ds(py, self.inner.clone())
    }

    fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        self.unary(py, MathFn::Abs)
    }

    /// `ds ** e` for a number `e` (units raised, or rooted for `1/n`).
    fn __pow__(
        &self,
        py: Python<'_>,
        e: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let Ok(e) = e.extract::<f64>() else {
            return Ok(py.NotImplemented());
        };
        Ok(new_ds(py, py.detach(|| self.inner.pow(e)).py_err()?)?.into_any())
    }

    /// numpy ufuncs, variable by variable (the unit-aware set of `DataArray`).
    #[pyo3(signature = (ufunc, method, *inputs, **kwargs))]
    fn __array_ufunc__(
        slf: &Bound<'_, Self>,
        ufunc: &Bound<'_, PyAny>,
        method: &str,
        inputs: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if method != "__call__" || kwargs.is_some_and(|k| !k.is_empty()) {
            return Ok(py.NotImplemented());
        }
        let name: String = ufunc.getattr("__name__")?.extract()?;
        let me_ref = slf.borrow();
        let me = &*me_ref;
        match inputs.len() {
            1 => {
                if matches!(name.as_str(), "logical_not" | "invert") {
                    return Ok(
                        new_ds(py, py.detach(|| me.inner.logical_not()).py_err()?)?.into_any()
                    );
                }
                if name == "positive" {
                    return Ok(new_ds(py, me.inner.clone())?.into_any());
                }
                if let Some(f) = MathFn::from_numpy(&name) {
                    return Ok(me.unary(py, f)?.into_any());
                }
            }
            2 => {
                let first_is_me = inputs.get_item(0)?.is(slf);
                let o = inputs.get_item(if first_is_me { 1 } else { 0 })?;
                if let Some(op) = binary_op(&name) {
                    return me.arith(py, op, &o, !first_is_me);
                }
                if let Some(op) = compare_op(&name) {
                    return me.cmp(py, op, &o, !first_is_me);
                }
                if let Some(op) = logic_op(&name) {
                    return me.logic(py, op, &o, !first_is_me);
                }
            }
            _ => {}
        }
        Err(PyTypeError::new_err(format!(
            "numpy.{name} is not supported on annoray Datasets; apply it per variable with Dataset.map"
        )))
    }

    /// Keep values where `cond` (a boolean DataArray, a Dataset of booleans for the common
    /// variables, or a callable taking this dataset) holds; NaN (NaT) or `other` elsewhere.
    /// `other` is a number or a quantity, converted into each variable's unit (a bare number
    /// beside a unit only when NaN, or 0 in a ratio-scale unit).
    #[pyo3(signature = (cond, other=None))]
    fn r#where(
        slf: &Bound<'_, Self>,
        cond: &Bound<'_, PyAny>,
        other: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDataset>> {
        let py = slf.py();
        let this = &slf.borrow().inner;
        let cond = &crate::utility::dataset_condition(slf, cond)?;
        let other = match other {
            Some(o) if !o.is_none() => Some(crate::utility::WhereOther::from_py(o)?),
            _ => None,
        };
        let pick = |v: &DataArray, c: &DataArray| match &other {
            None => v.where_(c),
            Some(o) => o.apply(v, c),
        };
        let out = if let Ok(c) = cond.cast::<PyDataset>() {
            let c = &c.borrow().inner;
            py.detach(|| this.zip_with_dataset(c, |v, c| pick(&v.to_owned(), &c.to_owned())))
        } else if let Ok(c) = cond.cast::<PyDataArray>() {
            let c = &c.borrow().inner;
            py.detach(|| this.zip_with_array(c, |v, c| pick(&v.to_owned(), c)))
        } else {
            return Err(PyTypeError::new_err(
                "cond must be a DataArray or Dataset of booleans",
            ));
        };
        new_ds(py, out.py_err()?)
    }

    /// Where each data variable is missing (NaN/NaT).
    fn isnull(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.isnull()).py_err()?)
    }

    /// Where each data variable is present.
    fn notnull(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        new_ds(py, py.detach(|| self.inner.notnull()).py_err()?)
    }

    /// `func(variable, *args, **kwargs)` for every data variable (a DataArray in, a DataArray
    /// out), as xarray's `Dataset.map`; attributes kept unless `keep_attrs=False`.
    #[pyo3(signature = (func, keep_attrs=None, args=None, **kwargs))]
    fn map(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        keep_attrs: Option<bool>,
        args: Option<&Bound<'_, PyTuple>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyDataset>> {
        let empty = PyTuple::empty(py);
        let args = args.unwrap_or(&empty);
        let err: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
        let out = self.inner.map_data_vars(keep_attrs.unwrap_or(true), |v| {
            let call = || -> PyResult<DataArray> {
                let arg = new_da(py, v.to_owned())?;
                let mut all = vec![arg.into_any()];
                all.extend(args.iter().map(Bound::unbind));
                let r = func.call(PyTuple::new(py, all)?, kwargs)?;
                Ok(r.cast::<PyDataArray>()
                    .map_err(|_| {
                        PyTypeError::new_err("Dataset.map: func must return an annoray DataArray")
                    })?
                    .borrow()
                    .inner
                    .clone())
            };
            call().map_err(|e| {
                let msg = e.to_string();
                *err.borrow_mut() = Some(e);
                annoray::CoreError::InvalidReduction(msg)
            })
        });
        if let Some(e) = err.into_inner() {
            return Err(e);
        }
        new_ds(py, out.py_err()?)
    }
}
