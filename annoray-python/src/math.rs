//! Unit-aware numpy ufuncs on data arrays (`__array_ufunc__`) and the unary operators.

use annoray::{BinOp, DataArray, MathFn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::convert::OrPy;
use crate::native::{PyDataArray, new_da, variable_from_py};

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

#[pymethods]
impl PyDataArray {
    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.math(MathFn::Neg)).py_err()?)
    }

    fn __pos__(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        new_da(py, self.inner.clone())
    }

    fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.math(MathFn::Abs)).py_err()?)
    }

    /// `da ** e` for a number `e`: the unit is raised (or rooted for `1/n`).
    fn __pow__(
        &self,
        py: Python<'_>,
        e: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let Ok(e) = e.extract::<f64>() else {
            return Ok(py.NotImplemented());
        };
        Ok(new_da(py, py.detach(|| self.inner.pow(e)).py_err()?)?.into_any())
    }

    /// numpy ufuncs: the unit-aware ones (`np.sqrt`, `np.sin`, `np.add`, `np.maximum`, …)
    /// keep labels and units; others apply to unitless data and refuse data with a unit.
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
        let me = slf.get();
        match inputs.len() {
            1 => {
                if name == "positive" {
                    return Ok(new_da(py, me.inner.clone())?.into_any());
                }
                if let Some(f) = MathFn::from_numpy(&name) {
                    let out = py.detach(|| me.inner.math(f)).py_err()?;
                    return Ok(new_da(py, out)?.into_any());
                }
            }
            2 => {
                let first_is_me = inputs.get_item(0)?.is(slf);
                let other = inputs.get_item(if first_is_me { 1 } else { 0 })?;
                if name == "power" && first_is_me {
                    return Self::__pow__(me, py, &other, None);
                }
                if let Some(op) = binary_op(&name) {
                    return me.binary_with(py, op, &other, !first_is_me);
                }
            }
            _ => {}
        }
        // Not unit-aware: only for data without a unit, and one array among the inputs.
        let arrays: Vec<Bound<'_, PyDataArray>> = inputs
            .iter()
            .filter_map(|x| x.cast::<PyDataArray>().ok().cloned())
            .collect();
        if arrays.iter().any(|a| a.get().inner.unit().is_some()) {
            return Err(PyTypeError::new_err(format!(
                "numpy.{name} is not unit-aware; apply it to .values or convert to plain numbers first"
            )));
        }
        if arrays.len() != 1 {
            return Ok(py.NotImplemented());
        }
        let raw: Vec<Bound<'_, PyAny>> = inputs
            .iter()
            .map(|x| match x.cast::<PyDataArray>() {
                Ok(a) => a.getattr("values"),
                Err(_) => Ok(x),
            })
            .collect::<PyResult<_>>()?;
        let result = ufunc.call1(PyTuple::new(py, raw)?)?;
        let shape: Vec<usize> = result.getattr("shape")?.extract()?;
        if shape != me.inner.variable().shape() {
            return Ok(py.NotImplemented());
        }
        let dims = PyTuple::new(py, me.inner.dims().iter().map(|d| d.as_str()))?;
        let var = variable_from_py(py, Some(dims.as_any()), &result, None, None)?;
        let mut out = DataArray::new(var);
        for (k, c) in me.inner.coords() {
            out = out.with_coord(k.clone(), c.clone()).py_err()?;
        }
        if let Some(n) = me.inner.name() {
            out = out.with_name(n.clone());
        }
        Ok(new_da(py, out)?.into_any())
    }
}
