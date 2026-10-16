//! Comparison and logical operators on data arrays (`<`, `==`, `&`, `~`, …), and the numpy
//! ufuncs that call them.

use annoray::{CmpOp, LogicOp};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

use crate::convert::OrPy;
use crate::native::{PyDataArray, new_da, operand};

impl PyDataArray {
    /// `self op other` (or `other op self` when `reflected`); NotImplemented for operands
    /// that are not scalars, quantities or data arrays.
    pub(crate) fn compare_with(
        &self,
        py: Python<'_>,
        op: CmpOp,
        other: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        let Some(o) = operand(py, other)? else {
            return Ok(py.NotImplemented());
        };
        let o = o.get();
        let out = py
            .detach(|| {
                if reflected {
                    o.compare(op, &self.inner)
                } else {
                    self.inner.compare(op, o)
                }
            })
            .py_err()?;
        Ok(new_da(py, out)?.into_any())
    }

    pub(crate) fn logical_with(
        &self,
        py: Python<'_>,
        op: LogicOp,
        other: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        let Some(o) = operand(py, other)? else {
            return Ok(py.NotImplemented());
        };
        let o = o.get();
        let out = py
            .detach(|| {
                if reflected {
                    o.logical(op, &self.inner)
                } else {
                    self.inner.logical(op, o)
                }
            })
            .py_err()?;
        Ok(new_da(py, out)?.into_any())
    }
}

#[pymethods]
impl PyDataArray {
    /// Elementwise `<`, `<=`, `>`, `>=`, `==`, `!=` (booleans), as xarray. Numbers compare
    /// in this array's unit (quantities are converted; a bare number only if it is 0 and the
    /// unit is ratio-scale); NaN compares false (true for `!=`).
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
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
        self.compare_with(py, op, other, false)
    }

    fn __and__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logical_with(py, LogicOp::And, other, false)
    }

    fn __rand__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logical_with(py, LogicOp::And, other, true)
    }

    fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logical_with(py, LogicOp::Or, other, false)
    }

    fn __ror__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logical_with(py, LogicOp::Or, other, true)
    }

    fn __xor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logical_with(py, LogicOp::Xor, other, false)
    }

    fn __rxor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.logical_with(py, LogicOp::Xor, other, true)
    }

    /// Logical not of booleans (`~mask`).
    fn __invert__(&self, py: Python<'_>) -> PyResult<Py<PyDataArray>> {
        new_da(py, py.detach(|| self.inner.logical_not()).py_err()?)
    }
}
