//! Comparison and logical operators on data arrays (`<`, `==`, `&`, `~`, …), and the numpy
//! ufuncs that call them.

use annoray::{CmpOp, DataArray, LogicOp};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

use crate::convert::OrPy;
use crate::native::{Operand, PyDataArray, new_da, operand, positional_operand};

impl PyDataArray {
    /// `self op other` (or `other op self` when `reflected`); NotImplemented for operands
    /// that are not scalars, quantities, numpy arrays or data arrays.
    /// `other` as an operand beside this array, a plain array by position.
    fn operand_of<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Operand<'py>>> {
        if let Some(o) = operand(py, other)? {
            return Ok(Some(o));
        }
        Ok(
            positional_operand(py, other, self.inner.dims(), self.inner.shape())?
                .map(|a| Operand::Owned(Box::new(a))),
        )
    }

    /// A plain array operand has no name to disagree with this array's.
    fn named_like_self(&self, out: DataArray, plain: bool) -> DataArray {
        match self.inner.name() {
            Some(n) if plain => out.with_name(n.clone()),
            _ => out,
        }
    }

    pub(crate) fn compare_with(
        &self,
        py: Python<'_>,
        op: CmpOp,
        other: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        let Some(o) = self.operand_of(py, other)? else {
            return Ok(py.NotImplemented());
        };
        let plain = matches!(o, Operand::Owned(_)) && o.get().ndim() > 0;
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
        Ok(new_da(py, self.named_like_self(out, plain))?.into_any())
    }

    pub(crate) fn logical_with(
        &self,
        py: Python<'_>,
        op: LogicOp,
        other: &Bound<'_, PyAny>,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        let Some(o) = self.operand_of(py, other)? else {
            return Ok(py.NotImplemented());
        };
        let plain = matches!(o, Operand::Owned(_)) && o.get().ndim() > 0;
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
        Ok(new_da(py, self.named_like_self(out, plain))?.into_any())
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
