//! `annoray._annoray`'s side of the exchange format other extension modules use (the
//! `annoray-pyo3` crate): the same code as theirs, on annoray's own objects.
//!
//! Values cross as numpy arrays: from a data array or dataset they are lent (read-only views
//! of annoray's data, no copy) where they can be; into annoray they are copied once.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use annoray::Variable;
use annoray_pyo3::parts;

use crate::native::{PyDataArray, PyDataset, new_da, new_ds, values_of};
use crate::tree::{PyDataTree, new_root};

/// Views of `owner`'s values (in memory, numeric, date-time or duration), without a copy.
fn lend<'py>(owner: &Bound<'py, PyAny>, var: &Variable) -> PyResult<Option<Bound<'py, PyAny>>> {
    if var.is_lazy() || !parts::lends(var.dtype()) {
        return Ok(None);
    }
    values_of(owner.clone(), var).map(Some)
}

/// The exchange format's version (see `annoray-pyo3`).
#[pyfunction]
pub(crate) fn _interop_protocol() -> u32 {
    annoray_pyo3::PROTOCOL
}

/// A `DataArray` from its parts (values copied once).
#[pyfunction]
pub(crate) fn _array_from_parts(
    py: Python<'_>,
    parts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDataArray>> {
    new_da(py, parts::dataarray_from_parts(parts)?)
}

/// A `DataArray`'s parts (values lent where they can be).
#[pyfunction]
pub(crate) fn _array_to_parts<'py>(da: &Bound<'py, PyDataArray>) -> PyResult<Bound<'py, PyDict>> {
    let owner = da.as_any();
    parts::dataarray_to_parts(da.py(), &da.borrow().inner, &|v| lend(owner, v))
}

/// A `Dataset` from its parts (values copied once).
#[pyfunction]
pub(crate) fn _dataset_from_parts(
    py: Python<'_>,
    parts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDataset>> {
    new_ds(py, parts::dataset_from_parts(parts)?)
}

/// A `Dataset`'s parts (values lent where they can be).
#[pyfunction]
pub(crate) fn _dataset_to_parts<'py>(ds: &Bound<'py, PyDataset>) -> PyResult<Bound<'py, PyDict>> {
    let owner = ds.as_any();
    parts::dataset_to_parts(ds.py(), &ds.borrow().inner, &|v| lend(owner, v))
}

/// A `DataTree` from its parts (values copied once).
#[pyfunction]
pub(crate) fn _tree_from_parts(
    py: Python<'_>,
    parts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDataTree>> {
    let (tree, name) = parts::tree_from_parts(parts)?;
    new_root(py, tree, name)
}

/// The parts of the subtree at a `DataTree` node (its values copied: trees do not lend).
#[pyfunction]
pub(crate) fn _tree_to_parts<'py>(node: &Bound<'py, PyDataTree>) -> PyResult<Bound<'py, PyDict>> {
    let py = node.py();
    let name: Option<String> = node.getattr("name")?.extract()?;
    let tree = node.borrow().subtree_copy(py)?;
    parts::tree_into_parts(py, tree, name.as_deref())
}
