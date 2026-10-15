//! Native `DataTree`: a tree of datasets, as netCDF-4 groups are (xarray's `DataTree`).

use std::path::PathBuf;

use annoray::DataTree;
use annoray::io::{ToNetcdf, WriteOptions, open_datatree_with};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::OrPy;
use crate::native::{PyDataset, new_dataset, open_options, warn_all, xr_error};

/// A tree of datasets: each node has its own variables and named children, and inherits
/// its ancestors' coordinates (xarray's `DataTree`).
#[pyclass(frozen, name = "DataTree", module = "annoray")]
pub(crate) struct PyDataTree {
    pub(crate) inner: DataTree,
}

fn new_tree(py: Python<'_>, inner: DataTree) -> PyResult<Py<PyDataTree>> {
    Py::new(py, PyDataTree { inner })
}

/// `/a/b`, `a/b/` and `a//b` all as `/a/b`; the root as `/`.
fn normalize(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    format!("/{}", parts.join("/"))
}

/// The tree at `prefix` from `nodes` (normalized path → dataset); missing nodes are empty.
fn build(prefix: &str, nodes: &[(String, annoray::Dataset)]) -> PyResult<DataTree> {
    let own = nodes
        .iter()
        .find(|(p, _)| p == prefix)
        .map(|(_, d)| d.clone())
        .unwrap_or_default();
    let mut tree = DataTree::new(own);
    let base = if prefix == "/" {
        "/".to_owned()
    } else {
        format!("{prefix}/")
    };
    let mut children: Vec<&str> = Vec::new();
    for (p, _) in nodes {
        if let Some(rest) = p.strip_prefix(&base)
            && let Some(first) = rest.split('/').next().filter(|s| !s.is_empty())
            && !children.contains(&first)
        {
            children.push(first);
        }
    }
    for c in children {
        let child = build(&format!("{base}{c}"), nodes)?;
        tree.add_child(c, child).py_err()?;
    }
    Ok(tree)
}

#[pymethods]
impl PyDataTree {
    /// `DataTree(dataset=None)`: a single node.
    #[new]
    #[pyo3(signature = (dataset=None))]
    fn py_new(dataset: Option<&Bound<'_, PyDataset>>) -> Self {
        Self {
            inner: DataTree::new(dataset.map(|d| d.get().inner.clone()).unwrap_or_default()),
        }
    }

    /// A tree from `{path: Dataset}` (`"/"` for the root; missing parents are empty).
    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut nodes = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            let path = normalize(&k.extract::<String>()?);
            nodes.push((path, v.cast::<PyDataset>()?.get().inner.clone()));
        }
        Ok(Self {
            inner: build("/", &nodes)?,
        })
    }

    /// Every node's path, depth first (`"/"` first).
    #[getter]
    fn paths(&self) -> Vec<String> {
        self.inner.paths()
    }

    /// This node's own variables (without inherited coordinates).
    #[getter]
    fn dataset(&self, py: Python<'_>) -> PyResult<Py<PyDataset>> {
        new_dataset(py, self.inner.dataset().clone())
    }

    /// Child name → subtree.
    #[getter]
    fn children<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, c) in self.inner.children() {
            d.set_item(k.as_str(), new_tree(py, c.clone())?)?;
        }
        Ok(d)
    }

    /// The subtree at `path` (`"a/b"` or `"/a/b"`).
    fn __getitem__(&self, py: Python<'_>, path: &str) -> PyResult<Py<PyDataTree>> {
        let node = self
            .inner
            .get(path)
            .map_err(|_| PyKeyError::new_err(path.to_owned()))?;
        new_tree(py, node.clone())
    }

    fn __contains__(&self, path: &str) -> bool {
        self.inner.get(path).is_ok()
    }

    /// The node at `path` as a dataset, with its ancestors' coordinates when `inherit`.
    #[pyo3(signature = (path="/", inherit=true))]
    fn to_dataset(&self, py: Python<'_>, path: &str, inherit: bool) -> PyResult<Py<PyDataset>> {
        let ds = if inherit {
            self.inner.inherited(path).py_err()?
        } else {
            self.inner.get(path).py_err()?.dataset().clone()
        };
        new_dataset(py, ds)
    }

    /// A copy with every node's values read into memory.
    fn load(&self, py: Python<'_>) -> PyResult<Py<PyDataTree>> {
        let mut t = self.inner.clone();
        py.detach(|| t.load()).py_err()?;
        new_tree(py, t)
    }

    /// Write as netCDF-4 groups (lazy values stream chunk by chunk).
    #[pyo3(signature = (path, format="NETCDF4"))]
    fn to_netcdf(&self, py: Python<'_>, path: PathBuf, format: &str) -> PyResult<()> {
        let opts = WriteOptions::format(crate::raw::format_from_name(format)?);
        py.detach(|| self.inner.to_netcdf_with(&path, &opts))
            .map_err(xr_error)
    }

    /// As an `xarray.DataTree` (needs xarray).
    fn to_xarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let xr = py.import("xarray")?;
        let d = PyDict::new(py);
        for path in self.inner.paths() {
            let node = self.inner.get(&path).py_err()?;
            let ds = Bound::new(
                py,
                PyDataset {
                    inner: node.dataset().clone(),
                },
            )?;
            d.set_item(path, ds.call_method0("to_xarray")?)?;
        }
        xr.getattr("DataTree")?.call_method1("from_dict", (d,))
    }

    fn __len__(&self) -> usize {
        self.inner.children().len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let names: Vec<&str> = self.inner.children().keys().map(|k| k.as_str()).collect();
        PyList::new(py, names)?.try_iter().map(Bound::into_any)
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

/// Open a netCDF-4/HDF5 file's groups as a `DataTree` (lazily: values are read on demand).
#[pyfunction]
#[pyo3(signature = (path, *, mask_and_scale=true, decode_times=true, decode_units=true,
    decode_coords=true, concat_characters=true, lazy=true))]
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
pub(crate) fn open_datatree(
    py: Python<'_>,
    path: PathBuf,
    mask_and_scale: bool,
    decode_times: bool,
    decode_units: bool,
    decode_coords: bool,
    concat_characters: bool,
    lazy: bool,
) -> PyResult<Py<PyDataTree>> {
    let opts = open_options(
        None,
        mask_and_scale,
        decode_times,
        decode_units,
        decode_coords,
        concat_characters,
        lazy,
    );
    let opened = py
        .detach(|| open_datatree_with(&path, &opts))
        .map_err(xr_error)?;
    warn_all(py, &opened.warnings)?;
    new_tree(py, opened.tree)
}

/// An `xarray.DataTree` as an annoray `DataTree`.
pub(crate) fn tree_from_xarray(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<PyDataTree> {
    let mut nodes = Vec::new();
    for node in obj.getattr("subtree")?.try_iter()? {
        let node = node?;
        let path: String = node.getattr("path")?.extract()?;
        let kw = PyDict::new(py);
        kw.set_item("inherit", false)?;
        let ds = node.getattr("to_dataset")?.call((), Some(&kw))?;
        let ours = crate::native::from_xarray(py, &ds)?;
        nodes.push((
            normalize(&path),
            ours.bind(py).cast::<PyDataset>()?.get().inner.clone(),
        ));
    }
    Ok(PyDataTree {
        inner: build("/", &nodes)?,
    })
}
