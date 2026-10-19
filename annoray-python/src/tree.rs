//! Native `DataTree`: a tree of datasets, as netCDF-4 groups are (xarray's `DataTree`).
//!
//! A Python `DataTree` is a node: a shared root ([`TreeRoot`], holding the whole tree) and
//! a path in it. Nodes taken from a tree (`tree["a/b"]`, `.children`, `.parent`, `.root`)
//! are views of that same tree, as in xarray: they know their parent and root, see the
//! coordinates they inherit, and edits through any of them change the tree. Edits are
//! checked against the inherited coordinates before anything changes. The xarray API on top
//! (traversal, computation, comparison) is in `annoray/_tree.py`.

use std::path::PathBuf;

use annoray::io::{ToNetcdf, WriteOptions, open_datatree_with};
use annoray::{DataTree, Dataset};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::OrPy;
use crate::native::{PyDataset, new_dataset, open_options, warn_all, xr_error};

/// The tree a family of `DataTree` nodes shares.
#[pyclass(module = "annoray._annoray")]
pub(crate) struct TreeRoot {
    pub(crate) inner: DataTree,
    /// The root's name (`None` unless given).
    name: Option<String>,
}

/// A node of a tree of datasets: its own variables and named children, inheriting its
/// ancestors' dimension coordinates (xarray's `DataTree`).
#[pyclass(name = "DataTree", module = "annoray", weakref)]
pub(crate) struct PyDataTree {
    root: Py<TreeRoot>,
    /// `"/"` or `"/a/b"`.
    path: String,
}

/// `/a/b`, `a/b/` and `a//b` all as `/a/b`; the root as `/`.
fn normalize(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    format!("/{}", parts.join("/"))
}

/// `child` under `parent` (both normalized).
fn join(parent: &str, child: &str) -> String {
    normalize(&format!("{parent}/{child}"))
}

/// The tree at `prefix` from `nodes` (normalized path → dataset); missing nodes are empty.
fn build(prefix: &str, nodes: &[(String, Dataset)]) -> PyResult<DataTree> {
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

/// A new tree (its own root) holding `inner`.
pub(crate) fn new_root(
    py: Python<'_>,
    inner: DataTree,
    name: Option<String>,
) -> PyResult<Py<PyDataTree>> {
    let root = Py::new(py, TreeRoot { inner, name })?;
    Py::new(
        py,
        PyDataTree {
            root,
            path: "/".to_owned(),
        },
    )
}

impl PyDataTree {
    fn view(&self, py: Python<'_>, path: String) -> PyResult<Py<PyDataTree>> {
        Py::new(
            py,
            PyDataTree {
                root: self.root.clone_ref(py),
                path,
            },
        )
    }

    /// This node's subtree, copied.
    fn node_tree(&self, py: Python<'_>) -> PyResult<DataTree> {
        let root = self.root.borrow(py);
        Ok(root
            .inner
            .get(&self.path)
            .map_err(|_| removed(&self.path))?
            .clone())
    }

    /// This node's subtree, copied (for writers).
    pub(crate) fn subtree_copy(&self, py: Python<'_>) -> PyResult<DataTree> {
        self.node_tree(py)
    }

    /// `(parent path, name)` of this node; `None` at the root.
    fn split_self(&self) -> Option<(String, String)> {
        let (parent, name) = self.path.rsplit_once('/')?;
        if name.is_empty() {
            return None;
        }
        Some((normalize(parent), name.to_owned()))
    }
}

fn removed(path: &str) -> PyErr {
    PyKeyError::new_err(format!(
        "node {path:?} is not in its tree (it was removed or replaced); take a copy() first"
    ))
}

#[pymethods]
impl PyDataTree {
    /// `DataTree(dataset=None, children=None, name=None)`: a root node with `dataset`'s
    /// variables and copies of the `children` trees.
    #[new]
    #[pyo3(signature = (dataset=None, children=None, name=None))]
    fn py_new(
        py: Python<'_>,
        dataset: Option<&Bound<'_, PyDataset>>,
        children: Option<&Bound<'_, PyDict>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let mut inner = DataTree::new(
            dataset
                .map(|d| d.borrow().inner.clone())
                .unwrap_or_default(),
        );
        if let Some(ch) = children {
            for (k, v) in ch.iter() {
                let sub = v.cast::<PyDataTree>()?.borrow().node_tree(py)?;
                inner.add_child(k.extract::<String>()?, sub).py_err()?;
            }
        }
        Ok(Self {
            root: Py::new(py, TreeRoot { inner, name })?,
            path: "/".to_owned(),
        })
    }

    /// A tree from `{path: Dataset}` (`"/"` for the root; missing parents are empty).
    #[staticmethod]
    #[pyo3(signature = (d, name=None))]
    fn from_dict(
        py: Python<'_>,
        d: &Bound<'_, PyDict>,
        name: Option<String>,
    ) -> PyResult<Py<PyDataTree>> {
        let mut nodes = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            let path = normalize(&k.extract::<String>()?);
            let ds = if v.is_none() {
                Dataset::new()
            } else {
                v.cast::<PyDataset>()?.borrow().inner.clone()
            };
            nodes.push((path, ds));
        }
        new_root(py, build("/", &nodes)?, name)
    }

    /// This node's path from the root (`"/"`, `"/a/b"`).
    #[getter]
    fn path(&self) -> String {
        self.path.clone()
    }

    /// This node's name (the root's is `None` unless given).
    #[getter]
    fn name(&self, py: Python<'_>) -> Option<String> {
        match self.split_self() {
            Some((_, n)) => Some(n),
            None => self.root.borrow(py).name.clone(),
        }
    }

    #[setter]
    fn set_name(&self, py: Python<'_>, name: Option<String>) -> PyResult<()> {
        if self.split_self().is_some() {
            return Err(PyValueError::new_err(
                "only a root's name can be set; a child's name is its key in its parent",
            ));
        }
        self.root.borrow_mut(py).name = name;
        Ok(())
    }

    /// The parent node (`None` at the root).
    #[getter]
    fn parent(&self, py: Python<'_>) -> PyResult<Option<Py<PyDataTree>>> {
        self.split_self().map(|(p, _)| self.view(py, p)).transpose()
    }

    /// The root of the tree this node is in.
    #[getter]
    fn root(&self, py: Python<'_>) -> PyResult<Py<PyDataTree>> {
        self.view(py, "/".to_owned())
    }

    /// Whether this node is its tree's root.
    #[getter]
    fn is_root(&self) -> bool {
        self.path == "/"
    }

    /// The names of the children, in order.
    fn _child_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let root = self.root.borrow(py);
        Ok(root
            .inner
            .get(&self.path)
            .map_err(|_| removed(&self.path))?
            .children()
            .keys()
            .map(ToString::to_string)
            .collect())
    }

    /// The node at `path` below this one, as a view (`None` if there is none).
    fn _node(&self, py: Python<'_>, path: &str) -> PyResult<Option<Py<PyDataTree>>> {
        let full = if path.starts_with('/') {
            normalize(path)
        } else {
            join(&self.path, path)
        };
        let exists = self.root.borrow(py).inner.get(&full).is_ok();
        if exists {
            Ok(Some(self.view(py, full)?))
        } else {
            Ok(None)
        }
    }

    /// Whether `other` is a node of the same tree.
    fn _same_root(&self, other: &Bound<'_, PyDataTree>) -> bool {
        self.root.is(&other.borrow().root)
    }

    /// This node's dataset: with the coordinates it inherits (`inherit`), or its own. A path
    /// as the first argument (annoray's earlier form, `to_dataset("/a/b")`) names a node
    /// below this one instead.
    #[pyo3(signature = (path_or_inherit=None, inherit=true))]
    fn to_dataset(
        &self,
        py: Python<'_>,
        path_or_inherit: Option<&Bound<'_, PyAny>>,
        inherit: bool,
    ) -> PyResult<Py<PyDataset>> {
        let (path, inherit) = match path_or_inherit {
            None => (self.path.clone(), inherit),
            Some(p) if p.is_instance_of::<pyo3::types::PyBool>() => {
                (self.path.clone(), p.extract()?)
            }
            Some(p) => {
                let rel: String = p.extract()?;
                // As `tree[path]`: absolute from the root, or relative to this node.
                let full = if rel.starts_with('/') {
                    normalize(&rel)
                } else {
                    join(&self.path, &rel)
                };
                (full, inherit)
            }
        };
        let root = self.root.borrow(py);
        let ds = if inherit {
            root.inner.inherited(&path).map_err(|_| removed(&path))?
        } else {
            root.inner
                .get(&path)
                .map_err(|_| removed(&path))?
                .dataset()
                .clone()
        };
        new_dataset(py, ds)
    }

    /// Replace this node's own dataset (checked against what it inherits and its
    /// descendants).
    fn _set_dataset(&self, py: Python<'_>, ds: &Bound<'_, PyDataset>) -> PyResult<()> {
        let new = ds.borrow().inner.clone();
        self.root
            .borrow_mut(py)
            .inner
            .set_dataset_at(&self.path, new)
            .py_err()?;
        Ok(())
    }

    /// Set child `name` to a copy of `tree`'s subtree (replacing any child of that name).
    fn _set_child(&self, py: Python<'_>, name: &str, tree: &Bound<'_, PyDataTree>) -> PyResult<()> {
        let sub = tree.borrow().node_tree(py)?;
        self.root
            .borrow_mut(py)
            .inner
            .set_child_at(&self.path, name, sub)
            .py_err()?;
        Ok(())
    }

    /// Remove child `name`.
    fn _del_child(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        self.root
            .borrow_mut(py)
            .inner
            .remove_child_at(&self.path, name)
            .map_err(|_| PyKeyError::new_err(name.to_owned()))?;
        Ok(())
    }

    /// This node's subtree as a new tree (its own root), keeping this node's name.
    fn _detached(&self, py: Python<'_>) -> PyResult<Py<PyDataTree>> {
        let name = self.name(py);
        new_root(py, self.node_tree(py)?, name)
    }

    /// Detach this node from its parent: it becomes the root of its own tree (xarray's
    /// `orphan`). Other views of it in the old tree no longer find it.
    fn orphan(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some((parent, name)) = self.split_self() else {
            return Ok(());
        };
        let sub = self
            .root
            .borrow_mut(py)
            .inner
            .remove_child_at(&parent, &name)
            .map_err(|_| removed(&self.path))?;
        self.root = Py::new(
            py,
            TreeRoot {
                inner: sub,
                name: Some(name),
            },
        )?;
        self.path = "/".to_owned();
        Ok(())
    }

    /// Read every lazily loaded value of this node's subtree into memory (in place).
    fn _load(&self, py: Python<'_>) -> PyResult<()> {
        let mut sub = self.node_tree(py)?;
        py.detach(|| sub.load()).py_err()?;
        let mut root = self.root.borrow_mut(py);
        match self.split_self() {
            None => root.inner = sub,
            Some((parent, name)) => {
                root.inner.set_child_at(&parent, &name, sub).py_err()?;
            }
        }
        Ok(())
    }

    /// Write this node's subtree as netCDF-4 groups (lazy values stream chunk by chunk).
    #[pyo3(signature = (path, format="NETCDF4"))]
    fn to_netcdf(&self, py: Python<'_>, path: PathBuf, format: &str) -> PyResult<()> {
        let opts = WriteOptions::format(crate::raw::format_from_name(format)?);
        let sub = self.node_tree(py)?;
        py.detach(|| sub.to_netcdf_with(&path, &opts))
            .map_err(xr_error)
    }

    /// As an `xarray.DataTree` (needs xarray).
    fn to_xarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let xr = py.import("xarray")?;
        let sub = self.node_tree(py)?;
        let d = PyDict::new(py);
        for path in sub.paths() {
            let node = sub.get(&path).py_err()?;
            let ds = crate::native::new_ds(py, node.dataset().clone())?.into_bound(py);
            d.set_item(path, ds.call_method0("to_xarray")?)?;
        }
        let kw = PyDict::new(py);
        kw.set_item("name", self.name(py))?;
        xr.getattr("DataTree")?
            .call_method("from_dict", (d,), Some(&kw))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let sub = self.node_tree(py)?;
        Ok(sub.to_string().replacen(
            "<xarray.DataTree>",
            &format!("<annoray.DataTree {}>", self.path),
            1,
        ))
    }

    /// Every node's path below this one (this one first, depth first), relative to the
    /// root.
    #[getter]
    fn paths(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let sub = self.node_tree(py)?;
        Ok(sub
            .paths()
            .into_iter()
            .map(|p| join(&self.path, &p))
            .collect())
    }

    fn _list_children<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let names = self._child_names(py)?;
        let views = names
            .iter()
            .map(|n| self.view(py, join(&self.path, n)))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, views)
    }
}

/// Open a netCDF-4/HDF5 file's groups as a `DataTree` (lazily: values are read on demand).
#[pyfunction]
#[pyo3(signature = (path, *, mask_and_scale=true, decode_times=true, decode_units=true,
    decode_coords=None, concat_characters=true, lazy=true))]
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
pub(crate) fn open_datatree(
    py: Python<'_>,
    path: &Bound<'_, PyAny>,
    mask_and_scale: bool,
    decode_times: bool,
    decode_units: bool,
    decode_coords: Option<&Bound<'_, PyAny>>,
    concat_characters: bool,
    lazy: bool,
) -> PyResult<Py<PyDataTree>> {
    let source = crate::native::open_source(path)?;
    let mut opts = open_options(
        None,
        mask_and_scale,
        decode_times,
        decode_units,
        true,
        concat_characters,
        lazy,
    );
    (opts.decode.decode_coords, opts.decode.decode_coords_all) =
        crate::native::coords_option(decode_coords)?;
    let opened = py
        .detach(|| match source {
            crate::native::OpenSource::Path(p) => open_datatree_with(&p, &opts),
            crate::native::OpenSource::Bytes(b) => annoray::io::open_datatree_from_bytes(b, &opts),
        })
        .map_err(xr_error)?;
    warn_all(py, &opened.warnings)?;
    new_root(py, opened.tree, None)
}

/// An `xarray.DataTree` as an annoray `DataTree`.
pub(crate) fn tree_from_xarray(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyDataTree>> {
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
            ours.bind(py).cast::<PyDataset>()?.borrow().inner.clone(),
        ));
    }
    let name: Option<String> = obj.getattr("name")?.extract().ok();
    new_root(py, build("/", &nodes)?, name)
}
