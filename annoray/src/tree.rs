//! [`DataTree`]: a tree of datasets, as netCDF-4 groups are (xarray's `DataTree`).
//!
//! Each node holds its own dataset and named children. Dimension (index) coordinates are
//! inherited, as in xarray: a node's [`inherited`](DataTree::inherited) dataset adds its
//! ancestors' dimension coordinates (those it does not define itself), first. Other
//! coordinates stay with the node that defines them. A child must agree with what it
//! inherits: a dimension shared with an inherited dimension coordinate must have the same
//! length, and a coordinate of the same name must have the same values.

use std::fmt;

use indexmap::IndexMap;

use crate::ops::variables_equal;
use crate::{CoreError, Dataset, Name, Result, Variable};

/// A tree of datasets.
#[derive(Debug, Clone, Default)]
pub struct DataTree {
    dataset: Dataset,
    children: IndexMap<Name, DataTree>,
}

fn conflict(path: &str, reason: impl Into<String>) -> CoreError {
    CoreError::CannotCombine {
        name: Name::from(path),
        reason: reason.into(),
    }
}

fn split(path: &str) -> impl Iterator<Item = &str> {
    path.split('/').filter(|p| !p.is_empty())
}

impl DataTree {
    /// A tree with one node.
    #[must_use]
    pub fn new(dataset: Dataset) -> Self {
        Self {
            dataset,
            children: IndexMap::new(),
        }
    }

    /// Add a child (a whole subtree).
    ///
    /// # Errors
    /// [`CoreError::NameConflict`] if the name is taken or contains `/`;
    /// [`CoreError::CannotCombine`] if the subtree disagrees with this node's coordinates.
    pub fn add_child(&mut self, name: impl Into<Name>, child: DataTree) -> Result<()> {
        let name = name.into();
        if name.contains('/') || name.is_empty() || self.children.contains_key(&name) {
            return Err(CoreError::NameConflict(name));
        }
        child.validate(&self.index_coords(), &format!("/{name}"))?;
        self.children.insert(name, child);
        Ok(())
    }

    /// [`add_child`](Self::add_child), by value.
    ///
    /// # Errors
    /// As for [`add_child`](Self::add_child).
    pub fn with_child(mut self, name: impl Into<Name>, child: DataTree) -> Result<Self> {
        self.add_child(name, child)?;
        Ok(self)
    }

    /// This node's own dataset.
    #[must_use]
    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    /// Replace this node's dataset.
    ///
    /// # Errors
    /// [`CoreError::CannotCombine`] if the children disagree with the new coordinates.
    pub fn set_dataset(&mut self, dataset: Dataset) -> Result<()> {
        let coords: IndexMap<Name, Variable> = dataset
            .coords()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (name, c) in &self.children {
            c.validate(&coords, &format!("/{name}"))?;
        }
        self.dataset = dataset;
        Ok(())
    }

    /// The children, by name.
    #[must_use]
    pub fn children(&self) -> &IndexMap<Name, DataTree> {
        &self.children
    }

    /// This node's dataset and its children, by value (nothing is copied).
    #[must_use]
    pub fn into_parts(self) -> (Dataset, IndexMap<Name, DataTree>) {
        (self.dataset, self.children)
    }

    /// The node at a `/`-separated path (`""` or `"/"` for this one).
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] (naming the path) if there is no such node.
    pub fn get(&self, path: &str) -> Result<&DataTree> {
        let mut node = self;
        for part in split(path) {
            node = node
                .children
                .get(part)
                .ok_or_else(|| CoreError::NoSuchVariable(Name::from(path)))?;
        }
        Ok(node)
    }

    /// The dataset at `path` with the coordinates it inherits from its ancestors.
    ///
    /// # Errors
    /// As for [`get`](Self::get).
    pub fn inherited(&self, path: &str) -> Result<Dataset> {
        let mut inherited: IndexMap<Name, Variable> = IndexMap::new();
        let mut node = self;
        for part in split(path) {
            inherited.extend(node.index_coords());
            node = node
                .children
                .get(part)
                .ok_or_else(|| CoreError::NoSuchVariable(Name::from(path)))?;
        }
        // Inherited coordinates first (as xarray orders them), then the node's own.
        let mut ds = Dataset::new();
        for (k, v) in inherited {
            if !node.dataset.contains(&k) {
                ds.add_coord(k, v)?;
            }
        }
        for (k, v) in node.dataset.coords() {
            ds.add_coord(k.clone(), v.clone())?;
        }
        for (k, v) in node.dataset.data_vars() {
            ds.add_var(k.clone(), v.clone())?;
        }
        *ds.attrs_mut() = node.dataset.attrs().clone();
        *ds.encoding_mut() = node.dataset.encoding().clone();
        Ok(ds)
    }

    /// Every node's path, depth first, this one (`"/"`) first.
    #[must_use]
    pub fn paths(&self) -> Vec<String> {
        fn walk(t: &DataTree, path: &str, out: &mut Vec<String>) {
            out.push(if path.is_empty() {
                "/".to_owned()
            } else {
                path.to_owned()
            });
            for (k, c) in &t.children {
                walk(c, &format!("{path}/{k}"), out);
            }
        }
        let mut out = Vec::new();
        walk(self, "", &mut out);
        out
    }

    /// A tree of the same shape with `f` applied to every node's own dataset.
    ///
    /// # Errors
    /// Whatever `f` returns, or [`CoreError::CannotCombine`] if the results disagree.
    pub fn map_over_datasets(
        &self,
        f: &mut dyn FnMut(&Dataset) -> Result<Dataset>,
    ) -> Result<DataTree> {
        let mut out = DataTree::new(f(&self.dataset)?);
        for (k, c) in &self.children {
            out.add_child(k.clone(), c.map_over_datasets(f)?)?;
        }
        Ok(out)
    }

    /// Read all lazily loaded data in the tree.
    ///
    /// # Errors
    /// [`CoreError::Read`] if a source cannot be read.
    pub fn load(&mut self) -> Result<()> {
        self.dataset.load()?;
        for c in self.children.values_mut() {
            c.load()?;
        }
        Ok(())
    }

    /// The coordinates the node at `path` inherits from its ancestors (not its own).
    fn inherited_at(&self, path: &str) -> Result<IndexMap<Name, Variable>> {
        let mut inherited: IndexMap<Name, Variable> = IndexMap::new();
        let mut node = self;
        for part in split(path) {
            inherited.extend(node.index_coords());
            node = node
                .children
                .get(part)
                .ok_or_else(|| CoreError::NoSuchVariable(Name::from(path)))?;
        }
        Ok(inherited)
    }

    fn node_mut(&mut self, path: &str) -> Result<&mut DataTree> {
        let mut node = self;
        for part in split(path) {
            node = node
                .children
                .get_mut(part)
                .ok_or_else(|| CoreError::NoSuchVariable(Name::from(path)))?;
        }
        Ok(node)
    }

    /// Replace the dataset of the node at `path`, returning the old one. The new dataset must
    /// agree with the coordinates the node inherits, and its descendants with it.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] for a missing node; [`CoreError::CannotCombine`] for a
    /// disagreement. The tree is unchanged on error.
    pub fn set_dataset_at(&mut self, path: &str, dataset: Dataset) -> Result<Dataset> {
        let inherited = self.inherited_at(path)?;
        let node = self.node_mut(path)?;
        let probe = DataTree {
            dataset,
            children: std::mem::take(&mut node.children),
        };
        let checked = probe.validate(&inherited, path);
        let DataTree { dataset, children } = probe;
        node.children = children;
        checked?;
        Ok(std::mem::replace(&mut node.dataset, dataset))
    }

    /// Set child `name` of the node at `path` to `child` (added, or replacing the subtree of
    /// that name, which is returned). The subtree must agree with what it inherits there.
    ///
    /// # Errors
    /// [`CoreError::NameConflict`] for a name with `/` or empty; [`CoreError::NoSuchVariable`]
    /// for a missing node; [`CoreError::CannotCombine`] for a disagreement. The tree is
    /// unchanged on error.
    pub fn set_child_at(
        &mut self,
        path: &str,
        name: impl Into<Name>,
        child: DataTree,
    ) -> Result<Option<DataTree>> {
        let name = name.into();
        if name.contains('/') || name.is_empty() {
            return Err(CoreError::NameConflict(name));
        }
        let mut inherited = self.inherited_at(path)?;
        let node = self.node_mut(path)?;
        inherited.extend(node.index_coords());
        let base = if split(path).next().is_none() {
            String::new()
        } else {
            format!("/{}", split(path).collect::<Vec<_>>().join("/"))
        };
        child.validate(&inherited, &format!("{base}/{name}"))?;
        Ok(node.children.insert(name, child))
    }

    /// Remove child `name` of the node at `path`, returning its subtree.
    ///
    /// # Errors
    /// [`CoreError::NoSuchVariable`] if there is no such node or child.
    pub fn remove_child_at(&mut self, path: &str, name: &str) -> Result<DataTree> {
        self.node_mut(path)?
            .children
            .shift_remove(name)
            .ok_or_else(|| CoreError::NoSuchVariable(Name::from(name)))
    }

    /// The coordinates children inherit: dimension (index) coordinates only, as in xarray.
    /// Other coordinates stay with the node that defines them.
    fn index_coords(&self) -> IndexMap<Name, Variable> {
        self.dataset
            .coords()
            .iter()
            .filter(|(k, v)| v.dims() == [(*k).clone()])
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Check this subtree against the coordinates it inherits.
    fn validate(&self, inherited: &IndexMap<Name, Variable>, path: &str) -> Result<()> {
        let sizes = self.dataset.sizes();
        for (k, v) in inherited {
            // Dimension coordinates fix the length of their dimension below them.
            if let ([d], [n]) = (v.dims(), v.shape())
                && d == k
                && let Some(m) = sizes.get(d)
                && m != n
            {
                return Err(conflict(
                    path,
                    format!("dimension {d:?} has length {m}, but {n} above"),
                ));
            }
            if let Some(own) = self.dataset.coords().get(k)
                && !variables_equal(own, v)
            {
                return Err(conflict(
                    path,
                    format!("coordinate {k:?} differs from the one inherited"),
                ));
            }
        }
        let mut below = inherited.clone();
        below.extend(self.index_coords());
        for (name, c) in &self.children {
            c.validate(&below, &format!("{path}/{name}"))?;
        }
        Ok(())
    }
}

impl fmt::Display for DataTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "<xarray.DataTree>")?;
        for path in self.paths() {
            writeln!(f, "Group: {path}")?;
            if let Ok(node) = self.get(&path) {
                for line in node.dataset.to_string().lines().skip(1) {
                    writeln!(f, "    {line}")?;
                }
            }
        }
        Ok(())
    }
}
