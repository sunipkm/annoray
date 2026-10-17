# DataTree

A tree of datasets; a node is a view of the shared tree at a path.

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   DataTree
```

## Tree structure

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.name
   DataTree.parent
   DataTree.children
   DataTree.root
   DataTree.path
   DataTree.paths
   DataTree.depth
   DataTree.level
   DataTree.width
   DataTree.lineage
   DataTree.parents
   DataTree.ancestors
   DataTree.descendants
   DataTree.leaves
   DataTree.groups
   DataTree.subtree
   DataTree.is_root
   DataTree.is_leaf
   DataTree.is_empty
   DataTree.has_data
   DataTree.has_attrs
   DataTree.relative_to
   DataTree.find_common_ancestor
   DataTree.same_tree
   DataTree.isomorphic
```

## Node contents

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.dataset
   DataTree.ds
   DataTree.to_dataset
   DataTree.data_vars
   DataTree.coords
   DataTree.dims
   DataTree.sizes
   DataTree.attrs
   DataTree.encoding
   DataTree.values
   DataTree.keys
   DataTree.items
   DataTree.get
```

## Editing

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.assign
   DataTree.copy
   DataTree.orphan
   DataTree.drop_nodes
   DataTree.drop_vars
   DataTree.rename_vars
```

## Selecting nodes

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.match
   DataTree.filter
   DataTree.filter_like
   DataTree.prune
```

## Mapping over nodes

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.map_over_datasets
   DataTree.pipe
```

## Memory and copying

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.astype
   DataTree.load
   DataTree.compute
   DataTree.persist
   DataTree.chunk
```

## Indexing

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.isel
   DataTree.sel
   DataTree.head
   DataTree.tail
   DataTree.thin
   DataTree.reindex
   DataTree.interp
```

## Missing values

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.isnull
   DataTree.notnull
   DataTree.fillna
   DataTree.dropna
   DataTree.where
```

## Comparisons

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.equals
   DataTree.identical
```

## Aggregation

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.all
   DataTree.any
   DataTree.count
   DataTree.max
   DataTree.mean
   DataTree.median
   DataTree.min
   DataTree.prod
   DataTree.std
   DataTree.sum
   DataTree.var
   DataTree.quantile
```

## Cumulative

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.cumsum
   DataTree.cumprod
```

## Computation

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.clip
   DataTree.round
   DataTree.diff
   DataTree.shift
   DataTree.roll
```

## Reshaping and renaming

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.squeeze
```

## Input and output

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.to_netcdf
   DataTree.to_zarr
   DataTree.to_dict
   DataTree.to_xarray
   DataTree.from_dict
```
