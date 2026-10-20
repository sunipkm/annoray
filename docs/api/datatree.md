---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# DataTree

A tree of datasets; a node is a view of the shared tree at a path.

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   DataTree
```

Each section has a short example; the examples share this setup:

```{code-cell} python
import numpy as np
import annoray as ar

tree = ar.DataTree.from_dict({
    "/": ar.Dataset(attrs={"title": "campaign"}),
    "/surface": ar.Dataset({"t": ("time", [280.0, 281.0, 283.0], {"units": "K"})}),
    "/upper": ar.Dataset({"t": ("time", [250.0, 248.5, 247.0], {"units": "K"})}),
})
```

## Tree structure

```{code-cell} python
print(tree.groups)
print(tree["surface"].path, tree["surface"].parent.path, tree["surface"].is_leaf)
```

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

```{code-cell} python
print(tree["surface"].dataset["t"].units)
print(list(tree["upper"].data_vars))
```

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

```{code-cell} python
tree2 = tree.assign(note=ar.DataArray("synthetic"))
print(list(tree2.data_vars), list(tree.data_vars))
```

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

```{code-cell} python
print(tree.match("surf*").groups)
print(tree.filter(lambda node: node.has_data).groups)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.match
   DataTree.filter
   DataTree.filter_like
   DataTree.prune
```

## Mapping over nodes

```{code-cell} python
celsius = tree.map_over_datasets(lambda ds: ds.map(lambda v: v.to("degC")))
print(celsius["surface"]["t"].values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.map_over_datasets
   DataTree.pipe
```

## Memory and copying

```{code-cell} python
print(tree.astype("float32")["surface"]["t"].dtype)
```

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

```{code-cell} python
print(tree.isel(time=0)["upper"]["t"].values)
```

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

```{code-cell} python
print(tree.where(tree > ar.Quantity(249.0, "K"))["upper"]["t"].values)
```

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

```{code-cell} python
print(tree.equals(tree.copy()))
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.equals
   DataTree.identical
```

## Aggregation

```{code-cell} python
print(tree.mean("time")["surface"]["t"].item())
```

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

```{code-cell} python
print(tree.cumsum("time")["upper"]["t"].values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.cumsum
   DataTree.cumprod
```

## Computation

```{code-cell} python
print(tree.diff("time")["surface"]["t"].values)
```

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

```{code-cell} python
print(tree.isel(time=slice(0, 1)).squeeze()["surface"]["t"].values)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.squeeze
```

## Input and output

```{code-cell} python
import os, tempfile

path = os.path.join(tempfile.mkdtemp(), "tree.nc")
tree.to_netcdf(path)
print(ar.open_datatree(path).groups)
```

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataTree.to_netcdf
   DataTree.to_zarr
   DataTree.to_dict
   DataTree.to_xarray
   DataTree.from_dict
```
