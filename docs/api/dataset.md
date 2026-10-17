# Dataset

A dict-like collection of data arrays sharing dimensions and coordinates.

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Dataset
```

## Attributes

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.dims
   Dataset.sizes
   Dataset.values
   Dataset.attrs
   Dataset.encoding
   Dataset.coords
   Dataset.data_vars
   Dataset.variables
   Dataset.indexes
   Dataset.xindexes
   Dataset.nbytes
   Dataset.chunks
   Dataset.loc
```

## Contents

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.keys
   Dataset.items
   Dataset.get
   Dataset.update
   Dataset.assign
   Dataset.set_coords
```

## Memory and copying

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.copy
   Dataset.astype
   Dataset.load
   Dataset.compute
   Dataset.persist
   Dataset.chunk
   Dataset.close
```

## Indexing

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.isel
   Dataset.sel
   Dataset.head
   Dataset.tail
   Dataset.thin
   Dataset.drop_isel
   Dataset.drop_sel
   Dataset.query
   Dataset.reindex
   Dataset.reindex_like
   Dataset.interp
   Dataset.interp_like
   Dataset.sortby
   Dataset.drop_duplicates
   Dataset.get_index
   Dataset.set_index
   Dataset.reset_index
   Dataset.set_xindex
   Dataset.swap_dims
```

## Missing values

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.isnull
   Dataset.notnull
   Dataset.fillna
   Dataset.dropna
   Dataset.ffill
   Dataset.bfill
   Dataset.interpolate_na
   Dataset.where
   Dataset.combine_first
```

## Comparisons

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.equals
   Dataset.identical
   Dataset.isin
```

## Aggregation

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.all
   Dataset.any
   Dataset.argmax
   Dataset.argmin
   Dataset.count
   Dataset.max
   Dataset.mean
   Dataset.median
   Dataset.min
   Dataset.prod
   Dataset.std
   Dataset.sum
   Dataset.var
   Dataset.quantile
   Dataset.idxmax
   Dataset.idxmin
```

## Cumulative

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.cumsum
   Dataset.cumprod
   Dataset.cumulative_integrate
```

## Computation

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.clip
   Dataset.round
   Dataset.conj
   Dataset.conjugate
   Dataset.real
   Dataset.imag
   Dataset.diff
   Dataset.differentiate
   Dataset.integrate
   Dataset.shift
   Dataset.roll
   Dataset.rank
   Dataset.polyfit
   Dataset.pipe
   Dataset.reduce
   Dataset.map
   Dataset.map_blocks
```

## Grouped and windowed

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.groupby
   Dataset.groupby_bins
   Dataset.resample
   Dataset.rolling
   Dataset.rolling_exp
   Dataset.coarsen
   Dataset.weighted
```

## Reshaping and renaming

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.transpose
   Dataset.expand_dims
   Dataset.squeeze
   Dataset.stack
   Dataset.unstack
   Dataset.to_stacked_array
   Dataset.pad
   Dataset.rename
   Dataset.rename_dims
   Dataset.rename_vars
   Dataset.drop_vars
   Dataset.drop_dims
   Dataset.assign_coords
   Dataset.assign_attrs
   Dataset.reset_coords
   Dataset.merge
   Dataset.filter_by_attrs
   Dataset.info
```

## Time and text

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.convert_calendar
```

## Plotting

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.plot
```

## Input and output

```{eval-rst}
.. autosummary::
   :toctree: generated/

   Dataset.to_netcdf
   Dataset.to_zarr
   Dataset.to_dataarray
   Dataset.to_array
   Dataset.to_dataframe
   Dataset.to_pandas
   Dataset.to_dict
   Dataset.to_xarray
   Dataset.from_dict
   Dataset.from_dataframe
```
