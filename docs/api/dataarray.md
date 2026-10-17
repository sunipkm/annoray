# DataArray

A labelled, units-aware N-dimensional array.

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   DataArray
```

## Attributes

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.dims
   DataArray.sizes
   DataArray.shape
   DataArray.size
   DataArray.ndim
   DataArray.dtype
   DataArray.values
   DataArray.data
   DataArray.name
   DataArray.attrs
   DataArray.encoding
   DataArray.coords
   DataArray.indexes
   DataArray.xindexes
   DataArray.nbytes
   DataArray.chunks
   DataArray.T
   DataArray.is_lazy
   DataArray.loc
```

## Units

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.units
   DataArray.unit
   DataArray.dimension
   DataArray.to
```

## Memory and copying

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.copy
   DataArray.astype
   DataArray.load
   DataArray.compute
   DataArray.persist
   DataArray.chunk
   DataArray.close
   DataArray.item
   DataArray.to_numpy
```

## Indexing

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.isel
   DataArray.sel
   DataArray.head
   DataArray.tail
   DataArray.thin
   DataArray.drop_isel
   DataArray.drop_sel
   DataArray.query
   DataArray.put
   DataArray.put_sel
   DataArray.reindex
   DataArray.reindex_like
   DataArray.interp
   DataArray.interp_like
   DataArray.sortby
   DataArray.drop_duplicates
   DataArray.get_index
   DataArray.set_index
   DataArray.reset_index
   DataArray.set_xindex
   DataArray.swap_dims
   DataArray.broadcast_like
```

## Missing values

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.isnull
   DataArray.notnull
   DataArray.fillna
   DataArray.dropna
   DataArray.ffill
   DataArray.bfill
   DataArray.interpolate_na
   DataArray.where
   DataArray.combine_first
```

## Comparisons

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.equals
   DataArray.identical
   DataArray.broadcast_equals
   DataArray.isin
```

## Aggregation

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.all
   DataArray.any
   DataArray.argmax
   DataArray.argmin
   DataArray.count
   DataArray.max
   DataArray.mean
   DataArray.median
   DataArray.min
   DataArray.prod
   DataArray.std
   DataArray.sum
   DataArray.var
   DataArray.quantile
   DataArray.idxmax
   DataArray.idxmin
```

## Cumulative

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.cumsum
   DataArray.cumprod
   DataArray.cumulative
   DataArray.cumulative_integrate
```

## Computation

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.clip
   DataArray.round
   DataArray.conj
   DataArray.conjugate
   DataArray.real
   DataArray.imag
   DataArray.diff
   DataArray.differentiate
   DataArray.integrate
   DataArray.shift
   DataArray.roll
   DataArray.rank
   DataArray.dot
   DataArray.polyfit
   DataArray.argsort
   DataArray.pipe
   DataArray.reduce
   DataArray.map_blocks
```

## Grouped and windowed

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.groupby
   DataArray.groupby_bins
   DataArray.resample
   DataArray.rolling
   DataArray.rolling_exp
   DataArray.coarsen
   DataArray.weighted
```

## Reshaping and renaming

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.transpose
   DataArray.expand_dims
   DataArray.squeeze
   DataArray.stack
   DataArray.unstack
   DataArray.pad
   DataArray.rename
   DataArray.drop_vars
   DataArray.assign_coords
   DataArray.assign_attrs
   DataArray.reset_coords
```

## Time and text

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.convert_calendar
   DataArray.dt
   DataArray.str
```

## Plotting

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.plot
```

## Input and output

```{eval-rst}
.. autosummary::
   :toctree: generated/

   DataArray.to_netcdf
   DataArray.to_zarr
   DataArray.to_dataset
   DataArray.to_dataframe
   DataArray.to_series
   DataArray.to_pandas
   DataArray.to_dict
   DataArray.to_xarray
   DataArray.to_astropy
   DataArray.from_dict
   DataArray.from_series
```
