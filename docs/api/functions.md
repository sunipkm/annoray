# Top-level functions

## Opening and saving

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   open_dataset
   open_dataarray
   open_datatree
   open_groups
   open_mfdataset
   open_zarr
   load_dataset
   load_dataarray
   decode_cf
   to_netcdf
   to_zarr
   save_mfdataset
```

## Combining

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   concat
   merge
   align
   broadcast
   combine_by_coords
   combine_nested
```

## Computation

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   where
   apply_ufunc
   dot
   cross
   corr
   cov
   polyval
   full_like
   zeros_like
   ones_like
   map_blocks
   map_over_datasets
   group_subtrees
   unify_chunks
```

## Time

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   date_range
   cftime_range
   infer_freq
```

## Streaming and options

Lazy data (opened from files, or computed from it) is reduced and written in slabs within a
memory budget.

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   set_stream_bytes
   stream_bytes
   set_options
   get_options
   show_versions
```

## Interoperability

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   from_xarray
   to_astropy
   from_astropy
```

## Extending

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   register_dataarray_accessor
   register_dataset_accessor
```

