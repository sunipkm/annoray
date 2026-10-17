# GroupBy, windows and weights

Objects returned by {meth}`DataArray.groupby` / {meth}`DataArray.resample`,
{meth}`DataArray.rolling`, {meth}`DataArray.rolling_exp`, {meth}`DataArray.coarsen` and
{meth}`DataArray.weighted` (and their `Dataset` counterparts). Each reduction returns a new
object with the grouped or windowed dimension reduced.

## GroupBy

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   GroupBy
   GroupBy.mean
   GroupBy.sum
   GroupBy.std
   GroupBy.var
   GroupBy.min
   GroupBy.max
   GroupBy.median
   GroupBy.prod
   GroupBy.count
   GroupBy.all
   GroupBy.any
   GroupBy.argmin
   GroupBy.argmax
   GroupBy.first
   GroupBy.last
   GroupBy.quantile
   GroupBy.map
   GroupBy.reduce
```

## Rolling

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Rolling
   Rolling.mean
   Rolling.sum
   Rolling.std
   Rolling.var
   Rolling.min
   Rolling.max
   Rolling.median
   Rolling.prod
   Rolling.count
   Rolling.all
   Rolling.any
   Rolling.argmin
   Rolling.argmax
   Rolling.construct
   Rolling.reduce
```

## RollingExp

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   RollingExp
   RollingExp.mean
   RollingExp.sum
   RollingExp.std
   RollingExp.var
   RollingExp.count
   RollingExp.cov
   RollingExp.corr
```

## Coarsen

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Coarsen
   Coarsen.mean
   Coarsen.sum
   Coarsen.std
   Coarsen.var
   Coarsen.min
   Coarsen.max
   Coarsen.median
   Coarsen.prod
   Coarsen.count
   Coarsen.construct
```

## Weighted

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Weighted
   Weighted.mean
   Weighted.sum
   Weighted.std
   Weighted.var
   Weighted.quantile
   Weighted.sum_of_weights
```

