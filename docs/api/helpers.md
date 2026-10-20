---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# GroupBy, windows and weights

Objects returned by {meth}`DataArray.groupby` / {meth}`DataArray.resample`,
{meth}`DataArray.rolling`, {meth}`DataArray.rolling_exp`, {meth}`DataArray.coarsen` and
{meth}`DataArray.weighted` (and their `Dataset` counterparts). Each reduction returns a new
object with the grouped or windowed dimension reduced.

Each section has a short example; the examples share this setup:

```{code-cell} python
import numpy as np
import annoray as ar

hourly = ar.DataArray(
    288.0 + 5.0 * np.sin(np.arange(48) * 2 * np.pi / 24),
    dims="time",
    coords={"time": ar.date_range("2026-07-01", 48, "h")},
    units="K",
)
```

## GroupBy

```{code-cell} python
by_hour = hourly.groupby("time.hour")
print(by_hour.mean().sizes)
print(hourly.resample(time="1D").max().values)
print(hourly.resample(time="1D").quantile(0.5).values)
```

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

```{code-cell} python
r = hourly.rolling(time=6, center=True)
print(r.mean().values[2:6])
print(r.construct("window").sizes)
```

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

```{code-cell} python
print(hourly.rolling_exp(time=12).mean().values[-3:])
```

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

```{code-cell} python
print(hourly.coarsen(time=6).mean().values[:4])
```

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

```{code-cell} python
weights = ar.DataArray(np.where(np.arange(48) % 24 < 12, 2.0, 1.0), dims="time")
w = hourly.weighted(weights)
print(w.mean("time").item(), w.sum_of_weights().item())
```

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

