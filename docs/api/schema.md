---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Schemas

A {class}`Schema` states what a dataset must hold: its data variables and coordinates
(dimensions, element types, units, sizes, attributes) and its attributes.
{meth}`Schema.check` raises {class}`SchemaError` listing every way a dataset differs, not
only the first; {meth}`Schema.problems` returns the same list. Only metadata is read, so a
file opened lazily is checked without reading its values. Variables the schema does not name
are allowed unless it is `closed=True`. xarray has no counterpart.

Each variable is a dict of what it must be, every key optional:

| Key | Means |
|---|---|
| `dims` | its dimensions, in order (`ndim`: only how many) |
| `dtype` | its element type: a name (`"float32"`) or a numpy dtype; no casts |
| `units` | its unit, equal as units compare (`"km"` is `"1000 m"`); `None` for no unit, which is not `"1"` |
| `convertible` | a unit it must convert to |
| `dimension` | a {class}`Dimension`, or a unit whose dimension it must have (`"m"`: any length) |
| `sizes`, `min_sizes` | dimension lengths, exactly or at least |
| `attrs` | its attributes, as for the dataset's |
| `optional` | whether it may be absent |

Attributes map to the value they must have, a `set` of values they may have, or `...` for
any value.

```{code-cell} python
import numpy as np
import annoray as ar

schema = ar.Schema(
    data_vars={"intensity": {"dims": ["y", "x"], "dtype": "float32", "units": "1"}},
    coords={"x": {"dims": "x", "dimension": "m"}},
    attrs={"format": "prism-map", "version": {"1", "2"}},
)
ds = ar.Dataset(
    {"intensity": (("y", "x"), np.zeros((3, 4)), {"units": "%"})},
    coords={"x": ("x", np.arange(4.0), {"units": "s"})},
    attrs={"format": "prism-map"},
)
try:
    schema.check(ds)
except ar.SchemaError as e:
    print(e)
```

```{eval-rst}
.. currentmodule:: annoray

.. autosummary::
   :toctree: generated/

   Schema
   Schema.check
   Schema.problems
   SchemaError
```
