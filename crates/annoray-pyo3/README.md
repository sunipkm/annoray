# annoray-pyo3

Exchange [annoray](https://docs.rs/annoray) values with the `annoray` Python package from
another PyO3 extension module.

An extension that holds annoray arrays and quantities in Rust takes and returns
`annoray.Unit`, `annoray.Quantity`, `annoray.DataArray`, `annoray.Dataset` and
`annoray.DataTree` through the boundary types `ArUnit`, `ArQuantity`, `ArDataArray`,
`ArDataset` and `ArDataTree`, which implement PyO3's `FromPyObject` and `IntoPyObject`. The
objects are the imported package's own: `isinstance(x, annoray.DataArray)` holds, and
annoray's operations accept them.

```rust,ignore
use annoray_pyo3::annoray::units::dims::{Length, Time, Velocity};
use annoray_pyo3::{ArDataArray, ArQuantity, ArUnit, unit_error};
use pyo3::prelude::*;

/// `da` converted to `unit` (an `annoray.Unit` or a unit string) in Rust.
#[pyfunction]
fn convert(py: Python<'_>, da: ArDataArray, unit: ArUnit) -> PyResult<ArDataArray> {
    let out = da.0.to_unit(&unit.0).map_err(|e| unit_error(py, e))?;
    Ok(ArDataArray(out))
}

/// The time to cover `distance` at `speed`, dimensions checked by the compiler.
#[pyfunction]
fn travel_time(
    py: Python<'_>,
    distance: ArQuantity<Length>,
    speed: ArQuantity<Velocity>,
) -> PyResult<ArQuantity<Time>> {
    let time = (&distance.0 / &speed.0).map_err(|e| unit_error(py, e))?;
    Ok(ArQuantity(time))
}
```

- **Everything carries:** dimensions, coordinates with their units, attributes, names, time
  zones and every element type (floats, complex numbers, integers, booleans, text,
  date-times, durations, CF dates in any calendar to the nanosecond).
- **Copies:** values cross as numpy arrays, copied once into the side that keeps them (each
  extension has its own allocator); `ArDataArrayRef` reads an `annoray.DataArray`'s numbers,
  date-times and durations in place, without a copy. The crate documentation lists each
  direction's copies.
- **Checked:** the first conversion checks that the installed `annoray` package is
  compatible with this crate (release and exchange protocol), raising `ImportError` if not.
- **Linking:** use the PyO3 and numpy versions annoray's package is built with (re-exported
  as `annoray_pyo3::pyo3` and `annoray_pyo3::numpy`); the extension sets PyO3's features
  (`abi3-*`, `extension-module`). The crate defines no module and no allocator.
