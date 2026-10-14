//! `rarray._rarray`: the Rust side of rarray (xarray-rs for Python).
//!
//! - [`native`]: units-aware `DataArray` and `Dataset` backed by xarray-rs.
//! - [`raw`]: raw file access and writing for the xarray backend engine (`engine="rarray"`).
//! - [`convert`]: values, labels, indexers and errors between Python and Rust.

mod convert;
mod native;
mod raw;

use pyo3::prelude::*;

/// The Rust extension module of rarray.
#[pymodule]
fn _rarray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<raw::RawGroup>()?;
    m.add_function(wrap_pyfunction!(raw::write, m)?)?;
    m.add_class::<native::PyDataArray>()?;
    m.add_class::<native::PyDataset>()?;
    m.add_class::<native::PyGroupBy>()?;
    m.add_class::<native::PyRolling>()?;
    m.add_class::<native::PyDt>()?;
    m.add_function(wrap_pyfunction!(native::open_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(native::open_mfdataset, m)?)?;
    m.add_function(wrap_pyfunction!(native::concat, m)?)?;
    m.add_function(wrap_pyfunction!(native::merge, m)?)?;
    m.add_function(wrap_pyfunction!(native::align, m)?)?;
    m.add_function(wrap_pyfunction!(native::from_xarray, m)?)?;
    m.add("UnitError", m.py().get_type::<convert::UnitError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
