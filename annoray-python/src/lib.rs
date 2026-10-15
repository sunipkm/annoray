//! `annoray._annoray`: the Rust side of annoray's Python package.
//!
//! - [`native`]: units-aware `DataArray` and `Dataset` backed by annoray.
//! - [`tree`]: `DataTree`, a tree of datasets (netCDF-4 groups).
//! - [`raw`]: raw file access and writing for the xarray backend engine (`engine="annoray"`).
//! - [`convert`]: values, labels, indexers and errors between Python and Rust.

mod alloc;
mod convert;
mod native;
mod raw;
mod tree;

use pyo3::prelude::*;

#[global_allocator]
static ALLOCATOR: alloc::Allocator = alloc::Allocator;

/// The Rust extension module of annoray.
#[pymodule]
fn _annoray(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<raw::RawGroup>()?;
    m.add_function(wrap_pyfunction!(raw::write, m)?)?;
    m.add_class::<native::PyDataArray>()?;
    m.add_class::<native::PyDataset>()?;
    m.add_class::<native::PyGroupBy>()?;
    m.add_class::<native::PyRolling>()?;
    m.add_class::<native::PyDt>()?;
    m.add_class::<tree::PyDataTree>()?;
    m.add_function(wrap_pyfunction!(tree::open_datatree, m)?)?;
    m.add_function(wrap_pyfunction!(native::open_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(native::open_mfdataset, m)?)?;
    m.add_function(wrap_pyfunction!(native::concat, m)?)?;
    m.add_function(wrap_pyfunction!(native::date_range, m)?)?;
    m.add_function(wrap_pyfunction!(native::merge, m)?)?;
    m.add_function(wrap_pyfunction!(native::align, m)?)?;
    m.add_function(wrap_pyfunction!(native::from_xarray, m)?)?;
    m.add_function(wrap_pyfunction!(native::set_stream_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(native::stream_bytes, m)?)?;
    m.add("UnitError", m.py().get_type::<convert::UnitError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
