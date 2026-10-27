//! Exchange annoray values with the `annoray` Python package from another PyO3 extension
//! module.
//!
//! An extension that holds [`annoray`] arrays and [`annoray::units`] quantities in Rust takes
//! and returns `annoray.Unit`, `annoray.Quantity`, `annoray.DataArray`, `annoray.Dataset` and
//! `annoray.DataTree` objects through the boundary types of this crate, which implement
//! PyO3's `FromPyObject` and `IntoPyObject`:
//!
//! | Rust | Python | type |
//! |---|---|---|
//! | [`Unit`](annoray::Unit) | `annoray.Unit` (or a unit string, from Python) | [`ArUnit`] |
//! | [`Quantity<K, T>`](annoray::units::Quantity) | `annoray.Quantity` (or a unit string, from Python) | [`ArQuantity`] |
//! | [`DataArray`](annoray::DataArray) | `annoray.DataArray` | [`ArDataArray`]; [`ArDataArrayRef`] reads values in place |
//! | [`Dataset`](annoray::Dataset) | `annoray.Dataset` | [`ArDataset`] |
//! | [`DataTree`](annoray::DataTree) | `annoray.DataTree` | [`ArDataTree`] |
//!
//! A downstream `#[pyfunction]` takes and returns them directly (this example is
//! `tests/ext`'s, which the tests build and run):
//!
//! ```ignore
//! use annoray_pyo3::annoray::units::dims::{Length, Time, Velocity};
//! use annoray_pyo3::{ArDataArray, ArQuantity, ArUnit, unit_error};
//! use pyo3::prelude::*;
//!
//! /// `da` converted to `unit` (an `annoray.Unit` or a unit string) in Rust.
//! #[pyfunction]
//! fn convert(py: Python<'_>, da: ArDataArray, unit: ArUnit) -> PyResult<ArDataArray> {
//!     let out = da.0.to_unit(&unit.0).map_err(|e| unit_error(py, e))?;
//!     Ok(ArDataArray(out))
//! }
//!
//! /// The time to cover `distance` at `speed`: `travel_time(Quantity(10, "km"), "km/h")`.
//! #[pyfunction]
//! fn travel_time(
//!     py: Python<'_>,
//!     distance: ArQuantity<Length>,  // a length, or TypeError / UnitError
//!     speed: ArQuantity<Velocity>,
//! ) -> PyResult<ArQuantity<Time>> {
//!     let time = (&distance.0 / &speed.0).map_err(|e| unit_error(py, e))?;
//!     Ok(ArQuantity(time))  // an annoray.Quantity
//! }
//! ```
//!
//! # The objects are `annoray`'s own
//!
//! PyO3 creates each `#[pyclass]` separately in every extension module that links it, so an
//! extension cannot use `annoray`'s classes as Rust types: its copies would not be the
//! classes `import annoray` loads (`isinstance` would fail, and annoray would reject them).
//! This crate therefore builds and reads objects only through the imported `annoray` package:
//! values cross as numpy arrays and plain Python data (the [`parts`] format), and
//! `annoray._annoray` builds its own objects from them, with the same code as this crate.
//!
//! The first conversion imports `annoray` and checks that it is compatible with this crate
//! (the same release under Cargo's caret rule, and the same exchange [`PROTOCOL`]); it raises
//! `ImportError` otherwise. [`annoray_module`] runs the check explicitly, for example when the
//! extension module is imported.
//!
//! # Copies
//!
//! Each extension module has its own memory allocator, so values cannot change owner between
//! two extension modules; what crosses is copied once, into the side that keeps it:
//!
//! | direction | copies of the values |
//! |---|---|
//! | Rust → Python, by value ([`ArDataArray`], [`ArDataset`], [`ArDataTree`]) | one (numbers move into numpy without a copy; annoray copies them into its own memory) |
//! | Rust → Python, by reference (`&ArDataArray`, `&ArDataset`) | two |
//! | Python → Rust ([`ArDataArray`], [`ArDataset`]) | one (annoray lends read-only numpy views; they are copied into the Rust value) |
//! | Python → Rust, in place ([`ArDataArrayRef`]) | none, for numbers, date-times and durations |
//! | Python → Rust ([`ArDataTree`]) | two (trees do not lend their values) |
//!
//! Date-times and durations take one more copy from Rust to Python (from annoray's `Instant`
//! and `Duration` to numpy's layout); text and CF dates are converted element by element.
//! Values read lazily from a file are read when converted. Encodings (how values are stored
//! on disk) do not cross.
//!
//! # Linking
//!
//! The extension module chooses PyO3's features (`abi3-*`, `extension-module`) and annoray's;
//! this crate sets none and defines no module or allocator. It needs the PyO3 and numpy
//! versions annoray's Python package is built with (re-exported here as [`pyo3`] and
//! [`numpy`]), since an extension links only one PyO3.

pub mod compat;
pub mod convert;
mod data;
pub mod parts;
mod units;

pub use compat::{PROTOCOL, VERSION, annoray_module, compatible};
pub use data::{ArDataArray, ArDataArrayRef, ArDataTree, ArDataset, ArVariableRef};
pub use units::{ArQuantity, ArUnit, Magnitude, unit_error, value_error};

/// The annoray crate this one exchanges values of.
pub use annoray;
/// The numpy bindings this crate uses.
pub use numpy;
/// The PyO3 this crate uses.
pub use pyo3;
