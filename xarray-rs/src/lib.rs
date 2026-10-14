//! xarray-rs: labelled, units-aware n-dimensional arrays for Rust, with netCDF and HDF5 I/O.
//!
//! This is the facade crate; it re-exports the workspace crates. See `PLAN.md` for the roadmap.
//!
//! ```
//! use xarray_rs::{Variable, isel, ndarray::{ArrayD, IxDyn}};
//!
//! # fn main() -> xarray_rs::Result<()> {
//! let data = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |ix| (ix[0] * 100 + ix[1] * 10 + ix[2]) as i32);
//! let mut v = Variable::new(["time", "lat", "lon"], data)?;
//!
//! // Python-style slices, by name or by position. Results are views.
//! let last = isel!(v[time = -1, lon = ::-2])?;
//! assert_eq!(last.dims(), ["lat", "lon"]);
//! assert_eq!(last.data::<i32>()?[[0, 0]], 103);
//!
//! // Mutable views write through to the owner.
//! isel!(mut v[0, .., 1:3])?.data_mut::<i32>()?.fill(-1);
//! assert_eq!(v.data::<i32>()?[[0, 2, 1]], -1);
//! # Ok(())
//! # }
//! ```

pub use xr_core::*;
pub use xr_macros::{isel, sel};
/// Runtime units, CF calendars and dates (the `xr-units` crate).
pub use xr_units as units;
pub use xr_units::{Calendar, CfDate, Unit, UnitError, UnitPolicy};

pub mod io;

/// Any error from xarray-rs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Data-model errors (dimensions, indexing, arithmetic).
    #[error(transparent)]
    Core(#[from] CoreError),
    /// Unit and calendar errors.
    #[error(transparent)]
    Unit(#[from] UnitError),
    /// File I/O and format errors.
    #[error(transparent)]
    Io(#[from] xr_io::IoError),
    /// CF decoding and encoding errors.
    #[error(transparent)]
    Cf(#[from] xr_cf::CfError),
}

#[doc(hidden)]
pub mod __rt {
    pub use xr_core::macro_support::*;
}
