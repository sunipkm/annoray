//! CF conventions for xarray-rs: decoding raw netCDF data into a [`Dataset`](xr_core::Dataset)
//! and encoding it back, following xarray's `decode_cf` / `encode_cf` behaviour.
//!
//! | On disk | In memory | Encoding keeps |
//! |---|---|---|
//! | `NC_CHAR` arrays | strings | the string-length dimension |
//! | `_Unsigned = "true"` signed ints | unsigned ints | `unsigned` |
//! | `dtype = "bool"` ints | `bool` | — |
//! | `_FillValue` / `missing_value` | NaN (ints become floats) | the fill values and on-disk dtype |
//! | `scale_factor` / `add_offset` | unpacked floats | the factors and packed dtype |
//! | `units = "<u> since <date>"`, `calendar` | dates (chrono or CF calendar) | units and calendar |
//! | other `units` | [`Unit`](xr_core::xr_units::Unit) on the variable | the original spelling (the unit's label) |
//! | `coordinates` attributes, dimension variables | coordinates | — |
//!
//! Encoding reverses all of this, choosing xarray's defaults when a variable has no encoding
//! (dates as integers in the coarsest exact unit, `_FillValue = NaN` for floats, `string{N}`
//! char dimensions in classic files), and adapting types to the target format.

mod decode;
mod encode;
mod num;

pub use decode::{DecodeOptions, Decoded, decode};
pub use encode::{EncodeOptions, encode};

use xr_core::Name;

/// Errors from CF decoding and encoding.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CfError {
    /// A variable cannot be decoded or encoded.
    #[error("variable {var:?}: {reason}")]
    Invalid {
        /// The variable.
        var: Name,
        /// Why.
        reason: String,
    },
    /// A data-model error.
    #[error(transparent)]
    Core(#[from] xr_core::CoreError),
    /// A unit or calendar error.
    #[error(transparent)]
    Unit(#[from] xr_core::xr_units::UnitError),
    /// An I/O or format error.
    #[error(transparent)]
    Io(#[from] xr_io::IoError),
}

/// Result type for CF operations.
pub type Result<T, E = CfError> = core::result::Result<T, E>;
