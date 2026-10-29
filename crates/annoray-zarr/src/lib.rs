//! Pure-Rust Zarr v2 and v3 reader and writer for local directory stores; the Zarr layer of
//! [annoray](https://docs.rs/annoray), independent of it.
//!
//! - **Formats:** Zarr v2 (`.zarray`/`.zgroup`/`.zattrs`; `C` and `F` order; `.` or `/`
//!   dimension separators) and v3 (`zarr.json`; regular chunk grids; `default` and `v2` chunk
//!   key encodings; sharding).
//! - **Types:** booleans, integers, floats, fixed-width bytes and UCS-4 text, variable-length
//!   UTF-8 text.
//! - **Codecs:** `bytes` (either byte order), `transpose`, `vlen-utf8`, `sharding_indexed`,
//!   `blosc`, `gzip`, `zstd`, `crc32c`, and numcodecs' `zlib`, `lz4`, `bz2` (read) and `shuffle`
//!   (also as v3 `numcodecs.*` codecs). Zstandard and LZ4 need the `zstd`/`lz4` features.
//! - **Reading** decodes only the chunks a region touches; missing chunks are the fill value.
//!   **Writing** goes chunk by chunk (so data need not be memory-resident), skips chunks that
//!   are all fill value, and can write consolidated metadata.
//! - Stores and chunks are untrusted: keys cannot leave the store, decoded sizes are bounded,
//!   and malformed data gives an [`Error`], not a panic. Dimension names follow xarray's
//!   conventions (v2 `_ARRAY_DIMENSIONS`, v3 `dimension_names`).

#![deny(missing_docs)]

mod array;
mod codec;
mod data;
mod group;
mod json;
mod store;

pub use annoray_codecs::{BloscCodec, BloscShuffle};
pub use array::{Array, ArrayBuilder, Compressor};
pub use codec::{Codec, Sharding};
pub use data::{ArrayData, DType, Endian, Fill};
pub use group::{Group, Node};
pub use json::{float_value, special_float};
/// JSON values (attributes).
pub use serde_json::{Map, Number, Value};
pub use store::Store;

/// The Zarr format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZarrFormat {
    /// Version 2.
    V2,
    /// Version 3 (the default for new stores).
    #[default]
    V3,
}

/// Errors from reading or writing Zarr.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The file system reported an error.
    #[error("io: {0}")]
    Io(String),
    /// Metadata or a chunk is malformed, or values do not fit the array.
    #[error("zarr: {0}")]
    Invalid(String),
    /// Something this crate does not read or write.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// No such group or array.
    #[error("not found: {0}")]
    NotFound(String),
}

impl From<annoray_codecs::Error> for Error {
    fn from(e: annoray_codecs::Error) -> Self {
        match e {
            annoray_codecs::Error::Unsupported(m) => Self::Unsupported(m),
            other => Self::Invalid(other.to_string()),
        }
    }
}

/// Result type of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

pub(crate) fn bad(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
