//! Pure-Rust compression codecs for scientific array formats, shared by
//! [annoray-hdf5](https://docs.rs/annoray-hdf5) (HDF5 filters) and
//! [annoray-zarr](https://docs.rs/annoray-zarr) (Zarr codecs).
//!
//! - [`blosc`]: Blosc version-1 frames (BloscLZ, LZ4, zlib, Zstandard; byte and bit shuffle),
//!   decode and encode.
//! - [`bzip2`]: bzip2 streams, decode.
//! - [`zlib`]: zlib and gzip streams (raw deflate inside), decode and encode.
//! - [`zstd`], [`lz4`]: Zstandard frames and LZ4 blocks, behind the `zstd` and `lz4`
//!   features (pure Rust: `ruzstd`, `lz4_flex`).
//! - [`shuffle`], [`bitshuffle`]: byte and bit transpositions; [`crc`]: CRC-32 and CRC-32C.
//!
//! Input is untrusted: reads are bounds-checked, size arithmetic is checked, and every decoder
//! takes a limit on its output so that a corrupt or hostile stream cannot exhaust memory.
//! Malformed input yields [`Error`], never a panic.

#![deny(missing_docs)]

pub mod bits;
pub mod bitshuffle;
pub mod blosc;
pub mod bzip2;
mod ck;
pub mod crc;
pub mod lz4;
pub mod shuffle;
pub mod zlib;
pub mod zstd;

pub use blosc::{BloscCodec, BloscShuffle};

/// Errors from encoding or decoding.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The input is malformed (or a checksum does not match).
    #[error("{0}: corrupt data")]
    Corrupt(String),
    /// The output would exceed the given limit, or a size does not fit the format.
    #[error("{0}: data larger than expected")]
    TooLarge(String),
    /// A codec, variant or feature this crate does not provide.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Result type of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;
