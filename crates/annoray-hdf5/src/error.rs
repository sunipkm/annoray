//! Errors.

/// Errors from reading or writing HDF5.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The file system reported an error.
    #[error("io: {0}")]
    Io(String),
    /// The file is not valid HDF5, is corrupt, or is too large to write or load.
    #[error("hdf5: {0}")]
    Invalid(String),
    /// The file uses something this crate cannot read, or the data cannot be written.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A requested group or object does not exist.
    #[error("no object {0:?}")]
    NotFound(String),
}

/// Result type of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

pub(crate) fn bad(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

impl From<annoray_codecs::Error> for Error {
    fn from(e: annoray_codecs::Error) -> Self {
        match e {
            annoray_codecs::Error::Unsupported(m) => Self::Unsupported(m),
            other => Self::Invalid(other.to_string()),
        }
    }
}
