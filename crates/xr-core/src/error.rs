//! Error types for `xr-core`.

use crate::{DType, Name};

/// Result type used throughout `xr-core`.
pub type Result<T, E = CoreError> = core::result::Result<T, E>;

/// Errors raised while turning indexers into concrete axis selections.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IndexError {
    /// A scalar index fell outside its dimension.
    #[error("index {index} is out of bounds for dimension {dim:?} of length {len}")]
    OutOfBounds {
        /// Dimension being indexed.
        dim: Name,
        /// The index as given (before negative wrapping).
        index: isize,
        /// Length of the dimension.
        len: usize,
    },
    /// A slice had `step == 0`.
    #[error("slice step cannot be zero")]
    ZeroStep,
    /// An index or slice bound did not fit in `isize`, or arithmetic on it overflowed.
    #[error("index value does not fit in isize")]
    Overflow,
    /// More positional indexers than dimensions.
    #[error("too many indexers: {given} given for {ndim} dimension(s)")]
    TooManyIndexers {
        /// Number of non-ellipsis indexers given.
        given: usize,
        /// Number of dimensions of the indexed object.
        ndim: usize,
    },
    /// More than one `..` in a positional indexer list.
    #[error("an ellipsis (`..`) may appear at most once")]
    MultipleEllipsis,
    /// `..` used as the indexer of a named dimension.
    #[error("an ellipsis (`..`) is only valid in positional indexing")]
    EllipsisInNamed,
    /// The same dimension was indexed twice in one call.
    #[error("dimension {0:?} is indexed more than once")]
    DuplicateDim(Name),
    /// A string could not be parsed as an indexer.
    #[error("cannot parse {input:?} as an indexer: {reason}")]
    Parse {
        /// The rejected input.
        input: String,
        /// Why it was rejected.
        reason: &'static str,
    },
}

/// Errors raised by the core data model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CoreError {
    /// The number of dimension names does not match the array's dimensionality.
    #[error("{ndim}-dimensional data was given {names} dimension name(s)")]
    DimCountMismatch {
        /// Dimensionality of the data.
        ndim: usize,
        /// Number of names given.
        names: usize,
    },
    /// The same dimension name appears twice on one variable.
    #[error("duplicate dimension name {0:?}")]
    DuplicateDim(Name),
    /// No dimension with this name exists.
    #[error("no dimension named {0:?}")]
    NoSuchDim(Name),
    /// Typed access requested a different element type than the data holds.
    #[error("dtype mismatch: expected {expected}, found {found}")]
    DTypeMismatch {
        /// The requested element type.
        expected: DType,
        /// The element type actually stored.
        found: DType,
    },
    /// An indexing error.
    #[error(transparent)]
    Index(#[from] IndexError),
}
