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
    /// A unit error (incompatible units in arithmetic or lookup, bad unit strings, …).
    #[error(transparent)]
    Unit(#[from] xr_units::UnitError),
    /// No variable or coordinate with this name.
    #[error("no variable or coordinate named {0:?}")]
    NoSuchVariable(Name),
    /// A name is already used by another variable or coordinate.
    #[error("{0:?} already exists")]
    NameConflict(Name),
    /// A dimension would have two different lengths.
    #[error("dimension {dim:?} has length {expected} but a new variable gives it length {found}")]
    DimLengthMismatch {
        /// The dimension.
        dim: Name,
        /// Its existing length.
        expected: usize,
        /// The conflicting length.
        found: usize,
    },
    /// A coordinate uses a dimension its data array does not have.
    #[error("coordinate {coord:?} has dimension {dim:?}, which the data does not")]
    CoordDimNotInData {
        /// The coordinate.
        coord: Name,
        /// The extra dimension.
        dim: Name,
    },
    /// Label-based selection needs a dimension coordinate (a 1-D coordinate named after the dimension).
    #[error("dimension {0:?} has no coordinate to select labels from")]
    NoIndex(Name),
    /// A label was not found along a dimension.
    #[error("label {label} not found along dimension {dim:?}")]
    LabelNotFound {
        /// The dimension.
        dim: Name,
        /// The label, formatted.
        label: String,
    },
    /// The operation needs a sorted (monotonic) coordinate.
    #[error("the coordinate of dimension {0:?} is not monotonic")]
    NotMonotonic(Name),
    /// A label cannot be used with this coordinate.
    #[error("invalid label for dimension {dim:?}: {reason}")]
    InvalidLabel {
        /// The dimension.
        dim: Name,
        /// Why.
        reason: String,
    },
    /// Two operands disagree on the coordinate labels of a shared dimension.
    #[error("coordinates along dimension {0:?} differ; align the operands first")]
    NotAligned(Name),
    /// An operation is not defined for this element type.
    #[error("{op} is not supported for dtype {dtype}")]
    UnsupportedDType {
        /// The operation.
        op: &'static str,
        /// The element type.
        dtype: DType,
    },
    /// Reading lazily loaded data failed.
    #[error("reading data failed: {0}")]
    Read(String),
    /// Mutation needs the data in memory: call `load()` first.
    #[error("variable data is not loaded; call load() before modifying it")]
    NotLoaded,
    /// Integer arithmetic overflowed.
    #[error("integer overflow in {0}")]
    ArithmeticOverflow(&'static str),
}
