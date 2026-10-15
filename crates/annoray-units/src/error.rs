//! Error type for `annoray-units`.

/// Errors from unit parsing, unit arithmetic, conversion and calendar handling.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum UnitError {
    /// A unit string could not be parsed.
    #[error("cannot parse unit {input:?} at byte {pos}: {reason}")]
    Parse {
        /// The full input.
        input: String,
        /// Byte offset of the problem.
        pos: usize,
        /// What went wrong.
        reason: String,
    },
    /// An identifier is not a known unit, with or without a prefix.
    #[error("unknown unit {0:?}")]
    UnknownUnit(String),
    /// The two units measure different quantities.
    #[error("incompatible units: {lhs} and {rhs}")]
    Incompatible {
        /// Left-hand unit.
        lhs: String,
        /// Right-hand unit.
        rhs: String,
    },
    /// Same dimension, different scale, under the strict policy. Rescale explicitly.
    #[error(
        "units {lhs} and {rhs} differ in scale; rescale one explicitly (or use UnitPolicy::AutoRescale)"
    )]
    ScaleMismatch {
        /// Left-hand unit.
        lhs: String,
        /// Right-hand unit.
        rhs: String,
    },
    /// An operation is not defined for a unit of this kind (e.g. multiplying °C).
    #[error("{op} is not defined for {unit}")]
    UnsupportedOp {
        /// The operation.
        op: &'static str,
        /// The unit it was applied to.
        unit: String,
    },
    /// A root was taken whose result has fractional exponents or an irrational scale.
    #[error("the {n}th root of {unit} is not a unit with integer exponents")]
    InexactRoot {
        /// The root degree.
        n: i32,
        /// The unit.
        unit: String,
    },
    /// An exponent, scale or date component overflowed.
    #[error("arithmetic overflow")]
    Overflow,
    /// A calendar name is not one of the CF calendars.
    #[error("unknown calendar {0:?}")]
    UnknownCalendar(String),
    /// A date does not exist in its calendar.
    #[error("invalid date {date} in the {calendar} calendar")]
    InvalidDate {
        /// The date, formatted.
        date: String,
        /// The calendar name.
        calendar: &'static str,
    },
    /// The unit is not of the form `<time unit> since <date>`.
    #[error("{0} is not a time unit with a reference date (`<unit> since <date>`)")]
    NotAnEpoch(String),
    /// The step of an epoch unit is not a whole number of nanoseconds (e.g. `months since`).
    #[error("{0} cannot be used as a time step: it is not a fixed whole number of nanoseconds")]
    UnsupportedTimeStep(String),
    /// A floating-point time offset was NaN or infinite.
    #[error("time offset {0} is not finite")]
    NonFiniteTime(String),
    /// A date cannot be represented by chrono.
    #[error("{0} cannot be represented as a chrono date")]
    ChronoRange(String),
    /// An encoded date is not a whole number of time steps from the reference.
    #[error("{date} is not a whole number of {step} from the reference date")]
    NotWholeSteps {
        /// The date, formatted.
        date: String,
        /// The step unit.
        step: String,
    },
}
