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
    #[error("units {lhs} and {rhs} differ in scale; convert one first")]
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
    /// A unit of another unit library that this one cannot represent.
    #[error("unsupported: {0}")]
    Unsupported(String),
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
    /// A date is outside the range of an [`Instant`](crate::Instant) (1677-09-21 to
    /// 2262-04-11), or is missing.
    #[error("{0} is outside the range of datetime64[ns] (1677-09-21 to 2262-04-11)")]
    InstantRange(String),
    /// A time zone name is neither `UTC`, an offset such as `+05:30`, nor an IANA zone known
    /// to the time zone database.
    #[error(
        "unknown time zone {0:?}: expected UTC, an offset such as +05:30, or an IANA name such as America/New_York"
    )]
    UnknownTimeZone(String),
    /// A wall-clock time is ambiguous (it occurs twice, as clocks go back) or does not exist
    /// (it is skipped, as clocks go forward) in a time zone.
    #[error(
        "{time} is {reason} in {zone}; choose how to resolve it (compatible, earlier or later)"
    )]
    LocalTime {
        /// The wall-clock time.
        time: String,
        /// The zone.
        zone: String,
        /// `"ambiguous"` or `"skipped"`.
        reason: &'static str,
    },
    /// An encoded date is not a whole number of time steps from the reference.
    #[error("{date} is not a whole number of {step} from the reference date")]
    NotWholeSteps {
        /// The date, formatted.
        date: String,
        /// The step unit.
        step: String,
    },
}
