//! What a dataset must hold, stated once: its variables (dimensions, element types, units,
//! sizes, attributes) and attributes. [`Schema::check`] reports every way a dataset differs,
//! and [`#[derive(Schema)]`](macro@Schema) makes a struct of typed views from a dataset that
//! matches.
//!
//! ```
//! use annoray::dims::Length;
//! use annoray::ndarray::{ArrayD, IxDyn};
//! use annoray::schema::{Schema, Var};
//! use annoray::{Dataset, Variable};
//!
//! let schema = Schema::new()
//!     .attr_eq("format", "prism-map")
//!     .coord("x", Var::of::<f64>(["x"]).dimension::<Length>())
//!     .var("intensity", Var::of::<f32>(["x"]).unit("1"))
//!     .optional("mask", Var::of::<bool>(["x"]));
//!
//! let x = Variable::new(["x"], ArrayD::from_elem(IxDyn(&[3]), 0.5))?.with_unit("s".parse()?);
//! let ds = Dataset::new().with_coord("x", x)?;
//! let err = schema.check(&ds).unwrap_err();
//! assert_eq!(
//!     err.to_string(),
//!     "the dataset does not match the schema:\n  \
//!      attribute format: missing\n  \
//!      x: unit s is not of dimension m\n  \
//!      intensity: missing"
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::fmt;

use ndarray::ArrayView;

use crate::dims::Dimensional;
use crate::units::Dimension;
use crate::{AttrValue, Attrs, DType, DatasetBase, Element, Name, Repr, Unit, VariableBase};

pub use annoray_macros::Schema;

/// What a variable's unit must be.
#[derive(Debug, Clone, PartialEq)]
enum UnitRule {
    Any,
    NoUnit,
    /// Equal to this unit (as units compare: `km` equals `1000 m`).
    Exact(String),
    /// Convertible to this unit.
    Convertible(String),
    /// Of this dimension.
    Dimension(Dimension),
}

/// What an attribute must be.
#[derive(Debug, Clone, PartialEq)]
enum AttrRule {
    Present,
    Text,
    Number,
    Eq(AttrValue),
    In(Vec<AttrValue>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Size {
    Exact(usize),
    Min(usize),
}

/// What one variable must be: its dimensions (or only how many), element type, unit, sizes and
/// attributes. Each is unchecked unless set.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Var {
    dims: Option<Vec<Name>>,
    ndim: Option<usize>,
    dtype: Option<DType>,
    unit: UnitRule,
    sizes: Vec<(Name, Size)>,
    attrs: Vec<(Name, AttrRule)>,
}

impl Var {
    /// A variable over `dims`, in this order.
    pub fn new<D: Into<Name>>(dims: impl IntoIterator<Item = D>) -> Self {
        Self {
            dims: Some(dims.into_iter().map(Into::into).collect()),
            ..Self::any()
        }
    }

    /// A variable of element type `T` over `dims`.
    pub fn of<T: Element>(dims: impl IntoIterator<Item = impl Into<Name>>) -> Self {
        Self::new(dims).dtype(T::DTYPE)
    }

    /// Any variable (nothing checked until set).
    pub fn any() -> Self {
        Self {
            dims: None,
            ndim: None,
            dtype: None,
            unit: UnitRule::Any,
            sizes: Vec::new(),
            attrs: Vec::new(),
        }
    }

    /// Of element type `dtype` (exactly: no casts).
    pub const fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// With `n` dimensions, whatever their names.
    pub const fn ndim(mut self, n: usize) -> Self {
        self.ndim = Some(n);
        self
    }

    /// In unit `unit` (equal as units compare: `km` and `1000 m` are equal; no unit is not
    /// `1`). A unit that does not parse is a problem of the schema, reported by `check`.
    pub fn unit(mut self, unit: &str) -> Self {
        self.unit = UnitRule::Exact(unit.to_owned());
        self
    }

    /// In a unit convertible to `unit`.
    pub fn convertible(mut self, unit: &str) -> Self {
        self.unit = UnitRule::Convertible(unit.to_owned());
        self
    }

    /// In a unit of dimension `D` ([`dims`](crate::dims): `Length`, `Velocity`, …).
    pub fn dimension<D: Dimensional>(self) -> Self {
        self.dimension_of(D::DIMENSION)
    }

    /// In a unit of dimension `dim`.
    pub fn dimension_of(mut self, dim: Dimension) -> Self {
        self.unit = UnitRule::Dimension(dim);
        self
    }

    /// Without a unit.
    pub fn no_unit(mut self) -> Self {
        self.unit = UnitRule::NoUnit;
        self
    }

    /// Dimension `dim` of length `n`.
    pub fn size(mut self, dim: impl Into<Name>, n: usize) -> Self {
        self.sizes.push((dim.into(), Size::Exact(n)));
        self
    }

    /// Dimension `dim` of length `n` or more.
    pub fn min_size(mut self, dim: impl Into<Name>, n: usize) -> Self {
        self.sizes.push((dim.into(), Size::Min(n)));
        self
    }

    /// With attribute `name`.
    pub fn attr(mut self, name: impl Into<Name>) -> Self {
        self.attrs.push((name.into(), AttrRule::Present));
        self
    }

    /// With attribute `name` equal to `value`.
    pub fn attr_eq(mut self, name: impl Into<Name>, value: impl Into<AttrValue>) -> Self {
        self.attrs.push((name.into(), AttrRule::Eq(value.into())));
        self
    }

    /// With attribute `name` equal to one of `values`.
    pub fn attr_in<V: Into<AttrValue>>(
        mut self,
        name: impl Into<Name>,
        values: impl IntoIterator<Item = V>,
    ) -> Self {
        let values = values.into_iter().map(Into::into).collect();
        self.attrs.push((name.into(), AttrRule::In(values)));
        self
    }

    /// The same, with what `other` sets added (for the derive: the field type's element type
    /// and dimensionality, then the field's attributes).
    #[doc(hidden)]
    pub fn and(mut self, other: Self) -> Self {
        self.dims = other.dims.or(self.dims);
        self.ndim = other.ndim.or(self.ndim);
        self.dtype = other.dtype.or(self.dtype);
        if other.unit != UnitRule::Any {
            self.unit = other.unit;
        }
        self.sizes.extend(other.sizes);
        self.attrs.extend(other.attrs);
        self
    }

    fn check<S: Repr>(&self, name: &str, v: &VariableBase<S>, out: &mut Vec<Problem>) {
        let mut problem = |kind| out.push(Problem::new(name, kind));
        if let Some(dims) = &self.dims
            && v.dims() != dims.as_slice()
        {
            problem(ProblemKind::Dims {
                expected: dims.clone(),
                found: v.dims().to_vec(),
            });
        }
        if let Some(n) = self.ndim.filter(|n| *n != v.dims().len()) {
            problem(ProblemKind::Ndim {
                expected: n,
                found: v.dims().len(),
            });
        }
        if let Some(d) = self.dtype.filter(|d| *d != v.dtype()) {
            problem(ProblemKind::DType {
                expected: d,
                found: v.dtype(),
            });
        }
        let found = v.unit().map(ToString::to_string);
        let unit_problem = |expected: String| ProblemKind::Unit {
            expected,
            found: found.clone(),
        };
        match &self.unit {
            UnitRule::Any => {}
            UnitRule::NoUnit => {
                if v.unit().is_some() {
                    problem(unit_problem("no unit".into()));
                }
            }
            UnitRule::Exact(s) | UnitRule::Convertible(s) => match s.parse::<Unit>() {
                Err(e) => problem(ProblemKind::BadSchema(format!("unit {s:?}: {e}"))),
                Ok(want) => {
                    let exact = matches!(self.unit, UnitRule::Exact(_));
                    let ok = v.unit().is_some_and(|u| {
                        if exact {
                            *u == want
                        } else {
                            u.converter_to(&want).is_ok()
                        }
                    });
                    if !ok {
                        problem(unit_problem(if exact {
                            format!("unit {s}")
                        } else {
                            format!("a unit convertible to {s}")
                        }));
                    }
                }
            },
            UnitRule::Dimension(d) => {
                if v.unit().is_none_or(|u| u.dim() != *d) {
                    problem(unit_problem(format!("a unit of dimension {d}")));
                }
            }
        }
        for (dim, size) in &self.sizes {
            let n = v
                .dims()
                .iter()
                .position(|d| d == dim)
                .and_then(|i| v.shape().get(i).copied());
            let Some(n) = n else {
                problem(ProblemKind::BadSize {
                    dim: dim.clone(),
                    expected: describe_size(*size),
                    found: None,
                });
                continue;
            };
            let ok = match size {
                Size::Exact(m) => n == *m,
                Size::Min(m) => n >= *m,
            };
            if !ok {
                problem(ProblemKind::BadSize {
                    dim: dim.clone(),
                    expected: describe_size(*size),
                    found: Some(n),
                });
            }
        }
        check_attrs(&self.attrs, v.attrs(), Some(name), out);
    }
}

fn describe_size(s: Size) -> String {
    match s {
        Size::Exact(n) => format!("{n}"),
        Size::Min(n) => format!("at least {n}"),
    }
}

fn check_attrs(
    rules: &[(Name, AttrRule)],
    attrs: &Attrs,
    of: Option<&str>,
    out: &mut Vec<Problem>,
) {
    for (name, rule) in rules {
        let subject = match of {
            Some(v) => format!("{v} attribute {name}"),
            None => format!("attribute {name}"),
        };
        let Some(a) = attrs.get(name.as_str()) else {
            out.push(Problem::new(subject, ProblemKind::Missing));
            continue;
        };
        let expected = match rule {
            AttrRule::Present => None,
            AttrRule::Text => (!matches!(a, AttrValue::Str(_))).then(|| "text".to_owned()),
            AttrRule::Number => attr_number(a).is_none().then(|| "a number".to_owned()),
            AttrRule::Eq(want) => (a != want).then(|| show_attr(want)),
            AttrRule::In(wants) => (!wants.contains(a)).then(|| {
                let shown: Vec<String> = wants.iter().map(show_attr).collect();
                format!("one of {}", shown.join(", "))
            }),
        };
        if let Some(expected) = expected {
            out.push(Problem::new(
                subject,
                ProblemKind::AttrValue {
                    expected,
                    found: show_attr(a),
                },
            ));
        }
    }
}

/// An attribute as written in messages: text quoted, numbers as numbers.
fn show_attr(a: &AttrValue) -> String {
    fn list<T: fmt::Display>(v: &[T]) -> String {
        match v {
            [one] => one.to_string(),
            many => format!(
                "[{}]",
                many.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
    match a {
        AttrValue::Str(s) => format!("{s:?}"),
        AttrValue::Strs(v) => format!("{v:?}"),
        AttrValue::I8(v) => list(v),
        AttrValue::I16(v) => list(v),
        AttrValue::I32(v) => list(v),
        AttrValue::I64(v) => list(v),
        AttrValue::U8(v) => list(v),
        AttrValue::U16(v) => list(v),
        AttrValue::U32(v) => list(v),
        AttrValue::U64(v) => list(v),
        AttrValue::F32(v) => list(v),
        AttrValue::F64(v) => list(v),
    }
}

/// A one-value numeric attribute as `f64` (integers exactly up to 2⁵³).
fn attr_number(a: &AttrValue) -> Option<f64> {
    #[allow(clippy::cast_precision_loss)]
    Some(match a {
        AttrValue::I8(v) => f64::from(*one(v)?),
        AttrValue::I16(v) => f64::from(*one(v)?),
        AttrValue::I32(v) => f64::from(*one(v)?),
        AttrValue::I64(v) => *one(v)? as f64,
        AttrValue::U8(v) => f64::from(*one(v)?),
        AttrValue::U16(v) => f64::from(*one(v)?),
        AttrValue::U32(v) => f64::from(*one(v)?),
        AttrValue::U64(v) => *one(v)? as f64,
        AttrValue::F32(v) => f64::from(*one(v)?),
        AttrValue::F64(v) => *one(v)?,
        _ => return None,
    })
}

fn one<T>(v: &[T]) -> Option<&T> {
    match v {
        [x] => Some(x),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    Data,
    Coord,
}

#[derive(Debug, Clone, PartialEq)]
struct Entry {
    name: Name,
    place: Place,
    required: bool,
    var: Var,
}

/// What a dataset must hold. Variables not named are allowed unless the schema is
/// [`closed`](Self::closed).
#[derive(Debug, Clone, Default, PartialEq)]
#[must_use]
pub struct Schema {
    attrs: Vec<(Name, AttrRule)>,
    vars: Vec<Entry>,
    closed: bool,
}

impl Schema {
    /// A schema that any dataset matches.
    pub fn new() -> Self {
        Self::default()
    }

    fn entry(mut self, name: Name, place: Place, required: bool, var: Var) -> Self {
        self.vars.push(Entry {
            name,
            place,
            required,
            var,
        });
        self
    }

    /// A data variable `name`, as `var` says.
    pub fn var(self, name: impl Into<Name>, var: Var) -> Self {
        self.entry(name.into(), Place::Data, true, var)
    }

    /// A data variable `name` that may be absent, as `var` says when present.
    pub fn optional(self, name: impl Into<Name>, var: Var) -> Self {
        self.entry(name.into(), Place::Data, false, var)
    }

    /// A coordinate `name`, as `var` says.
    pub fn coord(self, name: impl Into<Name>, var: Var) -> Self {
        self.entry(name.into(), Place::Coord, true, var)
    }

    /// A coordinate `name` that may be absent, as `var` says when present.
    pub fn optional_coord(self, name: impl Into<Name>, var: Var) -> Self {
        self.entry(name.into(), Place::Coord, false, var)
    }

    /// Dataset attribute `name` present.
    pub fn attr(mut self, name: impl Into<Name>) -> Self {
        self.attrs.push((name.into(), AttrRule::Present));
        self
    }

    /// Dataset attribute `name` equal to `value`.
    pub fn attr_eq(mut self, name: impl Into<Name>, value: impl Into<AttrValue>) -> Self {
        self.attrs.push((name.into(), AttrRule::Eq(value.into())));
        self
    }

    /// Dataset attribute `name` equal to one of `values` (a format's supported versions, …).
    pub fn attr_in<V: Into<AttrValue>>(
        mut self,
        name: impl Into<Name>,
        values: impl IntoIterator<Item = V>,
    ) -> Self {
        let values = values.into_iter().map(Into::into).collect();
        self.attrs.push((name.into(), AttrRule::In(values)));
        self
    }

    /// Dataset attribute `name` present and text (for the derive's `&str` and `String`
    /// fields).
    #[doc(hidden)]
    pub fn attr_text(mut self, name: impl Into<Name>) -> Self {
        self.attrs.push((name.into(), AttrRule::Text));
        self
    }

    /// Dataset attribute `name` present and one number (for the derive's numeric fields).
    #[doc(hidden)]
    pub fn attr_number(mut self, name: impl Into<Name>) -> Self {
        self.attrs.push((name.into(), AttrRule::Number));
        self
    }

    /// No variables or coordinates but those named.
    pub fn closed(mut self) -> Self {
        self.closed = true;
        self
    }

    /// Every way `ds` differs from the schema, in the schema's order (empty if it matches).
    /// Only metadata is read: lazy values stay unread.
    #[must_use]
    pub fn problems<S: Repr>(&self, ds: &DatasetBase<S>) -> Vec<Problem> {
        let mut out = Vec::new();
        check_attrs(&self.attrs, ds.attrs(), None, &mut out);
        for e in &self.vars {
            let (found, elsewhere) = match e.place {
                Place::Data => {
                    let v = ds.data_vars().get(e.name.as_str());
                    if let Some(v) = v {
                        e.var.check(&e.name, v, &mut out);
                    }
                    (v.is_some(), ds.coords().contains_key(e.name.as_str()))
                }
                Place::Coord => {
                    let v = ds.coords().get(e.name.as_str());
                    if let Some(v) = v {
                        e.var.check(&e.name, v, &mut out);
                    }
                    (v.is_some(), ds.data_vars().contains_key(e.name.as_str()))
                }
            };
            if found {
                continue;
            }
            if elsewhere {
                out.push(Problem::new(
                    e.name.as_str(),
                    ProblemKind::Misplaced {
                        expected_coord: e.place == Place::Coord,
                    },
                ));
            } else if e.required {
                out.push(Problem::new(e.name.as_str(), ProblemKind::Missing));
            }
        }
        if self.closed {
            for name in ds.coords().keys().chain(ds.data_vars().keys()) {
                if !self.vars.iter().any(|e| e.name == *name) {
                    out.push(Problem::new(name.as_str(), ProblemKind::Unexpected));
                }
            }
        }
        // Two rules can find the same problem (an attribute two of them require).
        let mut seen: Vec<Problem> = Vec::with_capacity(out.len());
        for p in out {
            if !seen.contains(&p) {
                seen.push(p);
            }
        }
        seen
    }

    /// Whether `ds` matches the schema.
    ///
    /// # Errors
    /// [`SchemaError`] with every problem found.
    pub fn check<S: Repr>(&self, ds: &DatasetBase<S>) -> Result<(), SchemaError> {
        let problems = self.problems(ds);
        if problems.is_empty() {
            Ok(())
        } else {
            Err(SchemaError { problems })
        }
    }
}

/// One way a dataset differs from a [`Schema`](struct@Schema).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Problem {
    /// What it concerns: a variable (`intensity`), a dataset attribute (`attribute format`)
    /// or a variable's attribute (`intensity attribute long_name`).
    pub subject: String,
    /// What is wrong.
    pub kind: ProblemKind,
}

impl Problem {
    fn new(subject: impl Into<String>, kind: ProblemKind) -> Self {
        Self {
            subject: subject.into(),
            kind,
        }
    }
}

/// What is wrong, in a [`Problem`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ProblemKind {
    /// A required variable or attribute is absent.
    Missing,
    /// A variable is a coordinate where a data variable was expected, or the other way round.
    Misplaced {
        /// Whether a coordinate was expected.
        expected_coord: bool,
    },
    /// A variable a closed schema does not name.
    Unexpected,
    /// Other dimensions.
    Dims {
        /// The dimensions expected.
        expected: Vec<Name>,
        /// The variable's.
        found: Vec<Name>,
    },
    /// Another number of dimensions.
    Ndim {
        /// The number expected.
        expected: usize,
        /// The variable's.
        found: usize,
    },
    /// Another element type.
    DType {
        /// The type expected.
        expected: DType,
        /// The variable's.
        found: DType,
    },
    /// Another unit, or none.
    Unit {
        /// What was expected (`unit km`, `a unit of dimension m`, `no unit`).
        expected: String,
        /// The variable's unit.
        found: Option<String>,
    },
    /// A dimension of another length (or that the variable lacks).
    BadSize {
        /// The dimension.
        dim: Name,
        /// The length expected (`10`, `at least 2`).
        expected: String,
        /// The variable's, if it has the dimension.
        found: Option<usize>,
    },
    /// An attribute of another value or type.
    AttrValue {
        /// What was expected.
        expected: String,
        /// The attribute's value.
        found: String,
    },
    /// The schema itself is wrong (a unit that does not parse).
    BadSchema(String),
}

fn dim_list(d: &[Name]) -> String {
    format!(
        "({})",
        d.iter().map(Name::as_str).collect::<Vec<_>>().join(", ")
    )
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.subject)?;
        match &self.kind {
            ProblemKind::Missing => f.write_str("missing"),
            ProblemKind::Misplaced {
                expected_coord: true,
            } => f.write_str("is a data variable, expected a coordinate"),
            ProblemKind::Misplaced {
                expected_coord: false,
            } => f.write_str("is a coordinate, expected a data variable"),
            ProblemKind::Unexpected => f.write_str("not in the schema"),
            ProblemKind::Dims { expected, found } => write!(
                f,
                "dimensions {}, expected {}",
                dim_list(found),
                dim_list(expected)
            ),
            ProblemKind::Ndim { expected, found } => {
                write!(f, "{found} dimensions, expected {expected}")
            }
            ProblemKind::DType { expected, found } => {
                write!(f, "element type {found}, expected {expected}")
            }
            ProblemKind::Unit {
                expected,
                found: Some(u),
            } => match expected.strip_prefix("a unit of dimension ") {
                Some(d) => write!(f, "unit {u} is not of dimension {d}"),
                None => write!(f, "unit {u}, expected {expected}"),
            },
            ProblemKind::Unit {
                expected,
                found: None,
            } => write!(f, "no unit, expected {expected}"),
            ProblemKind::BadSize {
                dim,
                expected,
                found: Some(n),
            } => write!(f, "dimension {dim} of length {n}, expected {expected}"),
            ProblemKind::BadSize {
                dim,
                expected,
                found: None,
            } => write!(f, "no dimension {dim} (expected of length {expected})"),
            ProblemKind::AttrValue { expected, found } => {
                write!(f, "{found}, expected {expected}")
            }
            ProblemKind::BadSchema(why) => write!(f, "the schema is wrong: {why}"),
        }
    }
}

/// A dataset does not match a [`Schema`](struct@Schema): every [`Problem`] found.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub struct SchemaError {
    /// The problems, in the schema's order.
    pub problems: Vec<Problem>,
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the dataset does not match the schema:")?;
        for p in &self.problems {
            write!(f, "\n  {p}")?;
        }
        Ok(())
    }
}

// ----- the derive's field types -----

/// What the type of a variable field of a [`#[derive(Schema)]`](macro@Schema) struct
/// requires of the variable: an `ArrayView` its element type and dimensionality, a
/// `&VariableBase` nothing; `Option` of either makes the variable optional.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be a variable field of a schema struct",
    label = "not an `ArrayView`, a `&VariableBase` or an `Option` of one",
    note = "an attribute field is marked `#[schema(attr)]`: `&str`, `String`, `f64`, `i64`, `&AttrValue` or an `Option` of one"
)]
pub trait FieldSpec {
    /// Whether the variable may be absent.
    const OPTIONAL: bool = false;

    /// What the type requires of the variable.
    fn spec() -> Var;
}

/// A variable field of a [`#[derive(Schema)]`](macro@Schema) struct: an `ArrayView` (a
/// borrowed view of the values), a `&VariableBase` (unit and values together), or `Option` of
/// either, from a dataset of representation `S`.
pub trait VarField<'a, S: Repr>: FieldSpec + Sized {
    /// The field from the variable (after [`Schema::check`] passed).
    ///
    /// # Errors
    /// [`CoreError`](crate::CoreError) if the values cannot be read (a lazy variable is read here);
    /// [`SchemaError`] if the variable is absent and the field not optional, or of another
    /// type.
    fn extract(v: Option<&'a VariableBase<S>>, name: &str) -> Result<Self, crate::Error>;
}

/// One problem, as an error.
fn one_problem(subject: impl Into<String>, kind: ProblemKind) -> crate::Error {
    crate::Error::Schema(SchemaError {
        problems: vec![Problem::new(subject, kind)],
    })
}

fn present<'a, S: Repr>(
    v: Option<&'a VariableBase<S>>,
    name: &str,
) -> Result<&'a VariableBase<S>, crate::Error> {
    v.ok_or_else(|| one_problem(name, ProblemKind::Missing))
}

impl<T: Element, D: ndarray::Dimension> FieldSpec for ArrayView<'_, T, D> {
    fn spec() -> Var {
        let v = Var::any().dtype(T::DTYPE);
        match D::NDIM {
            Some(n) => v.ndim(n),
            None => v,
        }
    }
}

impl<'a, S: Repr, T: Element, D: ndarray::Dimension> VarField<'a, S> for ArrayView<'a, T, D> {
    fn extract(v: Option<&'a VariableBase<S>>, name: &str) -> Result<Self, crate::Error> {
        let v = present(v, name)?;
        if v.dtype() != T::DTYPE {
            return Err(one_problem(
                name,
                ProblemKind::DType {
                    expected: T::DTYPE,
                    found: v.dtype(),
                },
            ));
        }
        let a = v.data::<T>()?;
        let found = a.ndim();
        a.into_dimensionality::<D>().map_err(|_| {
            one_problem(
                name,
                ProblemKind::Ndim {
                    expected: D::NDIM.unwrap_or(found),
                    found,
                },
            )
        })
    }
}

impl<S: Repr> FieldSpec for &VariableBase<S> {
    fn spec() -> Var {
        Var::any()
    }
}

impl<'a, S: Repr> VarField<'a, S> for &'a VariableBase<S> {
    fn extract(v: Option<&'a VariableBase<S>>, name: &str) -> Result<Self, crate::Error> {
        present(v, name)
    }
}

impl<F: FieldSpec> FieldSpec for Option<F> {
    const OPTIONAL: bool = true;

    fn spec() -> Var {
        F::spec()
    }
}

impl<'a, S: Repr, F: VarField<'a, S>> VarField<'a, S> for Option<F> {
    fn extract(v: Option<&'a VariableBase<S>>, name: &str) -> Result<Self, crate::Error> {
        v.map(|v| F::extract(Some(v), name)).transpose()
    }
}

/// A field of a [`#[derive(Schema)]`](macro@Schema) struct that is a dataset attribute:
/// `&str` or `String` (text), `f64` or `i64` (one number), `&AttrValue` (any), or `Option`
/// of one (an optional attribute).
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be an attribute field of a schema struct",
    label = "not `&str`, `String`, `f64`, `i64`, `&AttrValue` or an `Option` of one"
)]
pub trait AttrField<'a>: Sized {
    /// Whether the attribute may be absent.
    const OPTIONAL: bool = false;

    /// The schema with this attribute's requirement added.
    fn require(schema: Schema, name: &str) -> Schema;

    /// The field from the attribute (after [`Schema::check`] passed).
    ///
    /// # Errors
    /// [`SchemaError`] if it is absent or of another type.
    fn extract(a: Option<&'a AttrValue>, name: &str) -> Result<Self, crate::Error>;
}

fn wrong(name: &str, what: &str, a: Option<&AttrValue>) -> crate::Error {
    let subject = format!("attribute {name}");
    match a {
        None => one_problem(subject, ProblemKind::Missing),
        Some(a) => one_problem(
            subject,
            ProblemKind::AttrValue {
                expected: what.to_owned(),
                found: show_attr(a),
            },
        ),
    }
}

impl<'a> AttrField<'a> for &'a str {
    fn require(schema: Schema, name: &str) -> Schema {
        schema.attr_text(name)
    }
    fn extract(a: Option<&'a AttrValue>, name: &str) -> Result<Self, crate::Error> {
        match a {
            Some(AttrValue::Str(s)) => Ok(s),
            other => Err(wrong(name, "text", other)),
        }
    }
}

impl<'a> AttrField<'a> for String {
    fn require(schema: Schema, name: &str) -> Schema {
        schema.attr_text(name)
    }
    fn extract(a: Option<&'a AttrValue>, name: &str) -> Result<Self, crate::Error> {
        <&str>::extract(a, name).map(str::to_owned)
    }
}

impl<'a> AttrField<'a> for f64 {
    fn require(schema: Schema, name: &str) -> Schema {
        schema.attr_number(name)
    }
    fn extract(a: Option<&'a AttrValue>, name: &str) -> Result<Self, crate::Error> {
        a.and_then(attr_number)
            .ok_or_else(|| wrong(name, "a number", a))
    }
}

impl<'a> AttrField<'a> for i64 {
    fn require(schema: Schema, name: &str) -> Schema {
        schema.attr_number(name)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn extract(a: Option<&'a AttrValue>, name: &str) -> Result<Self, crate::Error> {
        let n = <f64>::extract(a, name)?;
        if n.fract() == 0.0 && n.abs() < 9.007_199_254_740_992e15 {
            Ok(n as i64)
        } else {
            Err(wrong(name, "an integer", a))
        }
    }
}

impl<'a> AttrField<'a> for &'a AttrValue {
    fn require(schema: Schema, name: &str) -> Schema {
        schema.attr(name)
    }
    fn extract(a: Option<&'a AttrValue>, name: &str) -> Result<Self, crate::Error> {
        a.ok_or_else(|| wrong(name, "present", None))
    }
}

impl<'a, F: AttrField<'a>> AttrField<'a> for Option<F> {
    const OPTIONAL: bool = true;
    fn require(schema: Schema, _: &str) -> Schema {
        schema
    }
    fn extract(a: Option<&'a AttrValue>, name: &str) -> Result<Self, crate::Error> {
        a.map(|a| F::extract(Some(a), name)).transpose()
    }
}
