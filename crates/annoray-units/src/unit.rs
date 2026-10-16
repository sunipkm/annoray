//! [`Unit`]: a runtime unit with exact scale, kind (linear, affine, epoch, logarithmic) and label.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::time::{Calendar, EpochCodec, RefDate};
use crate::{BaseDim, Dimension, Rational, Scale, UnitError};

/// What kind of quantity a unit measures, beyond its dimension and scale.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnitKind {
    /// Ordinary ratio-scale unit: `value_SI = value · scale`.
    Linear,
    /// Interval scale with a zero offset, like °C: `value_SI = (value + offset) · scale`.
    Affine {
        /// Offset in this unit's own scale (273.15 for °C).
        offset: Rational,
    },
    /// Time since a reference date: `<scale> since <reference>`. Decode with [`Unit::epoch_codec`].
    Epoch(RefDate),
    /// Logarithmic ratio to a reference, like dB. `scale` multiplies the logarithm (0.1 for deci).
    Log(Box<LogRef>),
}

/// Base of a logarithmic unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum LogBase {
    Ten,
    Two,
    E,
}

/// The reference level of a logarithmic unit (`lg(re 1 mW)`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LogRef {
    /// Logarithm base.
    pub base: LogBase,
    /// Dimension of the reference quantity.
    pub reference_dim: Dimension,
    /// Scale of the reference quantity.
    pub reference_scale: Scale,
}

/// How to treat operands whose units have the same dimension but different scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnitPolicy {
    /// Refuse: the caller must rescale explicitly (the whippyunits rule).
    #[default]
    Strict,
    /// Convert the other operand into the left-hand unit.
    AutoRescale,
}

/// A value conversion `y = x · factor + offset`.
///
/// A factor that is exactly `1/n` for an integer `n` (m → km, min → h) is applied as a
/// division by `n`, which rounds correctly: `700 m` becomes exactly `0.7 km` (the float
/// `0.7`), where multiplying by the inexact `0.001` gives `0.7000000000000001`. Integer
/// factors are a single, correctly rounded multiplication already.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Converter {
    factor: f64,
    offset: f64,
    exact: Option<Rational>,
    /// `n` when the factor is exactly `1/n` (`n` an integer that `f64` holds exactly).
    divisor: Option<f64>,
}

/// Integers up to 2^53 are exact in `f64`.
const EXACT_INT: i128 = 1 << 53;

impl Converter {
    /// The identity conversion.
    pub const IDENTITY: Self = Self {
        factor: 1.0,
        offset: 0.0,
        exact: Some(Rational::ONE),
        divisor: None,
    };

    fn new(factor: f64, offset: f64, exact: Option<Rational>) -> Self {
        #[allow(clippy::cast_precision_loss)] // checked to be exact
        let divisor = exact
            .filter(|r| r.numer() == 1 && r.denom() > 1 && r.denom() <= EXACT_INT)
            .map(|r| r.denom() as f64);
        Self {
            factor,
            offset,
            exact,
            divisor,
        }
    }

    /// Multiplicative factor.
    #[must_use]
    pub const fn factor(&self) -> f64 {
        self.factor
    }

    /// Additive offset (non-zero only for affine conversions).
    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }

    /// The factor as an exact rational, when there is no offset and no factor of π.
    /// Integer data can be rescaled exactly with it.
    #[must_use]
    pub const fn exact_factor(&self) -> Option<Rational> {
        self.exact
    }

    /// Whether this conversion changes nothing.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.exact == Some(Rational::ONE) && self.offset == 0.0
    }

    /// Convert one value.
    #[must_use]
    pub fn apply(&self, x: f64) -> f64 {
        // Not `mul_add`: without FMA hardware enabled at compile time it is a slow library
        // call, and numpy (hence xarray) rounds after each operation.
        match self.divisor {
            Some(n) => x / n + self.offset,
            None => x * self.factor + self.offset,
        }
    }
}

/// The units of an addition, subtraction or comparison, and how to convert each operand first.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryPlan {
    /// The unit of the result.
    pub result: Unit,
    /// Conversion to apply to the left operand, if any.
    pub lhs: Option<Converter>,
    /// Conversion to apply to the right operand, if any.
    pub rhs: Option<Converter>,
}

/// A unit of measurement, checked at runtime.
///
/// Equality and hashing ignore the [label](Unit::label): `m/s` equals `m s-1`.
#[derive(Debug, Clone)]
pub struct Unit {
    dim: Dimension,
    scale: Scale,
    kind: UnitKind,
    label: Option<Arc<str>>,
}

impl PartialEq for Unit {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.scale == other.scale && self.kind == other.kind
    }
}

impl Eq for Unit {}

impl Hash for Unit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dim.hash(state);
        self.scale.hash(state);
        self.kind.hash(state);
    }
}

impl Unit {
    /// The dimensionless unit `1`.
    #[must_use]
    pub const fn one() -> Self {
        Self {
            dim: Dimension::NONE,
            scale: Scale::ONE,
            kind: UnitKind::Linear,
            label: None,
        }
    }

    /// The coherent SI unit of a base quantity (`kg`, `m`, `s`, …, `rad`).
    #[must_use]
    pub fn base(b: BaseDim) -> Self {
        Self::linear(Dimension::of(b), Scale::ONE)
    }

    /// A linear unit with the given dimension and scale.
    #[must_use]
    pub const fn linear(dim: Dimension, scale: Scale) -> Self {
        Self {
            dim,
            scale,
            kind: UnitKind::Linear,
            label: None,
        }
    }

    /// An affine unit: `value_SI = (value + offset) · scale`.
    #[must_use]
    pub const fn affine(dim: Dimension, scale: Scale, offset: Rational) -> Self {
        Self {
            dim,
            scale,
            kind: UnitKind::Affine { offset },
            label: None,
        }
    }

    /// `<step> since <reference>`.
    ///
    /// # Errors
    /// [`UnitError::Incompatible`] if `step` is not a linear unit of time.
    pub fn epoch(step: &Self, reference: RefDate) -> Result<Self, UnitError> {
        if step.kind != UnitKind::Linear || step.dim != Dimension::of(BaseDim::Time) {
            return Err(UnitError::Incompatible {
                lhs: step.to_string(),
                rhs: "s".to_owned(),
            });
        }
        Ok(Self {
            dim: step.dim,
            scale: step.scale,
            kind: UnitKind::Epoch(reference),
            label: None,
        })
    }

    /// A logarithmic unit relative to `reference`.
    ///
    /// # Errors
    /// [`UnitError::UnsupportedOp`] if `reference` is not linear.
    pub fn log(base: LogBase, reference: &Self) -> Result<Self, UnitError> {
        if reference.kind != UnitKind::Linear {
            return Err(reference.unsupported("a logarithmic reference"));
        }
        let r = LogRef {
            base,
            reference_dim: reference.dim,
            reference_scale: reference.scale,
        };
        Ok(Self {
            dim: Dimension::NONE,
            scale: Scale::ONE,
            kind: UnitKind::Log(Box::new(r)),
            label: None,
        })
    }

    /// The dimension.
    #[must_use]
    pub const fn dim(&self) -> Dimension {
        self.dim
    }

    /// The scale relative to the coherent SI unit of the same dimension.
    #[must_use]
    pub const fn scale(&self) -> Scale {
        self.scale
    }

    /// The kind.
    #[must_use]
    pub const fn kind(&self) -> &UnitKind {
        &self.kind
    }

    /// The original spelling, if this unit was parsed or labelled.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// The same unit with a label (used by `Display` and written back to files).
    #[must_use]
    pub fn with_label(mut self, label: impl Into<Arc<str>>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// The same unit without a label.
    #[must_use]
    pub fn without_label(mut self) -> Self {
        self.label = None;
        self
    }

    /// Whether this is a linear unit with no dimension (`1`, `%`, `1e-3`, …).
    #[must_use]
    pub fn is_dimensionless(&self) -> bool {
        self.kind == UnitKind::Linear && self.dim.is_none()
    }

    fn unsupported(&self, op: &'static str) -> UnitError {
        UnitError::UnsupportedOp {
            op,
            unit: self.to_string(),
        }
    }

    fn incompatible(&self, other: &Self) -> UnitError {
        UnitError::Incompatible {
            lhs: self.to_string(),
            rhs: other.to_string(),
        }
    }

    /// For an affine or epoch unit, the corresponding difference unit (°C → a kelvin-sized
    /// step; `days since 2000-01-01` → `d`). Other units are returned unchanged.
    #[must_use]
    pub fn delta(&self) -> Self {
        match self.kind {
            UnitKind::Affine { .. } | UnitKind::Epoch(_) => Self::linear(self.dim, self.scale),
            _ => self.clone(),
        }
    }

    /// Multiply by a pure scale factor (an SI prefix, say). Defined for linear and log units.
    pub(crate) fn scaled(&self, s: Scale) -> Result<Self, UnitError> {
        match self.kind {
            UnitKind::Linear | UnitKind::Log(_) => Ok(Self {
                scale: self.scale.checked_mul(s)?,
                label: None,
                ..self.clone()
            }),
            _ => Err(self.unsupported("scaling")),
        }
    }

    /// `self · rhs`. Defined for linear units, and for a log unit times a pure number.
    ///
    /// # Errors
    /// [`UnitError::UnsupportedOp`] for affine and epoch units (use [`delta`](Self::delta)
    /// first); [`UnitError::Overflow`].
    pub fn mul(&self, rhs: &Self) -> Result<Self, UnitError> {
        match (&self.kind, &rhs.kind) {
            (UnitKind::Linear, UnitKind::Linear) => Ok(labelled(
                Self::linear(
                    self.dim.checked_mul(rhs.dim)?,
                    self.scale.checked_mul(rhs.scale)?,
                ),
                combine(self.factors(), rhs.factors(), 1),
            )),
            (UnitKind::Log(_), UnitKind::Linear) if rhs.dim.is_none() => self.scaled(rhs.scale),
            (UnitKind::Linear, UnitKind::Log(_)) if self.dim.is_none() => rhs.scaled(self.scale),
            (UnitKind::Linear, _) => Err(rhs.unsupported("multiplication")),
            _ => Err(self.unsupported("multiplication")),
        }
    }

    /// `1 / self`.
    ///
    /// # Errors
    /// As for [`mul`](Self::mul).
    pub fn recip(&self) -> Result<Self, UnitError> {
        self.powi(-1)
    }

    /// `self / rhs`.
    ///
    /// # Errors
    /// As for [`mul`](Self::mul).
    pub fn div(&self, rhs: &Self) -> Result<Self, UnitError> {
        if let (UnitKind::Log(_), UnitKind::Linear) = (&self.kind, &rhs.kind)
            && rhs.dim.is_none()
        {
            return self.scaled(rhs.scale.recip()?);
        }
        self.mul(&rhs.recip()?)
    }

    /// `self ^ n`. Defined for linear units (any unit when `n == 1`).
    ///
    /// # Errors
    /// [`UnitError::UnsupportedOp`] for non-linear units; [`UnitError::Overflow`].
    pub fn powi(&self, n: i32) -> Result<Self, UnitError> {
        match self.kind {
            _ if n == 1 => Ok(self.clone()),
            UnitKind::Linear => Ok(labelled(
                Self::linear(self.dim.powi(n)?, self.scale.powi(n)?),
                self.factors()
                    .and_then(|f| scaled_exps(f, |e| e.checked_mul(n))),
            )),
            _ => Err(self.unsupported("exponentiation")),
        }
    }

    /// The `n`-th root (e.g. `sqrt` of `m2 s-2` is `m s-1`).
    ///
    /// # Errors
    /// [`UnitError::InexactRoot`] if an exponent is not divisible by `n` or the scale has no
    /// exact root; [`UnitError::UnsupportedOp`] for non-linear units.
    pub fn root(&self, n: i32) -> Result<Self, UnitError> {
        if self.kind != UnitKind::Linear {
            return Err(self.unsupported("a root"));
        }
        let inexact = || UnitError::InexactRoot {
            n,
            unit: self.to_string(),
        };
        Ok(labelled(
            Self::linear(
                self.dim.root(n).ok_or_else(inexact)?,
                self.scale.root(n).ok_or_else(inexact)?,
            ),
            self.factors().and_then(|f| {
                scaled_exps(f, |e| {
                    (e.checked_rem(n) == Some(0))
                        .then(|| e.checked_div(n))
                        .flatten()
                })
            }),
        ))
    }

    /// The label as factors with exponents, when it is a plain product (`km`, `m s-1`,
    /// `km/min`, `kg m2 s-2`).
    fn factors(&self) -> Option<Factors> {
        factors(self.label.as_deref()?)
    }

    /// Fold angle into dimensionless (whippyunits' erasure): `degree` becomes the pure number
    /// π/180, so converting to `1` yields radians.
    #[must_use]
    pub fn erase_angle(&self) -> Self {
        Self {
            dim: self.dim.without_angle(),
            label: None,
            ..self.clone()
        }
    }

    /// Lenient multiplication used when parsing unit strings: affine factors lose their
    /// offset, as in UDUNITS (`kg degree_C m-2` measures temperature differences).
    pub(crate) fn mul_lenient(&self, rhs: &Self) -> Result<Self, UnitError> {
        self.delta().mul(&rhs.delta())
    }

    pub(crate) fn div_lenient(&self, rhs: &Self) -> Result<Self, UnitError> {
        self.delta().div(&rhs.delta())
    }

    pub(crate) fn powi_lenient(&self, n: i32) -> Result<Self, UnitError> {
        if n == 1 {
            Ok(self.clone())
        } else {
            self.delta().powi(n)
        }
    }

    /// Whether values in `self` can be converted to `other`.
    #[must_use]
    pub fn is_convertible(&self, other: &Self) -> bool {
        self.converter_to(other).is_ok()
    }

    /// The conversion of values from `self` into `to`.
    ///
    /// Linear and affine units of the same dimension convert into each other (K ↔ °C ↔ °F).
    /// Log units convert only between the same reference (B ↔ dB). Epoch units are decoded with
    /// [`epoch_codec`](Self::epoch_codec) instead.
    ///
    /// # Errors
    /// [`UnitError::Incompatible`] if the units measure different things.
    pub fn converter_to(&self, to: &Self) -> Result<Converter, UnitError> {
        let offset = |k: &UnitKind| match k {
            UnitKind::Linear => Some(Rational::ZERO),
            UnitKind::Affine { offset } => Some(*offset),
            _ => None,
        };
        if self.dim != to.dim {
            return Err(self.incompatible(to));
        }
        let ratio = self.scale.checked_div(to.scale)?;
        match (offset(&self.kind), offset(&to.kind)) {
            (Some(o1), Some(o2)) => {
                // y = (x + o1)·s1/s2 − o2 = x·r + (o1·r − o2)
                let exact_r = ratio.to_rational();
                let off = match exact_r {
                    Some(r) => o1.checked_mul(r)?.checked_sub(o2)?.to_f64(),
                    None => o1.to_f64().mul_add(ratio.to_f64(), -o2.to_f64()),
                };
                let exact = if o1.is_zero() && o2.is_zero() {
                    exact_r
                } else {
                    None
                };
                Ok(Converter::new(ratio.to_f64(), off, exact))
            }
            _ => match (&self.kind, &to.kind) {
                (UnitKind::Log(a), UnitKind::Log(b)) if a == b => {
                    Ok(Converter::new(ratio.to_f64(), 0.0, ratio.to_rational()))
                }
                _ => Err(self.incompatible(to)),
            },
        }
    }

    /// Plan `self + rhs` under `policy`.
    ///
    /// A temperature difference may be added to an absolute (°C) temperature; two absolute
    /// temperatures may not be added.
    ///
    /// # Errors
    /// [`UnitError::Incompatible`], [`UnitError::ScaleMismatch`] (strict policy) or
    /// [`UnitError::UnsupportedOp`].
    pub fn plan_add(&self, rhs: &Self, policy: UnitPolicy) -> Result<BinaryPlan, UnitError> {
        use UnitKind::{Affine, Linear};
        match (&self.kind, &rhs.kind) {
            (Linear, Linear) | (Affine { .. }, Linear) => {
                self.plan_rhs_into(&self.delta(), rhs, self.clone(), policy)
            }
            (Linear, Affine { .. }) => rhs
                .plan_rhs_into(&rhs.delta(), self, rhs.clone(), policy)
                .map(|p| BinaryPlan {
                    result: p.result,
                    lhs: p.rhs,
                    rhs: None,
                }),
            (Affine { .. }, Affine { .. }) => {
                Err(self.unsupported("adding two absolute temperatures"))
            }
            _ => Err(self.unsupported("addition")),
        }
    }

    /// Plan `self - rhs` under `policy`. Absolute − absolute temperature gives a difference.
    ///
    /// # Errors
    /// As for [`plan_add`](Self::plan_add).
    pub fn plan_sub(&self, rhs: &Self, policy: UnitPolicy) -> Result<BinaryPlan, UnitError> {
        use UnitKind::{Affine, Linear};
        match (&self.kind, &rhs.kind) {
            (Linear, Linear) | (Affine { .. }, Linear) => {
                self.plan_rhs_into(&self.delta(), rhs, self.clone(), policy)
            }
            (Affine { .. }, Affine { .. }) => self.plan_rhs_into(self, rhs, self.delta(), policy),
            (Linear, Affine { .. }) => {
                Err(rhs.unsupported("subtracting an absolute temperature from a difference"))
            }
            _ => Err(self.unsupported("subtraction")),
        }
    }

    /// Plan a comparison of `self` with `rhs` under `policy`. The result unit is `self`.
    ///
    /// # Errors
    /// [`UnitError::Incompatible`] or [`UnitError::ScaleMismatch`] (strict policy).
    pub fn plan_cmp(&self, rhs: &Self, policy: UnitPolicy) -> Result<BinaryPlan, UnitError> {
        match (&self.kind, &rhs.kind) {
            (UnitKind::Epoch(_), _) | (_, UnitKind::Epoch(_)) => {
                Err(self.unsupported("comparison of encoded times"))
            }
            _ => self.plan_rhs_into(self, rhs, self.clone(), policy),
        }
    }

    /// `rhs` must be convertible into `target`; under the strict policy it must already equal it.
    fn plan_rhs_into(
        &self,
        target: &Self,
        rhs: &Self,
        result: Self,
        policy: UnitPolicy,
    ) -> Result<BinaryPlan, UnitError> {
        if rhs == target {
            return Ok(BinaryPlan {
                result,
                lhs: None,
                rhs: None,
            });
        }
        // Errors name the operands in their order (the conversion goes right to left).
        let conv = rhs.converter_to(target).map_err(|e| match e {
            UnitError::Incompatible { .. } => self.incompatible(rhs),
            other => other,
        })?;
        match policy {
            UnitPolicy::Strict => Err(UnitError::ScaleMismatch {
                lhs: target.to_string(),
                rhs: rhs.to_string(),
            }),
            UnitPolicy::AutoRescale => Ok(BinaryPlan {
                result,
                lhs: None,
                rhs: Some(conv),
            }),
        }
    }

    /// For a `<step> since <date>` unit, a codec between numbers and dates in `calendar`.
    ///
    /// # Errors
    /// [`UnitError::NotAnEpoch`] for other units; [`UnitError::UnsupportedTimeStep`] if the
    /// step is not a whole number of nanoseconds (e.g. `months since`);
    /// [`UnitError::InvalidDate`] if the reference date does not exist in `calendar`.
    pub fn epoch_codec(&self, calendar: Calendar) -> Result<EpochCodec, UnitError> {
        let UnitKind::Epoch(reference) = &self.kind else {
            return Err(UnitError::NotAnEpoch(self.to_string()));
        };
        // Steps must be whole seconds (s … weeks) or divide a second (ms, µs, ns). UDUNITS'
        // `month` and `year` are fractional-second astronomical lengths; cftime rejects them too.
        const NS_PER_SEC: i128 = 1_000_000_000;
        let ns = self
            .scale
            .checked_mul(Scale::from_decimal(1, 9)?)?
            .to_rational();
        let step_ns = match ns {
            Some(r)
                if r.denom() == 1
                    && r.numer() > 0
                    && (r.numer().checked_rem(NS_PER_SEC) == Some(0)
                        || NS_PER_SEC.checked_rem(r.numer()) == Some(0)) =>
            {
                r.numer()
            }
            _ => return Err(UnitError::UnsupportedTimeStep(self.to_string())),
        };
        EpochCodec::new(step_ns, reference, calendar)
    }

    /// A canonical spelling, ignoring the label: `1e-3 kg m-2`, `K @ 273.15`, `s since …`.
    #[must_use]
    pub fn canonical(&self) -> String {
        let body = match (self.scale.is_one(), self.dim.is_none()) {
            (true, _) => self.dim.to_string(),
            (false, true) => self.scale.to_string(),
            (false, false) => format!("{} {}", self.scale, self.dim),
        };
        match &self.kind {
            UnitKind::Linear => body,
            UnitKind::Affine { offset } => format!("{body} @ {}", offset.to_f64()),
            UnitKind::Epoch(r) => format!("{body} since {r}"),
            UnitKind::Log(r) => {
                let f = match r.base {
                    LogBase::Ten => "lg",
                    LogBase::Two => "lb",
                    LogBase::E => "ln",
                };
                let reference = Self::linear(r.reference_dim, r.reference_scale).canonical();
                if self.scale.is_one() {
                    format!("{f}(re {reference})")
                } else {
                    format!("{} {f}(re {reference})", self.scale)
                }
            }
        }
    }
}

/// A label as named factors with integer exponents, in order of appearance.
type Factors = Vec<(String, i32)>;

/// `label`'s factors, for labels written as a plain product with at most one `/`
/// (`km`, `m s-1`, `km/min`, `kg m2 s-2`); `None` for anything else (numbers, brackets,
/// powers with `^`, offsets), which keeps derived units on their canonical spelling.
fn factors(label: &str) -> Option<Factors> {
    let (num, den) = match label.split_once('/') {
        Some((a, b)) => (a, b),
        None => (label, ""),
    };
    if den.contains('/') {
        return None;
    }
    let mut out: Factors = Vec::new();
    for (part, sign) in [(num, 1), (den, -1)] {
        for tok in part.split_whitespace() {
            let (sym, exp) = split_power(tok)?;
            add_factor(&mut out, sym, exp.checked_mul(sign)?)?;
        }
    }
    (!out.is_empty()).then_some(out)
}

/// `km` → (`km`, 1); `s-1` → (`s`, -1); `m2` → (`m`, 2).
fn split_power(tok: &str) -> Option<(&str, i32)> {
    if tok.contains(['(', ')', '*', '^', '.', '·', '@']) {
        return None;
    }
    let split = tok.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    let (mut sym, mut exp) = (tok, 1i32);
    if split < tok.len() {
        let head = tok.get(..split)?;
        let start = if head.ends_with(['-', '+']) {
            split.checked_sub(1)?
        } else {
            split
        };
        exp = tok.get(start..)?.parse().ok()?;
        sym = tok.get(..start)?;
    }
    let first = sym.chars().next()?;
    (!first.is_ascii_digit() && !sym.ends_with(['-', '+'])).then_some((sym, exp))
}

fn add_factor(out: &mut Factors, sym: &str, exp: i32) -> Option<()> {
    match out.iter_mut().find(|(s, _)| s == sym) {
        Some((_, e)) => *e = e.checked_add(exp)?,
        None => out.push((sym.to_owned(), exp)),
    }
    Some(())
}

/// The factors of a product (`sign` 1) or quotient (`sign` -1) of two labels.
fn combine(a: Option<Factors>, b: Option<Factors>, sign: i32) -> Option<Factors> {
    let mut out = a?;
    for (sym, exp) in b? {
        add_factor(&mut out, &sym, exp.checked_mul(sign)?)?;
    }
    Some(out)
}

fn scaled_exps(f: Factors, g: impl Fn(i32) -> Option<i32>) -> Option<Factors> {
    f.into_iter().map(|(s, e)| Some((s, g(e)?))).collect()
}

/// `u` labelled with `factors` written UDUNITS style (`km min-1`, `km2`), if that spelling
/// parses back to exactly `u`; otherwise `u` as it is (shown in canonical form).
fn labelled(u: Unit, factors: Option<Factors>) -> Unit {
    let Some(f) = factors else { return u };
    let terms: Vec<String> = f
        .iter()
        .filter(|(_, e)| *e != 0)
        .map(|(s, e)| {
            if *e == 1 {
                s.clone()
            } else {
                format!("{s}{e}")
            }
        })
        .collect();
    if terms.is_empty() {
        return u;
    }
    let label = terms.join(" ");
    match label.parse::<Unit>() {
        Ok(p) if p == u => u.with_label(label),
        _ => u,
    }
}

impl fmt::Display for Unit {
    /// The label if there is one, otherwise [`canonical`](Unit::canonical).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.label {
            Some(l) => f.write_str(l),
            None => f.write_str(&self.canonical()),
        }
    }
}
