//! Parser for UDUNITS-2 / CF unit strings (and the UCUM-style `kg.m2/s2` spelling).
//!
//! ```text
//! unit     := product [ SHIFT ( date | number ) ]          SHIFT: @ since after from ref
//! product  := power { ( '*' | '.' | '·' | '/' | "per" | juxtaposition ) power }
//! power    := basic [ INT | ( '^' | '**' ) INT ]           `m2`, `s-1`, `m^2`, `(m-1)-1`
//! basic    := NUMBER | ID | '(' product ')' | (lg|ln|lb) '(' "re" product ')'
//! ```

use std::iter::Peekable;
use std::str::CharIndices;

use crate::time::looks_like_date;
use crate::unit::LogBase;
use crate::{Dimension, Rational, Scale, Unit, UnitError};

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    /// `mantissa · 10^exp10`, with the integer value when it is a plain integer.
    Num {
        mantissa: i128,
        exp10: i32,
        int: Option<i32>,
    },
    Id(String),
    /// An integer written directly after a unit or `)`: `m2`, `s-1`.
    Exp(i32),
    Mul,
    Div,
    Pow,
    LParen,
    RParen,
}

struct Lexed {
    tok: Tok,
    pos: usize,
}

fn err(input: &str, pos: usize, reason: impl Into<String>) -> UnitError {
    UnitError::Parse {
        input: input.to_owned(),
        pos,
        reason: reason.into(),
    }
}

fn is_id_char(c: char) -> bool {
    c.is_alphabetic() || matches!(c, '_' | '°' | '%' | '\'' | '"' | '℃' | '℉')
}

/// Parse a decimal literal (`12`, `.1`, `1e-3`, `273.15`) into `mantissa · 10^exp10`.
pub(crate) fn parse_number(s: &str) -> Result<(i128, i32), UnitError> {
    let bad = || err(s, 0, "not a number");
    let s = s.trim();
    let (sign, body) = match s.strip_prefix('-') {
        Some(rest) => (-1, rest),
        None => (1, s.strip_prefix('+').unwrap_or(s)),
    };
    let (num, exp) = match body.find(['e', 'E']) {
        Some(i) => {
            let (n, e) = body.split_at_checked(i).ok_or_else(bad)?;
            let e = e.get(1..).ok_or_else(bad)?;
            (n, e.parse::<i32>().map_err(|_| bad())?)
        }
        None => (body, 0),
    };
    let (int_part, frac) = num.split_once('.').unwrap_or((num, ""));
    if int_part.is_empty() && frac.is_empty() {
        return Err(bad());
    }
    let digits: String = int_part.chars().chain(frac.chars()).collect();
    if !digits.chars().all(|c| c.is_ascii_digit()) {
        return Err(bad());
    }
    // Drop trailing zeros into the exponent so 1000 and 1e3 give the same mantissa.
    let trimmed = digits.trim_end_matches('0');
    let zeros = i32::try_from(digits.len().saturating_sub(trimmed.len())).map_err(|_| bad())?;
    let frac_len = i32::try_from(frac.len()).map_err(|_| bad())?;
    let trimmed = trimmed.trim_start_matches('0');
    let mantissa: i128 = if trimmed.is_empty() {
        0
    } else {
        trimmed.parse().map_err(|_| UnitError::Overflow)?
    };
    let exp10 = exp
        .checked_sub(frac_len)
        .and_then(|e| e.checked_add(zeros))
        .ok_or(UnitError::Overflow)?;
    let mantissa = mantissa.checked_mul(sign).ok_or(UnitError::Overflow)?;
    Ok((mantissa, if mantissa == 0 { 0 } else { exp10 }))
}

fn lex(input: &str) -> Result<Vec<Lexed>, UnitError> {
    let mut out: Vec<Lexed> = Vec::new();
    let mut it: Peekable<CharIndices<'_>> = input.char_indices().peekable();
    let mut spaced = true;
    while let Some(&(pos, c)) = it.peek() {
        let glued = !spaced
            && matches!(
                out.last(),
                Some(Lexed {
                    tok: Tok::Id(_) | Tok::RParen,
                    ..
                })
            );
        spaced = false;
        let tok = if c.is_whitespace() {
            it.next();
            spaced = true;
            continue;
        } else if glued && (c.is_ascii_digit() || c == '-' || c == '+') {
            // Exponent glued to a unit: `m2`, `s-1`, `)-1`.
            let mut text = String::new();
            if c == '-' || c == '+' {
                text.push(c);
                it.next();
            }
            while let Some((_, d)) = it.next_if(|(_, d)| d.is_ascii_digit()) {
                text.push(d);
            }
            Tok::Exp(text.parse().map_err(|_| err(input, pos, "bad exponent"))?)
        } else if c.is_ascii_digit()
            || ((c == '.' || c == '-' || c == '+') && next_is_digit(input, pos, c))
        {
            let mut text = String::new();
            text.push(c);
            it.next();
            let mut seen_e = false;
            while let Some(&(_, d)) = it.peek() {
                let sign_after_e = (d == '-' || d == '+') && text.ends_with(['e', 'E']);
                let e_with_digits =
                    (d == 'e' || d == 'E') && !seen_e && exponent_follows(input, &mut it.clone());
                if d.is_ascii_digit() || d == '.' || sign_after_e || e_with_digits {
                    seen_e |= d == 'e' || d == 'E';
                    text.push(d);
                    it.next();
                } else {
                    break;
                }
            }
            let (mantissa, exp10) =
                parse_number(&text).map_err(|_| err(input, pos, "bad number"))?;
            let int = text.parse::<i32>().ok();
            Tok::Num {
                mantissa,
                exp10,
                int,
            }
        } else if is_id_char(c) {
            let mut text = String::new();
            while let Some((_, d)) = it.next_if(|(_, d)| is_id_char(*d) || d.is_ascii_digit()) {
                text.push(d);
            }
            // Trailing digits are an exponent (`m2`), interior ones are part of the name (`H2O`).
            let name_len = text.trim_end_matches(|d: char| d.is_ascii_digit()).len();
            let (name, exp) = text
                .split_at_checked(name_len)
                .ok_or_else(|| err(input, pos, "bad identifier"))?;
            out.push(Lexed {
                tok: Tok::Id(name.to_owned()),
                pos,
            });
            if !exp.is_empty() {
                let e = exp.parse().map_err(|_| err(input, pos, "bad exponent"))?;
                out.push(Lexed {
                    tok: Tok::Exp(e),
                    pos,
                });
            }
            continue;
        } else {
            it.next();
            match c {
                '*' if it.next_if(|(_, d)| *d == '*').is_some() => Tok::Pow,
                '*' | '.' | '·' => Tok::Mul,
                '/' => Tok::Div,
                '^' => Tok::Pow,
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                _ => return Err(err(input, pos, format!("unexpected character {c:?}"))),
            }
        };
        out.push(Lexed { tok, pos });
    }
    Ok(out)
}

fn next_is_digit(input: &str, pos: usize, c: char) -> bool {
    input
        .get(pos.saturating_add(c.len_utf8())..)
        .and_then(|r| r.chars().next())
        .is_some_and(|d| d.is_ascii_digit())
}

/// After an `e`, is there an exponent (`e3`, `e-3`) rather than a unit (`eV`)?
fn exponent_follows(_input: &str, it: &mut Peekable<CharIndices<'_>>) -> bool {
    it.next(); // the `e`
    it.next_if(|(_, c)| *c == '-' || *c == '+');
    it.peek().is_some_and(|(_, c)| c.is_ascii_digit())
}

/// The deepest parentheses nest: the parser recurses into each, so a bound keeps hostile
/// input (`((((…`) from exhausting the stack.
const MAX_NESTING: usize = 64;

struct Parser<'a, 'f> {
    input: &'a str,
    toks: Vec<Lexed>,
    i: usize,
    lookup: &'f dyn Fn(&str) -> Option<Unit>,
    /// Parentheses open around the current position.
    depth: usize,
}

impl Parser<'_, '_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i).map(|l| &l.tok)
    }

    fn pos(&self) -> usize {
        self.toks.get(self.i).map_or(self.input.len(), |l| l.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.i).map(|l| l.tok.clone());
        self.i = self.i.saturating_add(1);
        t
    }

    fn err(&self, reason: impl Into<String>) -> UnitError {
        err(self.input, self.pos(), reason)
    }

    /// A product inside parentheses, one level deeper.
    fn nested(&mut self) -> Result<Unit, UnitError> {
        if self.depth >= MAX_NESTING {
            return Err(self.err(format!("parentheses nest deeper than {MAX_NESTING} levels")));
        }
        self.depth = self.depth.saturating_add(1);
        let u = self.product();
        self.depth = self.depth.saturating_sub(1);
        u
    }

    fn product(&mut self) -> Result<Unit, UnitError> {
        let mut acc = self.power()?;
        loop {
            match self.peek() {
                Some(Tok::Mul) => {
                    self.bump();
                    acc = acc.mul_lenient(&self.power()?)?;
                }
                Some(Tok::Div) => {
                    self.bump();
                    acc = acc.div_lenient(&self.power()?)?;
                }
                Some(Tok::Id(s)) if s.eq_ignore_ascii_case("per") => {
                    self.bump();
                    acc = acc.div_lenient(&self.power()?)?;
                }
                Some(Tok::Num { .. } | Tok::Id(_) | Tok::LParen) => {
                    acc = acc.mul_lenient(&self.power()?)?;
                }
                _ => return Ok(acc),
            }
        }
    }

    fn power(&mut self) -> Result<Unit, UnitError> {
        let base = self.basic()?;
        match self.peek() {
            Some(Tok::Exp(n)) => {
                let n = *n;
                self.bump();
                base.powi_lenient(n)
            }
            Some(Tok::Pow) => {
                self.bump();
                match self.bump() {
                    Some(Tok::Num { int: Some(n), .. }) => base.powi_lenient(n),
                    _ => Err(self.err("expected an integer exponent")),
                }
            }
            _ => Ok(base),
        }
    }

    fn basic(&mut self) -> Result<Unit, UnitError> {
        let pos = self.pos();
        match self.bump() {
            Some(Tok::Num {
                mantissa, exp10, ..
            }) => {
                if mantissa == 0 {
                    return Err(err(
                        self.input,
                        pos,
                        "a unit's numeric factor cannot be zero",
                    ));
                }
                Ok(Unit::linear(
                    Dimension::NONE,
                    Scale::from_decimal(mantissa, exp10)?,
                ))
            }
            Some(Tok::LParen) => {
                let u = self.nested()?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(u),
                    _ => Err(self.err("expected ')'")),
                }
            }
            Some(Tok::Id(name)) => {
                let log = match name.as_str() {
                    "lg" => Some(LogBase::Ten),
                    "ln" => Some(LogBase::E),
                    "lb" => Some(LogBase::Two),
                    _ => None,
                };
                let is_log_call = log.is_some()
                    && self.peek() == Some(&Tok::LParen)
                    && matches!(self.toks.get(self.i.saturating_add(1)).map(|l| &l.tok), Some(Tok::Id(re)) if re == "re");
                if let (true, Some(base)) = (is_log_call, log) {
                    self.bump(); // (
                    self.bump(); // re
                    let reference = self.nested()?;
                    return match self.bump() {
                        Some(Tok::RParen) => Unit::log(base, &reference),
                        _ => Err(self.err("expected ')'")),
                    };
                }
                (self.lookup)(&name).ok_or(UnitError::UnknownUnit(name))
            }
            Some(_) => Err(err(self.input, pos, "expected a unit, number or '('")),
            None => Err(err(self.input, pos, "unexpected end of unit")),
        }
    }
}

/// Split off a top-level shift clause: `@ 273.15`, `since 1970-01-01`, ….
fn split_shift(input: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    let lower = input.to_ascii_lowercase();
    for (i, c) in input.char_indices() {
        match c {
            '(' => depth = depth.saturating_add(1),
            ')' => depth = depth.saturating_sub(1),
            '@' if depth == 0 => return Some((input.get(..i)?, input.get(i.checked_add(1)?..)?)),
            _ if depth == 0 && (i == 0 || input.get(..i)?.ends_with(char::is_whitespace)) => {
                for kw in ["since", "after", "from", "ref"] {
                    let rest = lower.get(i..)?;
                    let end = i.checked_add(kw.len())?;
                    if rest.starts_with(kw) && input.get(end..)?.starts_with(char::is_whitespace) {
                        return Some((input.get(..i)?, input.get(end..)?));
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse `input`, resolving identifiers with `lookup`. The result carries no label.
pub(crate) fn parse_with(
    input: &str,
    lookup: &dyn Fn(&str) -> Option<Unit>,
) -> Result<Unit, UnitError> {
    let (unit_part, shift) = match split_shift(input) {
        Some((u, s)) => (u, Some(s.trim())),
        None => (input, None),
    };
    let toks = lex(unit_part)?;
    let base = if toks.is_empty() {
        Unit::one()
    } else {
        let mut p = Parser {
            input,
            toks,
            i: 0,
            lookup,
            depth: 0,
        };
        let u = p.product()?;
        if p.i < p.toks.len() {
            return Err(p.err("unexpected token"));
        }
        u
    };
    let Some(shift) = shift else { return Ok(base) };
    if let Some(date) = looks_like_date(shift) {
        return Unit::epoch(&base, date);
    }
    let (m, e) = parse_number(shift).map_err(|_| {
        err(
            input,
            unit_part.len(),
            "expected a reference date or offset",
        )
    })?;
    if *base.kind() != crate::UnitKind::Linear {
        return Err(UnitError::UnsupportedOp {
            op: "offsetting",
            unit: base.to_string(),
        });
    }
    Ok(Unit::affine(
        base.dim(),
        base.scale(),
        Rational::from_decimal(m, e)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(parse_number("1e-3").unwrap(), (1, -3));
        assert_eq!(parse_number("1000").unwrap(), (1, 3));
        assert_eq!(parse_number(".1").unwrap(), (1, -1));
        assert_eq!(parse_number("273.15").unwrap(), (27315, -2));
        assert_eq!(parse_number("-2.50").unwrap(), (-25, -1));
        assert_eq!(parse_number("0").unwrap(), (0, 0));
        assert!(parse_number("1e").is_err());
        assert!(parse_number(".").is_err());
        assert!(parse_number("1x").is_err());
    }

    #[test]
    fn lexing() {
        let toks = |s: &str| {
            lex(s)
                .unwrap()
                .into_iter()
                .map(|l| l.tok)
                .collect::<Vec<_>>()
        };
        assert_eq!(toks("m2"), vec![Tok::Id("m".into()), Tok::Exp(2)]);
        assert_eq!(toks("s-1"), vec![Tok::Id("s".into()), Tok::Exp(-1)]);
        assert_eq!(toks("H2O"), vec![Tok::Id("H2O".into())]);
        assert_eq!(toks("(m-1)-1").last(), Some(&Tok::Exp(-1)));
        assert_eq!(toks("kg.m2/s2").len(), 7);
        assert_eq!(
            toks("m 2"),
            vec![
                Tok::Id("m".into()),
                Tok::Num {
                    mantissa: 2,
                    exp10: 0,
                    int: Some(2)
                }
            ]
        );
        assert_eq!(
            toks("1e-3 kg").first(),
            Some(&Tok::Num {
                mantissa: 1,
                exp10: -3,
                int: None
            })
        );
        assert_eq!(toks("2 eV").len(), 2);
        assert_eq!(
            toks("m**-2"),
            vec![
                Tok::Id("m".into()),
                Tok::Pow,
                Tok::Num {
                    mantissa: -2,
                    exp10: 0,
                    int: Some(-2)
                }
            ]
        );
    }

    #[test]
    fn nesting_is_bounded() {
        let reg = crate::registry::registry();
        let lookup = |id: &str| reg.lookup(id);
        let deep = |n: usize| format!("{}m{}", "(".repeat(n), ")".repeat(n));
        assert!(parse_with(&deep(MAX_NESTING), &lookup).is_ok());
        let e = parse_with(&deep(MAX_NESTING + 1), &lookup).unwrap_err();
        assert!(e.to_string().contains("nest deeper"), "{e}");
        // Far deeper than the stack would allow without the bound.
        assert!(parse_with(&"(".repeat(1_000_000), &lookup).is_err());
    }

    #[test]
    fn shift_split() {
        assert_eq!(
            split_shift("days since 1970-01-01"),
            Some(("days ", " 1970-01-01"))
        );
        assert_eq!(split_shift("K @ 273.15"), Some(("K ", " 273.15")));
        assert_eq!(
            split_shift("hours SINCE 2000-01-01"),
            Some(("hours ", " 2000-01-01"))
        );
        assert_eq!(split_shift("ref_unit"), None);
        assert_eq!(split_shift("(K @ 1)"), None);
    }
}
