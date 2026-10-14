//! The unit and prefix registry, built once from the generated UDUNITS table.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::parse::{parse_number, parse_with};
use crate::table::{PREFIXES, UNITS};
use crate::{BaseDim, Scale, Unit, UnitError};

/// How a table entry is defined.
pub(crate) enum Def {
    /// A base unit.
    Base(BaseDim),
    /// Exactly π.
    Pi,
    /// A UDUNITS expression in terms of other units.
    Expr(&'static str),
}

/// A unit from the UDUNITS database.
pub(crate) struct UnitEntry {
    pub(crate) def: Def,
    /// Singular and plural names (case-insensitive).
    pub(crate) names: &'static [&'static str],
    /// Symbols (case-sensitive).
    pub(crate) symbols: &'static [&'static str],
}

/// An SI prefix.
pub(crate) struct PrefixEntry {
    pub(crate) value: &'static str,
    pub(crate) names: &'static [&'static str],
    pub(crate) symbols: &'static [&'static str],
}

#[derive(Default)]
pub(crate) struct Registry {
    symbols: HashMap<&'static str, Unit>,
    names: HashMap<String, Unit>,
    symbol_prefixes: Vec<(&'static str, Scale)>,
    name_prefixes: Vec<(&'static str, Scale)>,
    /// Entries whose definitions could not be evaluated (kept for diagnostics and tests).
    pub(crate) failures: Vec<(&'static str, UnitError)>,
}

impl Registry {
    /// Resolve an identifier: exact symbol, then name (case-insensitive), then prefix + unit.
    pub(crate) fn lookup(&self, id: &str) -> Option<Unit> {
        if let Some(u) = self.symbols.get(id) {
            return Some(u.clone());
        }
        if let Some(u) = self.names.get(&id.to_lowercase()) {
            return Some(u.clone());
        }
        // Prefix + unit. Like UDUNITS, a prefix symbol may precede a unit name (`dbar`, `mbar`).
        let lower = id.to_lowercase();
        let unit_after = |rest: &str| {
            self.symbols
                .get(rest)
                .or_else(|| self.names.get(&rest.to_lowercase()))
        };
        for (p, scale) in &self.symbol_prefixes {
            if let Some(u) = id
                .strip_prefix(p)
                .filter(|r| !r.is_empty())
                .and_then(unit_after)
            {
                return u.scaled(*scale).ok();
            }
        }
        for (p, scale) in &self.name_prefixes {
            if let Some(u) = lower
                .strip_prefix(p)
                .filter(|r| !r.is_empty())
                .and_then(|r| self.names.get(r))
            {
                return u.scaled(*scale).ok();
            }
        }
        None
    }

    fn insert(&mut self, entry: &UnitEntry, unit: &Unit) {
        for s in entry.symbols {
            self.symbols.entry(s).or_insert_with(|| unit.clone());
        }
        for n in entry.names {
            self.names
                .entry(n.to_lowercase())
                .or_insert_with(|| unit.clone());
        }
    }

    fn build() -> Self {
        let mut reg = Self::default();
        for p in PREFIXES {
            match parse_number(p.value).and_then(|(m, e)| Scale::from_decimal(m, e)) {
                Ok(scale) => {
                    reg.symbol_prefixes
                        .extend(p.symbols.iter().map(|s| (*s, scale)));
                    reg.name_prefixes
                        .extend(p.names.iter().map(|s| (*s, scale)));
                }
                Err(e) => reg.failures.push((p.value, e)),
            }
        }
        // Longest prefix first, so `da` wins over `d`.
        reg.symbol_prefixes
            .sort_by_key(|(p, _)| std::cmp::Reverse(p.len()));
        reg.name_prefixes
            .sort_by_key(|(p, _)| std::cmp::Reverse(p.len()));

        // Definitions refer to other units in any order: resolve in passes until no progress.
        let mut pending: Vec<&UnitEntry> = UNITS.iter().collect();
        loop {
            let before = pending.len();
            let mut still = Vec::new();
            for entry in pending {
                let unit = match entry.def {
                    Def::Base(b) => Ok(Unit::base(b)),
                    Def::Pi => Ok(Unit::linear(crate::Dimension::NONE, Scale::pi())),
                    Def::Expr(e) => parse_with(e, &|id| reg.lookup(id)).map(Unit::without_label),
                };
                match unit {
                    Ok(u) => reg.insert(entry, &u),
                    Err(_) => still.push(entry),
                }
            }
            pending = still;
            if pending.is_empty() || pending.len() == before {
                break;
            }
        }
        for entry in pending {
            if let Def::Expr(e) = entry.def
                && let Err(err) = parse_with(e, &|id| reg.lookup(id))
            {
                let name = entry
                    .names
                    .first()
                    .or(entry.symbols.first())
                    .copied()
                    .unwrap_or(e);
                reg.failures.push((name, err));
            }
        }
        reg
    }
}

/// The process-wide registry.
pub(crate) fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(Registry::build)
}
