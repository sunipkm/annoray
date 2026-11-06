//! The unit and prefix registry, built once from the generated UDUNITS table, and the units an
//! application adds at run time ([`define_unit`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, PoisonError, RwLock};

use crate::extra::{AMENDMENTS, EXTRA_UNITS};
use crate::parse::{parse_number, parse_with};
use crate::table::{PREFIXES, UNITS};
use crate::{BaseDim, Scale, Unit, UnitError};

/// How a table entry is defined.
#[derive(Clone, Copy)]
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

/// Units defined at run time, by the exact spelling (they behave as symbols).
type Custom = HashMap<String, Unit>;

static CUSTOM: RwLock<Option<Custom>> = RwLock::new(None);
/// Whether [`CUSTOM`] has entries: most programs never define a unit, and do not take the lock.
static ANY_CUSTOM: AtomicBool = AtomicBool::new(false);

impl Registry {
    /// Resolve an identifier: exact symbol, then name (case-insensitive), then a unit defined
    /// at run time, then prefix + unit.
    pub(crate) fn lookup(&self, id: &str) -> Option<Unit> {
        if ANY_CUSTOM.load(Ordering::Acquire) {
            let custom = CUSTOM.read().unwrap_or_else(PoisonError::into_inner);
            self.lookup_in(id, custom.as_ref())
        } else {
            self.lookup_in(id, None)
        }
    }

    fn lookup_in(&self, id: &str, custom: Option<&Custom>) -> Option<Unit> {
        if let Some(u) = self.symbols.get(id) {
            return Some(u.clone());
        }
        if let Some(u) = self.names.get(&id.to_lowercase()) {
            return Some(u.clone());
        }
        if let Some(u) = custom.and_then(|c| c.get(id)) {
            return Some(u.clone());
        }
        // Prefix + unit. Like UDUNITS, a prefix symbol may precede a unit name (`dbar`, `mbar`).
        let lower = id.to_lowercase();
        let unit_after = |rest: &str| {
            self.symbols
                .get(rest)
                .or_else(|| self.names.get(&rest.to_lowercase()))
                .or_else(|| custom.and_then(|c| c.get(rest)))
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

    fn insert(&mut self, entry: &Entry, unit: &Unit) {
        for s in &entry.symbols {
            self.symbols.entry(s).or_insert_with(|| unit.clone());
        }
        for n in &entry.names {
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
        let entries = entries();
        let mut pending: Vec<&Entry> = entries.iter().collect();
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

/// A unit as the registry builds it: a UDUNITS entry with its amendments, or an extra unit.
pub(crate) struct Entry {
    pub(crate) def: Def,
    pub(crate) names: Vec<&'static str>,
    pub(crate) symbols: Vec<&'static str>,
}

/// The UDUNITS units with [`AMENDMENTS`] applied, then [`EXTRA_UNITS`]. A symbol or name an
/// amendment or extra unit gives is taken from whichever other unit had it.
pub(crate) fn entries() -> Vec<Entry> {
    let lower = |s: &str| s.to_lowercase();
    let claimed_symbols: Vec<&str> = AMENDMENTS
        .iter()
        .flat_map(|a| a.symbols.iter())
        .chain(EXTRA_UNITS.iter().flat_map(|e| e.symbols.iter()))
        .copied()
        .collect();
    let claimed_names: Vec<String> = AMENDMENTS
        .iter()
        .flat_map(|a| a.names.iter())
        .chain(EXTRA_UNITS.iter().flat_map(|e| e.names.iter()))
        .map(|n| lower(n))
        .collect();
    let mut out: Vec<Entry> = UNITS
        .iter()
        .map(|u| {
            let amendment = AMENDMENTS.iter().find(|a| {
                u.names.iter().any(|n| lower(n) == lower(a.unit)) || u.symbols.contains(&a.unit)
            });
            let mut names: Vec<&'static str> = u
                .names
                .iter()
                .copied()
                .filter(|n| !claimed_names.contains(&lower(n)))
                .collect();
            let mut symbols: Vec<&'static str> = u
                .symbols
                .iter()
                .copied()
                .filter(|s| !claimed_symbols.contains(s))
                .collect();
            let mut def = u.def;
            if let Some(a) = amendment {
                names.extend(a.names);
                symbols.extend(a.symbols);
                if let Some(d) = a.def {
                    def = Def::Expr(d);
                }
            }
            Entry {
                def,
                names,
                symbols,
            }
        })
        .collect();
    out.extend(EXTRA_UNITS.iter().map(|e| Entry {
        def: e.def,
        names: e.names.to_vec(),
        symbols: e.symbols.to_vec(),
    }));
    out
}

/// The process-wide registry.
pub(crate) fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(Registry::build)
}

/// Add a unit to this process's registry: `define_unit("SFU", "1e-22 W m-2 Hz-1")`. It parses from
/// then on, by that exact spelling and with SI prefix symbols (`kSFU`), and files that name it
/// read and write it as written. Returns the unit.
///
/// Definitions are never removed, so a unit once parsed keeps its meaning; defining a name
/// again with an equal unit does nothing. Each compiled copy of this crate has its own registry
/// (two Python extension modules that both link it define their units separately).
///
/// ```
/// use annoray_units::{Unit, define_unit};
///
/// define_unit("SFU", "1e-22 W m-2 Hz-1")?;
/// let sfu: Unit = "SFU".parse()?;
/// assert_eq!(sfu.converter_to(&"W m-2 Hz-1".parse()?)?.factor(), 1e-22);
/// assert_eq!(sfu.to_string(), "SFU");
/// # Ok::<(), annoray_units::UnitError>(())
/// ```
///
/// # Errors
/// [`UnitError::Parse`] for a `name` that would not read back as one unit name (`m2` is `m`
/// squared), or a `definition` that does not parse; [`UnitError::Redefined`] for a name that
/// already means another unit, with or without a prefix.
pub fn define_unit(name: &str, definition: &str) -> Result<Unit, UnitError> {
    if !crate::parse::is_unit_name(name) {
        return Err(UnitError::Parse {
            input: name.to_owned(),
            pos: 0,
            reason: "a unit's name is letters, not ending in a digit".to_owned(),
        });
    }
    let unit = definition.parse::<Unit>()?.without_label();
    let reg = registry();
    let mut custom = CUSTOM.write().unwrap_or_else(PoisonError::into_inner);
    if let Some(existing) = reg.lookup_in(name, custom.as_ref()) {
        if existing != unit {
            return Err(UnitError::Redefined {
                name: name.to_owned(),
            });
        }
    } else {
        custom
            .get_or_insert_with(Custom::new)
            .insert(name.to_owned(), unit.clone());
        ANY_CUSTOM.store(true, Ordering::Release);
    }
    Ok(unit.with_label(name))
}
