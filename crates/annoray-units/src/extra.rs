//! Additions to the UDUNITS-2 database (`table.rs`, generated): exact definitions where
//! UDUNITS rounds, the usual symbols of CGS and US/imperial units (so they also take SI
//! prefixes: `kG`, `cP`, `mJy`), and a few units UDUNITS lacks.
//!
//! - **Exact values.** UDUNITS gives many customary units to 7 digits (the US gallon as
//!   3.785412 L); here they are defined from their legal definitions (231 in³), so
//!   conversions keep annoray's exact scales. Sources: NIST SP 811 (2008), NIST Handbook 44
//!   (2024) Appendix C, and the exact speed of light for the Gaussian units.
//! - **US survey foot.** Since 2023 NIST bases the fathom, chain, rod and furlong on the
//!   international foot; the survey-foot units keep their own names (`US_survey_foot`, …).
//! - **CGS electromagnetic units** are mapped onto SI quantities by their standard
//!   correspondence (1 G = 1e-4 T, 1 statC = 1/2997924580 C, 1 Oe = 1000/4π A/m), as UDUNITS
//!   and astropy do. The units convert; equations in Gaussian form still differ from their SI
//!   form by factors of 4π and c.
//! - **Symbols that would otherwise parse as something else** are given their usual meaning:
//!   `nmi` is the nautical mile (not a nano-mile), `mph` miles per hour and `kph` kilometres
//!   per hour (not milli- and kilo-phot), `oz` the avoirdupois ounce (UDUNITS: the US fluid
//!   ounce, which keeps `fl_oz` and `floz`).
//! - **Physical constants** are the 2019 SI's exact values (UDUNITS predates them: its
//!   electronvolt is CODATA 1986's) or, where not exact, CODATA 2022's; the parsec and light
//!   year follow from the exact astronomical unit and the Julian year (IAU).
//! - **Kept as UDUNITS has them**, because CF data use them: `gal`/`Gal` is the galileo
//!   (gravity anomalies in `mGal`/`mgal`), not the gallon (`gallon`, `US_gallon`,
//!   `UK_gallon`); `pt` is the US liquid pint.

use crate::registry::{Def, UnitEntry};

/// A change to a UDUNITS unit, found by any of its names or symbols: an exact definition, and
/// names and symbols added to it. Symbols and names given here are taken from any other unit
/// that had them.
pub(crate) struct Amend {
    /// A name or symbol of the UDUNITS unit.
    pub(crate) unit: &'static str,
    /// A new definition, if UDUNITS' is rounded or outdated.
    pub(crate) def: Option<&'static str>,
    /// Names added (case-insensitive).
    pub(crate) names: &'static [&'static str],
    /// Symbols added (case-sensitive; they take SI prefixes).
    pub(crate) symbols: &'static [&'static str],
}

const fn amend(
    unit: &'static str,
    def: Option<&'static str>,
    names: &'static [&'static str],
    symbols: &'static [&'static str],
) -> Amend {
    Amend {
        unit,
        def,
        names,
        symbols,
    }
}

pub(crate) static AMENDMENTS: &[Amend] = &[
    // ----- CGS -----
    amend("dyne", None, &[], &["dyn"]),
    amend("gauss", None, &[], &["G"]),
    amend("maxwell", None, &[], &["Mx"]),
    amend("poise", None, &[], &["P"]),
    amend("barye", None, &[], &["Ba"]),
    amend("oersted", Some("250/pi A/m"), &[], &[]),
    amend("abampere", None, &[], &["abA"]),
    amend("biot", None, &[], &["Bi"]),
    amend("abvolt", None, &[], &["abV"]),
    amend("abohm", None, &[], &["abΩ"]),
    // Gaussian electrostatic units, exact through c = 299 792 458 m/s.
    amend(
        "statcoulomb",
        Some("C/2997924580"),
        &["franklin", "franklins"],
        &["statC", "Fr", "esu"],
    ),
    amend("statampere", Some("statcoulomb/s"), &[], &["statA"]),
    amend("statvolt", Some("299.792458 V"), &[], &["statV"]),
    amend("statohm", Some("statvolt/statampere"), &[], &["statΩ"]),
    amend("statfarad", Some("statcoulomb/statvolt"), &[], &["statF"]),
    amend("stathenry", Some("statohm s"), &[], &["statH"]),
    amend("statmho", Some("1/statohm"), &[], &[]),
    // ----- US customary and imperial -----
    amend("nautical_mile", None, &[], &["nmi"]),
    amend("avoirdupois_pound", None, &[], &["lbm"]),
    amend(
        "avoirdupois_ounce",
        Some("avoirdupois_pound/16"),
        &["ounce", "ounces"],
        &["oz"],
    ),
    amend("troy_ounce", Some("31.1034768 g"), &[], &["ozt"]),
    amend("troy_pound", Some("12 troy_ounce"), &[], &[]),
    amend(
        "long_hundredweight",
        Some("112 avoirdupois_pound"),
        &[],
        &[],
    ),
    amend(
        "short_hundredweight",
        Some("100 avoirdupois_pound"),
        &[],
        &[],
    ),
    amend(
        "force_ounce",
        Some("avoirdupois_ounce 9.80665 m s-2"),
        &[],
        &[],
    ),
    amend("poundal", Some("avoirdupois_pound foot s-2"), &[], &[]),
    amend(
        "US_liquid_gallon",
        Some("231 in^3"),
        &["US_gallon", "US_gallons"],
        &[],
    ),
    amend("US_liquid_quart", None, &["US_quart", "US_quarts"], &["qt"]),
    amend("US_liquid_pint", None, &["US_pint", "US_pints"], &[]),
    amend("US_fluid_ounce", None, &[], &["fl_oz"]),
    amend("US_dry_gallon", Some("268.8025 in^3"), &[], &[]),
    amend("bushel", Some("2150.42 in^3"), &[], &[]),
    amend(
        "UK_liquid_gallon",
        None,
        &[
            "UK_gallon",
            "UK_gallons",
            "imperial_gallon",
            "imperial_gallons",
        ],
        &[],
    ),
    amend(
        "UK_liquid_quart",
        None,
        &["UK_quart", "UK_quarts", "imperial_quart", "imperial_quarts"],
        &[],
    ),
    amend(
        "UK_liquid_pint",
        None,
        &["UK_pint", "UK_pints", "imperial_pint", "imperial_pints"],
        &[],
    ),
    amend(
        "UK_fluid_ounce",
        None,
        &["imperial_fluid_ounce", "imperial_fluid_ounces"],
        &[],
    ),
    // NIST (2023): these are now based on the international foot.
    amend("fathom", Some("6 foot"), &[], &[]),
    amend("chain", Some("66 foot"), &[], &[]),
    amend("rod", Some("16.5 foot"), &[], &[]),
    amend("furlong", Some("660 foot"), &[], &[]),
    amend("horsepower", Some("550 foot force_pound/s"), &[], &[]),
    amend("footcandle", Some("lm/foot^2"), &[], &[]),
    // ----- physical constants and related units -----
    amend("electronvolt", Some("1.602176634e-19 J"), &[], &[]),
    amend("e", Some("1.602176634e-19 C"), &[], &[]),
    // CODATA 2022 (not exact).
    amend(
        "atomic_mass_unit",
        Some("1.66053906892e-27 kg"),
        &["dalton", "daltons"],
        &["Da"],
    ),
    amend("light_year", Some("9460730472580800 m"), &[], &["ly"]),
    amend("parsec", Some("648000/pi astronomical_unit"), &[], &["pc"]),
    amend("watthour", None, &[], &["Wh"]),
];

/// Units UDUNITS does not have.
pub(crate) static EXTRA_UNITS: &[UnitEntry] = &[
    UnitEntry {
        def: Def::Expr("10 C"),
        names: &["abcoulomb", "abcoulombs"],
        symbols: &["abC"],
    },
    UnitEntry {
        def: Def::Expr("1e-26 W m-2 Hz-1"),
        names: &["jansky", "janskys"],
        symbols: &["Jy"],
    },
    UnitEntry {
        def: Def::Expr("14 avoirdupois_pound"),
        names: &["stone", "stones"],
        symbols: &[],
    },
    // Exactly 1/760 atmosphere; UDUNITS makes it a name of the (conventional) mmHg, which is
    // 133.322387415 Pa.
    UnitEntry {
        def: Def::Expr("101325/760 Pa"),
        names: &["torr", "torrs"],
        symbols: &["Torr"],
    },
    UnitEntry {
        def: Def::Expr("mile/hour"),
        names: &["mile_per_hour", "miles_per_hour"],
        symbols: &["mph"],
    },
    UnitEntry {
        def: Def::Expr("km/hour"),
        names: &["kilometer_per_hour", "kilometers_per_hour"],
        symbols: &["kph"],
    },
];
