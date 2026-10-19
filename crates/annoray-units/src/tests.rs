//! Crate-level tests: parsing against UDUNITS and CF, conversions, arithmetic rules.

use crate::time::Calendar;
use crate::{BaseDim, Dimension, LogBase, Rational, Scale, Unit, UnitError, UnitKind, UnitPolicy};

fn u(s: &str) -> Unit {
    s.parse().unwrap_or_else(|e| panic!("{s:?}: {e}"))
}

fn factor(from: &str, to: &str) -> f64 {
    u(from).converter_to(&u(to)).unwrap().factor()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-300)
}

#[test]
fn every_udunits_definition_resolves() {
    let reg = crate::registry::registry();
    assert!(
        reg.failures.is_empty(),
        "unresolved UDUNITS entries: {:?}",
        reg.failures
    );
}

#[test]
fn every_cf_canonical_unit_parses() {
    let table = include_str!("../../../fixtures/cf_canonical_units.txt");
    let mut n = 0;
    for line in table.lines().filter(|l| !l.starts_with('#')) {
        let parsed = line.parse::<Unit>();
        assert!(parsed.is_ok(), "{line:?}: {parsed:?}");
        n += 1;
    }
    assert_eq!(n, 115);
}

#[test]
fn spellings_are_equal() {
    assert_eq!(u("m/s"), u("m s-1"));
    assert_eq!(u("m/s"), u("m.s-1"));
    assert_eq!(u("m/s"), u("meters per second"));
    assert_eq!(u("kg.m2/s2"), u("J")); // UCUM style
    assert_eq!(u("kg m^2 s**-2"), u("joule"));
    assert_eq!(u("W m-2 sr-1 (m-1)-1"), u("W m-1 sr-1"));
    assert_eq!(u("hPa"), u("100 Pa"));
    assert_eq!(u("hectopascal"), u("hPa"));
    assert_eq!(u("mbar"), u("hPa"));
    assert_eq!(u("1e-3"), u("0.001"));
    assert_eq!(u("%"), u("percent"));
    assert_eq!(u("degrees_north"), u("degree"));
    assert_eq!(u("°C"), u("degC"));
    assert_eq!(u("Hz"), u("s-1"));
    assert_eq!(u(""), Unit::one());
    assert_eq!(u("1"), Unit::one());
}

#[test]
fn exact_scales() {
    // Hours, days and degrees are exact; equality has no tolerance.
    assert_eq!(u("h"), u("3600 s"));
    assert_eq!(u("day"), u("86400 s"));
    assert_eq!(u("min").scale().exponents(), (2, 1, 1, 0)); // 60 = 2² · 3 · 5
    assert_eq!(u("degree").scale().exponents(), (-2, -2, -1, 1)); // π/180
    assert_eq!(
        u("km").converter_to(&u("mm")).unwrap().exact_factor(),
        Some(Rational::from_int(1_000_000))
    );
    assert_eq!(
        u("inch").converter_to(&u("cm")).unwrap().exact_factor(),
        Some(Rational::new(127, 50).unwrap())
    );
    assert!(
        u("degree")
            .converter_to(&u("rad"))
            .unwrap()
            .exact_factor()
            .is_none()
    ); // involves π
}

#[test]
fn conversions() {
    assert_eq!(factor("km", "m"), 1000.0);
    assert!(close(factor("mm/day", "m/s"), 1e-3 / 86_400.0));
    assert_eq!(factor("degree_west", "degree_east"), -1.0);
    assert!(close(factor("degree", "rad"), std::f64::consts::PI / 180.0));
    assert!(close(factor("ft", "m"), 0.3048));
    assert!(close(factor("knot", "m/s"), 1852.0 / 3600.0));
    assert!(close(factor("dB", "B"), 0.1));
    assert!(close(factor("g/kg", "1"), 1e-3));

    let c_to_k = u("degC").converter_to(&u("K")).unwrap();
    assert!(close(c_to_k.apply(20.0), 293.15));
    let f_to_c = u("degF").converter_to(&u("degC")).unwrap();
    assert!(close(f_to_c.apply(212.0), 100.0));
    assert!(close(f_to_c.apply(-40.0), -40.0));
    assert!(f_to_c.exact_factor().is_none());

    assert!(matches!(
        u("m").converter_to(&u("s")),
        Err(UnitError::Incompatible { .. })
    ));
    assert!(u("dB").converter_to(&u("1")).is_err());
}

#[test]
fn angle_is_a_dimension() {
    assert_eq!(u("rad").dim(), Dimension::of(BaseDim::Angle));
    assert!(u("rad").converter_to(&u("1")).is_err());
    let erased = u("degree").erase_angle();
    assert!(close(
        erased.converter_to(&u("1")).unwrap().factor(),
        std::f64::consts::PI / 180.0
    ));
    assert_eq!(
        u("sr").dim(),
        Dimension::of(BaseDim::Angle).powi(2).unwrap()
    );
}

#[test]
fn arithmetic_rules() {
    let (m, km, s) = (u("m"), u("km"), u("s"));
    // Strict by default: same dimension, different scale must be rescaled explicitly.
    assert!(matches!(
        m.plan_add(&km, UnitPolicy::Strict),
        Err(UnitError::ScaleMismatch { .. })
    ));
    let plan = m.plan_add(&km, UnitPolicy::AutoRescale).unwrap();
    assert_eq!(plan.result, m);
    assert_eq!(plan.rhs.unwrap().factor(), 1000.0);
    assert!(
        m.plan_add(&u("meter"), UnitPolicy::Strict)
            .unwrap()
            .rhs
            .is_none()
    );
    assert!(matches!(
        m.plan_add(&s, UnitPolicy::AutoRescale),
        Err(UnitError::Incompatible { .. })
    ));

    // Multiplication keeps scales: m / ms carries a factor of 1000.
    let v = m.div(&u("ms")).unwrap();
    assert_eq!(v.dim(), u("m/s").dim());
    assert_eq!(v.converter_to(&u("m/s")).unwrap().factor(), 1000.0);
    assert_eq!(u("m2 s-2").root(2).unwrap(), u("m/s"));
    assert!(matches!(
        u("m3").root(2),
        Err(UnitError::InexactRoot { .. })
    ));

    // Affine temperatures.
    let (c, k) = (u("degC"), u("K"));
    assert!(matches!(
        c.plan_add(&c, UnitPolicy::AutoRescale),
        Err(UnitError::UnsupportedOp { .. })
    ));
    assert_eq!(c.plan_add(&k, UnitPolicy::Strict).unwrap().result, c); // °C + kelvin-sized step
    assert_eq!(c.plan_sub(&c, UnitPolicy::Strict).unwrap().result, k); // difference
    let plan = c.plan_sub(&u("degF"), UnitPolicy::AutoRescale).unwrap();
    assert_eq!(plan.result, k);
    assert!(close(plan.rhs.unwrap().apply(32.0), 0.0));
    assert!(k.plan_sub(&c, UnitPolicy::AutoRescale).is_err());
    assert!(matches!(c.mul(&m), Err(UnitError::UnsupportedOp { .. })));
    assert_eq!(c.delta().mul(&m).unwrap(), u("K m"));
    // UDUNITS drops offsets inside products, as CF's `kg degree_C m-2` needs.
    assert_eq!(u("kg degree_C m-2"), u("kg K m-2"));

    // Log units scale but do not multiply.
    assert!(matches!(u("dB").kind(), UnitKind::Log(r) if r.base == LogBase::Ten));
    assert!(u("dB").mul(&m).is_err());
}

#[test]
fn labels() {
    let x = u(" W m-2 ");
    assert_eq!(x.label(), Some("W m-2"));
    assert_eq!(x.to_string(), "W m-2");
    assert_eq!(x.clone().without_label().to_string(), "kg s-3");
    assert_eq!(u("km").without_label().to_string(), "1000 m");
    assert_eq!(u("degC").without_label().to_string(), "K @ 273.15");
    assert_eq!(
        Unit::linear(Dimension::NONE, Scale::from_decimal(1, -3).unwrap()).to_string(),
        "0.001"
    );
}

#[test]
fn epochs() {
    let t = u("days since 1970-01-01 00:00:00");
    assert!(matches!(t.kind(), UnitKind::Epoch(_)));
    let codec = t.epoch_codec(Calendar::Standard).unwrap();
    let d = codec.decode_int(19_000).unwrap();
    assert_eq!((d.year(), d.month(), d.day()), (2022, 1, 8));

    let h = u("hours since 2000-01-01T00:00:00Z")
        .epoch_codec(Calendar::NoLeap)
        .unwrap();
    assert_eq!(h.step_ns(), 3_600_000_000_000);
    assert!(matches!(
        u("years since 2000-01-01").epoch_codec(Calendar::Standard),
        Err(UnitError::UnsupportedTimeStep(_))
    ));
    assert_eq!(
        u("weeks since 2000-01-01")
            .epoch_codec(Calendar::Standard)
            .unwrap()
            .step_ns(),
        604_800_000_000_000
    );
    assert!(matches!(
        u("months since 2000-01-01").epoch_codec(Calendar::Standard),
        Err(UnitError::UnsupportedTimeStep(_))
    ));
    assert!(matches!(
        u("m").epoch_codec(Calendar::Standard),
        Err(UnitError::NotAnEpoch(_))
    ));
    assert!("m since 2000-01-01".parse::<Unit>().is_err());
    assert!(
        u("days since 1582-10-10")
            .epoch_codec(Calendar::Standard)
            .is_err()
    ); // in the gap
}

#[test]
fn bad_input_is_an_error_not_a_panic() {
    for s in [
        "flurbs",
        "m^",
        "m^x",
        "(m",
        "m)",
        "kg//s",
        "-1 m",
        "0 m",
        "m @",
        "K @ x",
        "lg(re",
        "°",
        "m**",
        "1e999999999 m",
        "m127",
        "m-999",
        "99999999999999999999999999999999999999999 m",
        "@",
        "since 2000-01-01",
        "s since 2000-02-30x",
        "*",
        "/m",
        "m//",
        "\u{0}",
        "m\u{301}",
    ] {
        let _ = s.parse::<Unit>(); // must return, whatever the result
    }
    assert!(matches!(
        "flurbs".parse::<Unit>(),
        Err(UnitError::UnknownUnit(_))
    ));
    assert!(matches!("(m".parse::<Unit>(), Err(UnitError::Parse { .. })));
    assert!(matches!(
        "0 m".parse::<Unit>(),
        Err(UnitError::Parse { .. })
    ));
}

mod properties {
    use proptest::prelude::*;

    use crate::time::{Calendar, CfDate, RefDate};
    use crate::{Unit, UnitPolicy};

    const UNITS: &[&str] = &[
        "m",
        "km",
        "mm",
        "inch",
        "ft",
        "nautical_mile",
        "s",
        "min",
        "h",
        "day",
        "ms",
        "kg",
        "g",
        "lb",
        "Pa",
        "hPa",
        "bar",
        "dbar",
        "W m-2",
        "degree",
        "rad",
        "arcsec",
        "K",
        "degC",
        "degF",
        "m/s",
        "knot",
        "J",
        "erg",
        "%",
        "1e-3",
    ];
    const PREFIXES: &[&str] = &["", "k", "M", "G", "m", "u", "n", "c", "d", "h", "da"];
    const CALENDARS: [Calendar; 6] = [
        Calendar::Standard,
        Calendar::ProlepticGregorian,
        Calendar::Julian,
        Calendar::NoLeap,
        Calendar::AllLeap,
        Calendar::Day360,
    ];

    fn unit() -> impl Strategy<Value = Unit> {
        proptest::sample::select(UNITS).prop_map(|s| s.parse().unwrap())
    }

    proptest! {
        /// Converting a → b → a is the identity; exactly so whenever the factor is rational.
        #[test]
        fn conversion_round_trips(a in unit(), b in unit(), x in -1e6f64..1e6) {
            let (Ok(ab), Ok(ba)) = (a.converter_to(&b), b.converter_to(&a)) else { return Ok(()) };
            if let (Some(f), Some(g)) = (ab.exact_factor(), ba.exact_factor()) {
                prop_assert_eq!(f.checked_mul(g).unwrap(), crate::Rational::ONE);
            }
            let back = ba.apply(ab.apply(x));
            prop_assert!((back - x).abs() <= 1e-9 * x.abs().max(1.0), "{} -> {} -> {}", x, ab.apply(x), back);
        }

        /// Prefixes are exact: (prefix p)(unit) equals p's factor times the unit, with no tolerance.
        #[test]
        fn prefixes_are_exact(p in proptest::sample::select(PREFIXES), q in proptest::sample::select(PREFIXES)) {
            let a: Unit = format!("{p}m").parse().unwrap();
            let b: Unit = format!("{q}m").parse().unwrap();
            let ab = a.converter_to(&b).unwrap().exact_factor().unwrap();
            let ba = b.converter_to(&a).unwrap().exact_factor().unwrap();
            prop_assert_eq!(ab.checked_mul(ba).unwrap(), crate::Rational::ONE);
            // Equal units need no conversion even under the strict policy.
            prop_assert!(a.plan_add(&a.clone().without_label(), UnitPolicy::Strict).unwrap().rhs.is_none());
        }

        /// Multiplication and division are inverse, and `==` agrees with an identity conversion.
        #[test]
        fn algebra(a in unit(), b in unit()) {
            let (Ok(ab), Ok(bb)) = (a.mul(&b), b.recip()) else { return Ok(()) };
            let back = ab.mul(&bb).unwrap();
            prop_assert_eq!(&back, &a.delta().without_label());
            prop_assert!(back.converter_to(&a.delta()).unwrap().is_identity());
        }

        /// Arbitrary text never panics the parser.
        #[test]
        fn parser_never_panics(s in "\\PC{0,24}") {
            let _ = s.parse::<Unit>();
        }

        /// Arbitrary unit-ish text never panics the parser either.
        #[test]
        fn unitish_never_panics(s in "[a-zA-Z0-9 ./^*()@:+-]{0,24}") {
            let _ = s.parse::<Unit>();
        }

        /// Integer offsets decode to dates and encode back exactly, in every calendar.
        #[test]
        fn epoch_round_trip(cal in proptest::sample::select(&CALENDARS[..]), n in -10_000_000_000i64..10_000_000_000, step in 0usize..5) {
            let units = ["days", "hours", "minutes", "seconds", "microseconds"][step];
            let codec = format!("{units} since 1850-01-01 06:00").parse::<Unit>().unwrap().epoch_codec(cal).unwrap();
            let d = codec.decode_int(n).unwrap();
            prop_assert_eq!(codec.encode_int(&d).unwrap(), n);
        }

        /// Reference dates in all calendars resolve, and chrono round-trips real-world calendars.
        #[test]
        fn chrono_round_trip(y in -3000i64..3000, m in 1u8..=12, d in 1u8..=28, cal in proptest::sample::select(&CALENDARS[..3])) {
            prop_assume!(y != 0 || cal.has_year_zero());
            let Ok(date) = CfDate::from_ymd(cal, y, m, d) else { return Ok(()) }; // e.g. the 1582 gap
            let dt = chrono::NaiveDateTime::try_from(date).unwrap();
            prop_assert_eq!(CfDate::from_chrono(dt, cal).unwrap(), date);
            let r = RefDate { year: y, month: m, day: d, hour: 0, minute: 0, second: 0, nanosecond: 0, utc_offset_minutes: 0 };
            prop_assert_eq!(r.in_calendar(cal).unwrap(), date);
        }
    }
}

/// Arithmetic keeps readable spellings (UDUNITS style) where the operands have them, and
/// falls back to the canonical form where it cannot; the units themselves are unaffected.
#[test]
fn labels_follow_arithmetic() {
    let show = |r: Result<Unit, UnitError>| r.unwrap().to_string();
    assert_eq!(show(u("km").div(&u("s"))), "km s-1");
    assert_eq!(show(u("km").powi(2)), "km2");
    assert_eq!(show(u("km/min").mul(&u("min"))), "km");
    assert_eq!(show(u("m/s").div(&u("s"))), "m s-2");
    assert_eq!(show(u("kg m2 s-2").div(&u("s"))), "kg m2 s-3");
    assert_eq!(show(u("m2 s-2").root(2)), "m s-1");
    assert_eq!(show(u("m").recip()), "m-1");
    // Same unit, whatever the spelling.
    assert_eq!(u("km").div(&u("s")).unwrap(), u("1000 m/s"));
    // No readable spelling to build on: canonical.
    assert_eq!(
        show(u("1000 m").mul(&u("s"))),
        u("1000 m").mul(&u("s")).unwrap().canonical()
    );
    // Everything cancels: the canonical (dimensionless) form.
    assert_eq!(u("m").div(&u("m")).unwrap(), Unit::one());
}

#[test]
fn reciprocal_integer_scales_divide_exactly() {
    // 700 m is exactly the float 0.7 km (dividing by 1000 rounds correctly; multiplying by
    // the inexact 0.001 would give 0.7000000000000001).
    let m_to_km = u("m").converter_to(&u("km")).unwrap();
    for i in 1..2000_i32 {
        let km = f64::from(i) / 10.0;
        assert_eq!(m_to_km.apply(f64::from(i) * 100.0), km, "{i}");
    }
    assert_eq!(u("min").converter_to(&u("h")).unwrap().apply(246.0), 4.1);
    // Integer scales still multiply, and the factor is unchanged.
    assert_eq!(u("km").converter_to(&u("m")).unwrap().apply(0.7), 700.0);
    assert_eq!(m_to_km.factor(), 0.001);
}

#[test]
fn every_amendment_names_a_udunits_unit() {
    for a in crate::extra::AMENDMENTS {
        let hits = crate::table::UNITS
            .iter()
            .filter(|u| {
                u.names.iter().any(|n| n.eq_ignore_ascii_case(a.unit))
                    || u.symbols.contains(&a.unit)
            })
            .count();
        assert_eq!(
            hits, 1,
            "amendment {:?} matches {hits} UDUNITS units",
            a.unit
        );
    }
}

#[test]
fn cgs_units_and_their_prefixes() {
    // Symbols take SI prefixes.
    for (a, b) in [
        ("G", "1e-4 T"),
        ("kG", "0.1 T"),
        ("mG", "1e-7 T"),
        ("uG", "1e-10 T"),
        ("dyn", "1e-5 N"),
        ("kdyn", "1e-2 N"),
        ("erg", "1e-7 J"),
        ("Mx", "1e-8 Wb"),
        ("P", "0.1 Pa s"),
        ("cP", "1e-3 Pa s"),
        ("St", "1e-4 m2/s"),
        ("cSt", "1e-6 m2/s"),
        ("Ba", "0.1 Pa"),
        ("Gal", "0.01 m/s2"),
        ("mGal", "1e-5 m/s2"),
        ("abA", "10 A"),
        ("abC", "10 C"),
        ("abV", "1e-8 V"),
        ("Jy", "1e-26 W m-2 Hz-1"),
        ("mJy", "1e-29 W m-2 Hz-1"),
        ("g cm-3", "1000 kg m-3"),
        ("erg s-1 cm-2", "1e-3 W m-2"),
    ] {
        assert_eq!(
            u(a).converter_to(&u(b)).unwrap().exact_factor(),
            Some(Rational::from_int(1)),
            "{a} = {b}"
        );
    }
    // Gaussian electrostatic units, exact through c.
    let c: i128 = 299_792_458;
    assert_eq!(
        u("C").converter_to(&u("statC")).unwrap().exact_factor(),
        Some(Rational::from_int(10 * c))
    );
    assert_eq!(u("Fr"), u("statC"));
    assert_eq!(u("esu"), u("statC"));
    assert_eq!(
        u("statV").converter_to(&u("V")).unwrap().exact_factor(),
        Some(Rational::new(c, 1_000_000).unwrap())
    );
    assert_eq!(u("statA"), u("statC/s"));
    assert!(close(factor("statohm", "ohm"), 8.987_551_787_368_176e11));
    assert!(close(factor("statF", "F"), 1.112_650_056_053_618_4e-12));
    // Oersted: 1000/4π A/m.
    assert!(close(
        factor("Oe", "A/m"),
        1000.0 / (4.0 * std::f64::consts::PI)
    ));
}

#[test]
fn us_customary_and_imperial_units_are_exact() {
    let exact = |a: &str, b: &str| u(a).converter_to(&u(b)).unwrap().exact_factor();
    let r = |n: i128, d: i128| Some(Rational::new(n, d).unwrap());
    assert_eq!(exact("gallon", "in3"), r(231, 1));
    assert_eq!(exact("US_gallon", "L"), r(3_785_411_784, 1_000_000_000));
    assert_eq!(exact("qt", "gallon"), r(1, 4));
    assert_eq!(exact("pt", "gallon"), r(1, 8));
    assert_eq!(exact("fl_oz", "gallon"), r(1, 128));
    assert_eq!(exact("UK_gallon", "L"), r(454_609, 100_000));
    assert_eq!(exact("imperial_pint", "UK_gallon"), r(1, 8));
    assert_eq!(exact("oz", "g"), r(28_349_523_125, 1_000_000_000));
    assert_eq!(exact("ounce", "lb"), r(1, 16));
    assert_eq!(exact("lbm", "kg"), r(45_359_237, 100_000_000));
    assert_eq!(exact("ozt", "g"), r(311_034_768, 10_000_000));
    assert_eq!(exact("stone", "lb"), r(14, 1));
    assert_eq!(exact("long_hundredweight", "lb"), r(112, 1));
    assert_eq!(exact("nmi", "m"), r(1852, 1));
    assert_eq!(exact("mph", "km/h"), r(1_609_344, 1_000_000));
    assert_eq!(exact("kph", "m/s"), r(5, 18));
    assert_eq!(exact("fathom", "m"), r(18_288, 10_000));
    assert_eq!(exact("furlong", "mile"), r(1, 8));
    assert_eq!(exact("chain", "foot"), r(66, 1));
    assert_eq!(exact("bushel", "in3"), r(215_042, 100));
    assert!(close(factor("hp", "W"), 745.699_871_582_270_2));
    assert!(close(factor("ozf", "N"), 0.278_013_850_953_781_2));
    // The survey foot keeps its own name.
    assert_eq!(exact("US_survey_foot", "m"), r(1200, 3937));
}

#[test]
fn symbols_that_used_to_mean_something_else() {
    // Not a nano-mile, a milli-phot, a kilo-phot, or (as in UDUNITS) a fluid ounce.
    assert!(u("nmi").is_convertible(&u("m")), "nmi");
    assert!(u("mph").is_convertible(&u("m/s")), "mph");
    assert!(u("kph").is_convertible(&u("m/s")), "kph");
    assert!(u("oz").is_convertible(&u("kg")), "oz");
    assert!(u("floz").is_convertible(&u("m3")), "floz");
    // Kept as UDUNITS (and CF data) have them: `gal` is the galileo, `pt` the US pint.
    assert_eq!(u("gal"), u("Gal"));
    assert_eq!(u("mgal"), u("mGal"));
    assert_eq!(u("pt"), u("US_liquid_pint"));
    // Unprefixed CGS symbols leave the prefix meanings intact.
    assert_eq!(u("GHz"), u("1e9 Hz"));
    assert_eq!(u("PW"), u("1e15 W"));
    assert_eq!(u("Pa"), u("N/m2"));
    assert_eq!(u("Gy"), u("J/kg"));
    assert_eq!(u("Gs"), u("1e9 s"));
}

#[test]
fn physical_constants_are_current() {
    let exact = |a: &str, b: &str| u(a).converter_to(&u(b)).unwrap().exact_factor();
    let r = |n: i128, d: i128| Some(Rational::new(n, d).unwrap());
    // 2019 SI: exact.
    assert_eq!(exact("eV", "1.602176634e-19 J"), r(1, 1));
    assert_eq!(exact("e", "1.602176634e-19 C"), r(1, 1));
    assert_eq!(exact("ly", "m"), r(9_460_730_472_580_800, 1));
    assert_eq!(exact("au", "m"), r(149_597_870_700, 1));
    assert!(close(factor("pc", "m"), 3.085_677_581_491_367e16));
    assert!(close(factor("Da", "kg"), 1.660_539_068_92e-27));
    assert_eq!(u("Da"), u("u"));
    // The torr is 1/760 atm; the conventional mmHg is not.
    assert_eq!(exact("760 Torr", "atm"), r(1, 1));
    assert!(close(factor("mmHg", "Pa"), 133.322_387_415));
    assert_eq!(exact("kWh", "MJ"), r(18, 5));
}
