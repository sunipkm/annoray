//! The table of interned units. It is process-wide, and the cap test changes its cap, so
//! these tests live in a binary of their own and run in one test function.

#![allow(clippy::unwrap_used, clippy::float_cmp)]

use annoray_units::{
    DEFAULT_INTERN_CAP, Quantity, Unit, UnitError, UnitId, intern_cap, interned, set_intern_cap,
};

#[test]
fn cap_spellings_and_threads() {
    assert_eq!(intern_cap(), DEFAULT_INTERN_CAP);

    // One entry per spelling; `1` is never an entry.
    let start = interned();
    let km = UnitId::parse("km").unwrap();
    assert!(km.same_spelling(UnitId::parse("  km").unwrap()));
    assert!(km.same_spelling(UnitId::intern("km".parse().unwrap()).unwrap()));
    let thousand_m = UnitId::parse("1000 m").unwrap();
    assert!(!km.same_spelling(thousand_m));
    assert_eq!(km, thousand_m);
    assert_eq!(thousand_m.to_string(), "1000 m");
    assert!(
        UnitId::intern(Unit::one())
            .unwrap()
            .same_spelling(UnitId::one())
    );
    assert_eq!(interned() - start, 2);

    // Epoch units are interned like any other; a quantity of one is `Copy`.
    let t = Quantity::<annoray_units::dims::Time>::new(3.0, "s since 1970-01-01").unwrap();
    let copy = t;
    assert_eq!(copy.unit().to_string(), t.unit().to_string());

    // Many threads, the same spellings: one entry each, the same handles everywhere.
    let before = interned();
    let handles: Vec<Vec<UnitId>> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..8)
            .map(|_| {
                s.spawn(|| {
                    (0..50)
                        .map(|i| UnitId::parse(&format!("days since 2000-01-{:02}", i % 28 + 1)))
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap()
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    assert_eq!(interned() - before, 28);
    for h in &handles[1..] {
        assert!(h.iter().zip(&handles[0]).all(|(a, b)| a.same_spelling(*b)));
    }

    // At the cap: known spellings still resolve, new ones are an error, plain units work.
    set_intern_cap(interned());
    assert!(UnitId::parse("km").unwrap().same_spelling(km));
    let full = UnitId::parse("furlong");
    assert_eq!(full.unwrap_err(), UnitError::InternFull { cap: interned() });
    assert!(matches!(
        Quantity::<annoray_units::Dyn>::new(1.0, "fortnight"),
        Err(UnitError::InternFull { .. })
    ));
    let m = Quantity::<annoray_units::Dyn>::new(2.0, "km").unwrap();
    assert!(matches!(m.powi(2), Err(UnitError::InternFull { .. })));
    assert_eq!((m * 3.0).unwrap().value(), 6.0); // the unit is kept: no new entry
    assert!("furlong".parse::<Unit>().is_ok());

    set_intern_cap(DEFAULT_INTERN_CAP);
    assert!(UnitId::parse("furlong").is_ok());
}
