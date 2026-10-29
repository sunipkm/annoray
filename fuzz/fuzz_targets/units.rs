//! The unit parser, and arithmetic and conversion of what it accepts.
#![no_main]

use annoray::Unit;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else { return };
    let (a, b) = s.split_once('|').unwrap_or((s, "m"));
    let (Ok(a), Ok(b)) = (a.parse::<Unit>(), b.parse::<Unit>()) else {
        return;
    };
    let _ = a.to_string().parse::<Unit>();
    let _ = a.canonical();
    let _ = a.mul(&b);
    let _ = a.div(&b);
    let _ = a.powi(3);
    let _ = a.root(2);
    if let Ok(c) = a.converter_to(&b) {
        let _ = c.apply(1.5);
    }
});
