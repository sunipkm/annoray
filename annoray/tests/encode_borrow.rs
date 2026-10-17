//! Encoding for writing borrows values that are written unchanged and owns only those an
//! encoding step changes.
#![allow(clippy::unwrap_used, clippy::panic)]

use annoray::cf::{EncodeOptions, encode};
use annoray::io::{Format, RawData};
use annoray::{AttrValue, Dataset, Variable};
use ndarray::array;

#[test]
fn unchanged_values_are_borrowed() {
    let mut ds = Dataset::new()
        .with_var(
            "t",
            Variable::new(["x"], array![1.0f32, f32::NAN, 3.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var(
            "packed",
            Variable::new(["x"], array![1.0f64, 2.0, 3.0].into_dyn()).unwrap(),
        )
        .unwrap()
        .with_var(
            "flag",
            Variable::new(["x"], array![true, false, true].into_dyn()).unwrap(),
        )
        .unwrap();
    ds.encoding_of_mut("packed").unwrap().scale_factor = Some(AttrValue::F64(vec![0.5]));
    let raw = encode(&ds, &EncodeOptions::format(Format::Netcdf4)).unwrap();
    let kind = |n: &str| match &raw.vars.get(n).unwrap().data {
        RawData::Borrowed(_) => "borrowed",
        RawData::Loaded(_) => "owned",
        RawData::Lazy(_) => "lazy",
    };
    assert_eq!(kind("t"), "borrowed");
    assert_eq!(kind("packed"), "owned");
    assert_eq!(kind("flag"), "owned");
    // Classic formats without unsigned integers convert them: owned.
    let u = Dataset::new()
        .with_var(
            "u",
            Variable::new(["x"], array![1u16, 2].into_dyn()).unwrap(),
        )
        .unwrap();
    let raw = encode(&u, &EncodeOptions::format(Format::Netcdf3Classic)).unwrap();
    assert!(matches!(
        raw.vars.get("u").unwrap().data,
        RawData::Loaded(_)
    ));
}
