//! A `units` attribute annoray cannot parse: kept as a plain attribute (with a warning), so the
//! variable has no unit and refuses conversion, and written back unchanged.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use annoray::io::{OpenOptions, ToNetcdf, open_dataset_with};
use annoray::ndarray::{ArrayD, IxDyn};
use annoray::{AttrValue, Dataset, Variable};

#[test]
fn unknown_units_round_trip_as_attributes() {
    let v = Variable::new(["x"], ArrayD::from_elem(IxDyn(&[3]), 1.0))
        .unwrap()
        .with_attrs([("units", "counts per detector pixel")]);
    let dir = std::env::temp_dir().join(format!("annoray-unknown-units-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("u.nc");
    Dataset::new()
        .with_var("v", v)
        .unwrap()
        .to_netcdf(&path)
        .unwrap();

    let opened = open_dataset_with(&path, &OpenOptions::default()).unwrap();
    assert!(
        opened
            .warnings
            .iter()
            .any(|w| w.contains("v: units \"counts per detector pixel\" kept as an attribute")),
        "{:?}",
        opened.warnings
    );
    let v = &opened.dataset.data_vars()["v"];
    assert!(v.unit().is_none());
    assert_eq!(
        v.attrs().get("units"),
        Some(&AttrValue::Str("counts per detector pixel".into()))
    );
    // No unit: conversion is refused rather than guessed.
    let da = opened.dataset.get("v").unwrap();
    assert!(da.to_unit(&"m".parse().unwrap()).is_err());

    // Written back exactly as read.
    let again = dir.join("again.nc");
    opened.dataset.to_netcdf(&again).unwrap();
    let back = open_dataset_with(&again, &OpenOptions::default()).unwrap();
    assert_eq!(
        back.dataset.data_vars()["v"].attrs().get("units"),
        Some(&AttrValue::Str("counts per detector pixel".into()))
    );
}
