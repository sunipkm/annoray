//! quantile, idxmin/idxmax, weighted and coarsen (values as numpy/xarray give them; the
//! Python suite compares against xarray exhaustively).
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]

use annoray::{Boundary, CoarsenOptions, DataArray, QuantileMethod, Side};
use ndarray::{ArrayD, IxDyn, array};

const NAN: f64 = f64::NAN;

fn vals(d: &DataArray) -> Vec<f64> {
    d.to_f64().unwrap().iter().copied().collect()
}

fn grid() -> DataArray {
    DataArray::from_array(
        ["y", "x"],
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 8.0, NAN, 6.0, 5.0])
            .unwrap(),
    )
    .unwrap()
    .with_dim_coord("x", array![0.0, 10.0, 20.0, 30.0])
    .unwrap()
    .with_unit("m".parse().unwrap())
}

#[test]
fn quantiles() {
    let g = grid();
    // numpy: quantile([1, 2, 3, 4], 0.5) = 2.5; [8, 6, 5] (NaN skipped) → 6.
    let q = g
        .quantile(&[0.5], true, "x", QuantileMethod::Linear, true)
        .unwrap();
    assert_eq!(vals(&q), [2.5, 6.0]);
    assert_eq!(q.unit(), g.unit());
    let qs = g
        .quantile(&[0.0, 1.0], false, "x", QuantileMethod::Linear, true)
        .unwrap();
    assert_eq!(qs.dims(), ["quantile", "y"]);
    assert_eq!(vals(&qs), [1.0, 5.0, 4.0, 8.0]);
    let nearest = g
        .quantile(&[0.5], true, "x", QuantileMethod::Nearest, true)
        .unwrap();
    assert_eq!(vals(&nearest)[0], 3.0); // index 1.5 → 2 (ties to even), as numpy
    assert!(
        g.quantile(&[1.5], true, "x", QuantileMethod::Linear, true)
            .is_err()
    );
}

#[test]
fn label_of_extremes() {
    let g = grid();
    assert_eq!(vals(&g.idxmax("x", true).unwrap()), [30.0, 0.0]);
    assert_eq!(vals(&g.idxmin("x", true).unwrap()), [0.0, 30.0]);
    // Without skipna, NaN is the extreme (numpy's argmin), as xarray.
    assert_eq!(vals(&g.idxmin("x", false).unwrap()), [0.0, 10.0]);
    assert!(g.idxmin("y", true).is_err()); // no coordinate along y
}

#[test]
fn weighted_mean_skips_nan_data() {
    let g = grid();
    let w = DataArray::from_array(["x"], array![1.0, 1.0, 2.0, 0.0].into_dyn())
        .unwrap()
        .with_dim_coord("x", array![0.0, 10.0, 20.0, 30.0])
        .unwrap();
    let m = g.weighted(&w).unwrap().mean("x").unwrap();
    // Row 0: (1 + 2 + 6) / 4; row 1: (8 + 12) / 3 (the NaN's weight is not counted).
    let got = vals(&m);
    assert!((got[0] - 9.0 / 4.0).abs() < 1e-12 && (got[1] - 20.0 / 3.0).abs() < 1e-12);
    assert_eq!(m.unit(), g.unit());
}

#[test]
fn coarsen_blocks() {
    let g = grid();
    let c = g.coarsen(&[("x", 2)], &CoarsenOptions::default()).unwrap();
    assert_eq!(vals(&c.mean().unwrap()), [1.5, 3.5, 8.0, 5.5]);
    assert_eq!(
        c.mean().unwrap().coords()["x"]
            .to_f64()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [5.0, 25.0]
    );
    assert!(g.coarsen(&[("x", 3)], &CoarsenOptions::default()).is_err());
    let trim = CoarsenOptions::default()
        .boundary(Boundary::Trim)
        .side(Side::Right);
    assert_eq!(
        vals(&g.coarsen(&[("x", 3)], &trim).unwrap().sum().unwrap()),
        [9.0, 11.0]
    );
    let pad = CoarsenOptions::default().boundary(Boundary::Pad);
    assert_eq!(
        vals(&g.coarsen(&[("x", 3)], &pad).unwrap().max().unwrap()),
        [3.0, 4.0, 8.0, 5.0]
    );
}
