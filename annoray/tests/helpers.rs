//! Small conveniences: `date_range`, `to_f64` and `DataArray::scalar`.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use annoray::Instant;
use annoray::{CoreError, DataArray, Variable, date_range};
use ndarray::array;

fn at(y: i64, m: u8, d: u8, h: u8) -> Instant {
    Instant::from_ymd_hms_nano((y, m, d), (h, 0, 0), 0).unwrap()
}

#[test]
fn date_ranges_follow_pandas() {
    assert_eq!(
        date_range("2000-01-01", 3, "D").unwrap().to_vec(),
        [at(2000, 1, 1, 0), at(2000, 1, 2, 0), at(2000, 1, 3, 0)]
    );
    assert_eq!(
        date_range("2000-02-28T18:00", 3, "6h").unwrap().to_vec(),
        [at(2000, 2, 28, 18), at(2000, 2, 29, 0), at(2000, 2, 29, 6)]
    );
    // Period starts begin at the first one at or after the start.
    assert_eq!(
        date_range("2000-01-15", 2, "MS").unwrap().to_vec(),
        [at(2000, 2, 1, 0), at(2000, 3, 1, 0)]
    );
    assert_eq!(
        date_range("2000-03-01", 2, "MS").unwrap().to_vec(),
        [at(2000, 3, 1, 0), at(2000, 4, 1, 0)]
    );
    assert_eq!(
        date_range("2000-02-01", 2, "QS").unwrap().to_vec(),
        [at(2000, 4, 1, 0), at(2000, 7, 1, 0)]
    );
    assert_eq!(
        date_range("2000-03-01", 2, "YS-JUL").unwrap().to_vec(),
        [at(2000, 7, 1, 0), at(2001, 7, 1, 0)]
    );
    assert_eq!(
        date_range("2000-01-01", 2, "2MS").unwrap().to_vec(),
        [at(2000, 1, 1, 0), at(2000, 3, 1, 0)]
    );
    assert!(date_range("2000-01-01", 0, "D").unwrap().is_empty());
    assert!(matches!(
        date_range("2000-01-01", 2, "ME"),
        Err(CoreError::InvalidReduction(_))
    ));
    assert!(matches!(
        date_range("2000-13-01", 2, "D"),
        Err(CoreError::InvalidLabel { .. })
    ));
    assert!(matches!(
        date_range("2000-01-01", 2, "fortnight"),
        Err(CoreError::InvalidReduction(_))
    ));
}

#[test]
fn to_f64_and_scalars() {
    let b = Variable::new(["x"], array![true, false].into_dyn()).unwrap();
    assert_eq!(b.to_f64().unwrap().into_raw_vec_and_offset().0, [1.0, 0.0]);
    let i = DataArray::from_array(["x"], array![1i16, -2].into_dyn()).unwrap();
    assert_eq!(i.to_f64().unwrap().into_raw_vec_and_offset().0, [1.0, -2.0]);
    let s = Variable::new(["x"], array!["a".to_string()].into_dyn()).unwrap();
    assert!(matches!(
        s.to_f64(),
        Err(CoreError::UnsupportedDType { .. })
    ));
    let q = DataArray::scalar(2.5)
        .unwrap()
        .with_unit("s".parse().unwrap());
    assert_eq!(q.ndim(), 0);
    assert_eq!(q.to_f64().unwrap().into_raw_vec_and_offset().0, [2.5]);
    assert_eq!(q.unit().unwrap().to_string(), "s");
}
