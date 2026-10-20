//! Time zones of date-time variables: wall-clock components, resampling, selection, display,
//! propagation through operations, and netCDF round trips.
#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]

use annoray::io::{ToNetcdf, open_dataset};
use annoray::{
    BinOp, ConcatOptions, DataArray, Dataset, Disambiguation, Duration, Instant, TimeZone,
    Variable, concat, date_range_tz,
};
use ndarray::{Array1, ArrayD, IxDyn};

fn at(s: &str) -> Instant {
    s.parse().unwrap()
}

fn ny() -> TimeZone {
    "America/New_York".parse().unwrap()
}

/// Hourly values 0, 1, … at `times`, on a `time` coordinate in `tz`.
fn series(times: &[Instant], tz: Option<TimeZone>) -> DataArray {
    let mut time = Variable::new(["time"], Array1::from(times.to_vec()).into_dyn()).unwrap();
    time.set_time_zone(tz).unwrap();
    #[allow(clippy::cast_precision_loss)]
    let values: Vec<f64> = (0..times.len()).map(|i| i as f64).collect();
    DataArray::from_array(
        ["time"],
        ArrayD::from_shape_vec(IxDyn(&[times.len()]), values).unwrap(),
    )
    .unwrap()
    .with_coord("time", time)
    .unwrap()
}

fn hours(start: &str, n: i64) -> Vec<Instant> {
    let t0 = at(start);
    (0..n)
        .map(|h| t0.checked_add(Duration::from_hours(h).unwrap()).unwrap())
        .collect()
}

fn ints(v: &Variable) -> Vec<i64> {
    v.data::<i64>().unwrap().iter().copied().collect()
}

#[test]
fn only_date_times_take_a_zone() {
    let mut v = Variable::new(["x"], Array1::from(vec![1.0]).into_dyn()).unwrap();
    assert!(v.set_time_zone(Some(ny())).is_err());
    assert!(v.set_time_zone(None).is_ok());
}

#[test]
fn components_are_wall_clock_times() {
    // 2024-07-01 02:00 UTC is 2024-06-30 22:00 in New York.
    let da = series(&[at("2024-07-01 02:00")], Some(ny()));
    let time = da.coord("time").unwrap();
    assert_eq!(ints(&time.dt("hour").unwrap()), [22]);
    assert_eq!(ints(&time.dt("day").unwrap()), [30]);
    assert_eq!(ints(&time.dt("dayofweek").unwrap()), [6]); // a Sunday
    let utc = time.tz_convert(None).unwrap();
    assert_eq!(ints(&utc.dt("hour").unwrap()), [2]);
    // Converting keeps the instants.
    assert_eq!(
        utc.data::<Instant>().unwrap(),
        time.data::<Instant>().unwrap()
    );
}

#[test]
fn localize_reads_wall_clock_times_in_a_zone() {
    let da = series(&[at("2024-07-01 08:00"), at("2024-11-03 01:30")], None);
    let time = da.coord("time").unwrap();
    let local = time
        .tz_localize(Some(ny()), Disambiguation::Earlier)
        .unwrap();
    assert_eq!(local.time_zone(), Some(&ny()));
    assert_eq!(
        local
            .data::<Instant>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [at("2024-07-01 12:00"), at("2024-11-03 05:30")]
    );
    // 01:30 on 2024-11-03 happens twice in New York.
    assert!(time.tz_localize(Some(ny()), Disambiguation::Raise).is_err());
    // Back to UTC wall clocks: the original values.
    let back = local.tz_localize(None, Disambiguation::Raise).unwrap();
    assert_eq!(
        back.data::<Instant>().unwrap(),
        time.data::<Instant>().unwrap()
    );
    assert_eq!(back.time_zone(), None);
}

#[test]
fn resampling_bins_on_local_days() {
    // 48 hours from midnight in New York (04:00 UTC in summer).
    let da = series(&hours("2024-07-01 04:00", 48), Some(ny()));
    let daily = da.resample("time", "D").unwrap().mean().unwrap();
    let time = daily.coord("time").unwrap();
    assert_eq!(time.time_zone(), Some(&ny()));
    assert_eq!(
        time.data::<Instant>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [at("2024-07-01 04:00"), at("2024-07-02 04:00")]
    );
    assert_eq!(
        daily
            .data::<f64>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [11.5, 35.5]
    );
    // The same instants binned on UTC days.
    let utc = series(&hours("2024-07-01 04:00", 48), None);
    let daily = utc.resample("time", "D").unwrap().mean().unwrap();
    assert_eq!(daily.len(), 3);
}

#[test]
fn days_across_a_daylight_saving_change() {
    let days = date_range_tz("2024-03-09", 3, "D", &ny()).unwrap();
    assert_eq!(
        days.to_vec(),
        [
            at("2024-03-09 05:00"),
            at("2024-03-10 05:00"),
            at("2024-03-11 04:00")
        ]
    );
    // Shorter steps are exact durations.
    let h = date_range_tz("2024-03-10 01:00", 2, "h", &ny()).unwrap();
    assert_eq!(h.to_vec(), [at("2024-03-10 06:00"), at("2024-03-10 07:00")]);
    // A start with an offset is an instant.
    let m = date_range_tz("2024-01-15T00:00Z", 1, "MS", &ny()).unwrap();
    assert_eq!(m.to_vec(), [at("2024-02-01 05:00")]);
}

#[test]
fn date_strings_select_wall_clock_times() {
    let da = series(&hours("2024-07-01 00:00", 48), Some(ny()));
    // 2024-07-01 in New York is 04:00 UTC on the 1st to 04:00 UTC on the 2nd.
    let day = da.sel([("time", "2024-07-01")]).unwrap();
    let t = day
        .coord("time")
        .unwrap()
        .data::<Instant>()
        .unwrap()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(t.first(), Some(&at("2024-07-01 04:00")));
    assert_eq!(t.last(), Some(&at("2024-07-02 03:00")));
    assert_eq!(day.coord("time").unwrap().time_zone(), Some(&ny()));
}

#[test]
fn operations_keep_the_zone() {
    let da = series(&hours("2024-07-01 00:00", 3), Some(ny()));
    let time = DataArray::new(da.coord("time").unwrap().to_owned());
    let hour = DataArray::scalar(Duration::from_hours(1).unwrap()).unwrap();
    let later = time.binary(BinOp::Add, &hour, Default::default()).unwrap();
    assert_eq!(later.time_zone(), Some(&ny()));
    let gap = later.binary(BinOp::Sub, &time, Default::default()).unwrap();
    assert_eq!(gap.time_zone(), None);
    let first = time.isel([("time", 0)]).unwrap();
    assert_eq!(first.time_zone(), Some(&ny()));
    let both = concat(&[&da, &da], "time", &ConcatOptions::default()).unwrap();
    assert_eq!(both.coord("time").unwrap().time_zone(), Some(&ny()));
    // Joining zones is refused.
    let utc = series(&hours("2024-07-02 00:00", 3), None);
    assert!(concat(&[&da, &utc], "time", &ConcatOptions::default()).is_err());
}

#[test]
fn display_shows_local_times() {
    let da = series(&[at("2024-07-01 02:00")], Some(ny()));
    let text = da.to_string();
    assert!(
        text.contains("datetime[America/New_York] 2024-06-30 22:00:00-04:00"),
        "{text}"
    );
    let time = da.coord("time").unwrap().to_string();
    assert!(time.contains("Time zone: America/New_York"), "{time}");
}

#[test]
fn netcdf_keeps_the_zone() {
    let dir = std::env::temp_dir().join(format!("annoray-tz-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("zoned.nc");
    let mut ds = Dataset::new();
    ds.add_array("v", series(&hours("2024-07-01 00:00", 3), Some(ny())))
        .unwrap();
    ds.to_netcdf(&path).unwrap();
    let back = open_dataset(&path).unwrap();
    let time = back.coords().get("time").unwrap();
    assert_eq!(time.time_zone(), Some(&ny()));
    assert!(!time.attrs().contains_key("time_zone"));
    assert_eq!(
        time.data::<Instant>()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        hours("2024-07-01 00:00", 3)
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
