"""Times with a zone: exact conversion of whenever, datetime and pandas inputs (no numpy
warnings), the zone of a variable, wall-clock components and bins, and round trips."""

from __future__ import annotations

import datetime as dt
import warnings
import zoneinfo

import numpy as np
import pytest

import annoray as ar

NY = "America/New_York"


@pytest.fixture(autouse=True)
def _no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def utc(*s: str) -> np.ndarray:
    return np.array(s, dtype="datetime64[ns]")


def test_aware_datetimes_convert_exactly_without_warnings():
    t = dt.datetime(2024, 5, 11, 3, 0, 0, 123456, tzinfo=zoneinfo.ZoneInfo(NY))
    da = ar.DataArray([t], dims="t")
    np.testing.assert_array_equal(da.values, utc("2024-05-11T07:00:00.123456"))
    assert da.dt.tz == NY


def test_fixed_offsets_and_utc():
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    da = ar.DataArray([dt.datetime(2024, 5, 11, 3, tzinfo=ist)], dims="t")
    np.testing.assert_array_equal(da.values, utc("2024-05-10T21:30"))
    assert da.dt.tz == "+05:30"
    da = ar.DataArray([dt.datetime(2024, 5, 11, tzinfo=dt.UTC)], dims="t")
    assert da.dt.tz == "UTC"


def test_naive_times_have_no_zone():
    da = ar.DataArray(utc("2024-05-11T03:00"), dims="t")
    assert da.dt.tz is None
    naive = dt.datetime.fromisoformat("2024-05-11")  # no zone
    assert ar.DataArray([naive], dims="t").dt.tz is None


def test_values_are_read_only_views():
    da = ar.DataArray(utc("2024-05-11", "2024-05-12"), dims="t")
    a, b = da.values, da.values
    assert np.shares_memory(a, b)
    assert not a.flags.writeable
    assert a.dtype == np.dtype("datetime64[ns]")


def test_whenever_inputs():
    w = pytest.importorskip("whenever")
    zoned = [w.ZonedDateTime(2024, 5, 11, h, tz=NY) for h in (3, 4)]
    da = ar.DataArray(zoned, dims="t")
    np.testing.assert_array_equal(da.values, utc("2024-05-11T07:00", "2024-05-11T08:00"))
    assert da.dt.tz == NY
    np.testing.assert_array_equal(da.dt.hour.values, [3, 4])

    instant = ar.DataArray([w.Instant.from_utc(2024, 5, 11, 7)], dims="t")
    np.testing.assert_array_equal(instant.values, utc("2024-05-11T07:00"))
    assert instant.dt.tz is None

    offset = ar.DataArray(
        [w.OffsetDateTime(2024, 5, 11, 3, offset=w.TimeDelta(hours=-4))], dims="t"
    )
    np.testing.assert_array_equal(offset.values, utc("2024-05-11T07:00"))
    assert offset.dt.tz == "-04:00"

    with pytest.raises(TypeError, match="assume_utc"):
        ar.DataArray([w.PlainDateTime(2024, 5, 11, 3)], dims="t")


def test_times_with_and_without_a_zone_do_not_mix():
    w = pytest.importorskip("whenever")
    with pytest.raises(TypeError, match="no time zone"):
        naive = dt.datetime.fromisoformat("2024-05-11")  # no zone
        ar.DataArray([w.ZonedDateTime(2024, 5, 11, tz=NY), naive], dims="t")


def test_different_zones_become_utc_instants():
    w = pytest.importorskip("whenever")
    times = [w.ZonedDateTime(2024, 5, 11, tz=NY), w.ZonedDateTime(2024, 5, 11, tz="Asia/Tokyo")]
    da = ar.DataArray(times, dims="t")
    assert da.dt.tz is None
    np.testing.assert_array_equal(da.values, utc("2024-05-11T04:00", "2024-05-10T15:00"))


def test_pandas_aware_index():
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2024-03-09", periods=3, freq="D", tz=NY)
    da = ar.DataArray(np.arange(3.0), dims="time", coords={"time": idx})
    assert da.time.dt.tz == NY
    np.testing.assert_array_equal(
        da.time.values, utc("2024-03-09T05:00", "2024-03-10T05:00", "2024-03-11T04:00")
    )
    np.testing.assert_array_equal(da.time.dt.hour.values, [0, 0, 0])
    one = ar.DataArray([pd.Timestamp("2024-05-11T03:00:00.000000001", tz=NY)], dims="t")
    np.testing.assert_array_equal(one.values, utc("2024-05-11T07:00:00.000000001"))


def test_date_range_with_a_zone_keeps_it():
    pytest.importorskip("pandas")
    times = ar.date_range("2024-03-09", periods=3, freq="D", tz=NY)
    assert ar.DataArray(times, dims="time").dt.tz == NY


def test_tz_convert_keeps_instants():
    da = ar.DataArray(utc("2024-07-01T02:00"), dims="t")
    ny = da.dt.tz_convert(NY)
    assert ny.dt.tz == NY
    np.testing.assert_array_equal(ny.values, da.values)
    assert ny.dt.hour.values[0] == 22
    assert ny.dt.day.values[0] == 30
    assert da.dt.tz_convert(zoneinfo.ZoneInfo("Asia/Kolkata")).dt.tz == "Asia/Kolkata"
    assert ny.dt.tz_convert(None).dt.tz is None


def test_tz_localize_reads_wall_clock_times():
    da = ar.DataArray(utc("2024-07-01T08:00", "2024-11-03T01:30"), dims="t")
    ny = da.dt.tz_localize(NY, disambiguate="earlier")
    np.testing.assert_array_equal(ny.values, utc("2024-07-01T12:00", "2024-11-03T05:30"))
    later = da.dt.tz_localize(NY, disambiguate="later")
    np.testing.assert_array_equal(later.values[1:], utc("2024-11-03T06:30"))
    with pytest.raises(ValueError, match="ambiguous"):
        da.dt.tz_localize(NY, disambiguate="raise")
    with pytest.raises(ValueError, match="unknown time zone"):
        da.dt.tz_localize("Mars/Olympus_Mons")


def test_zone_only_for_date_times():
    with pytest.raises(TypeError):
        ar.DataArray([1.0], dims="t").dt.tz_convert(NY)


def test_resample_bins_on_local_days():
    times = np.datetime64("2024-07-01T04:00", "ns") + np.arange(48) * np.timedelta64(1, "h")
    da = ar.DataArray(np.arange(48.0), dims="time", coords={"time": times})
    local = da.assign_coords(time=da.time.dt.tz_convert(NY))
    daily = local.resample(time="D").mean()
    np.testing.assert_array_equal(daily.values, [11.5, 35.5])
    assert daily.time.dt.tz == NY
    np.testing.assert_array_equal(daily.time.values, utc("2024-07-01T04:00", "2024-07-02T04:00"))
    assert da.resample(time="D").mean().sizes["time"] == 3


def test_selection_by_wall_clock_dates_and_zoned_labels():
    times = np.datetime64("2024-07-01T00:00", "ns") + np.arange(48) * np.timedelta64(1, "h")
    da = ar.DataArray(np.arange(48.0), dims="time", coords={"time": times})
    local = da.assign_coords(time=da.time.dt.tz_convert(NY))
    day = local.sel(time="2024-07-01")
    np.testing.assert_array_equal(day.values, np.arange(4.0, 28.0))
    noon = dt.datetime(2024, 7, 1, 12, tzinfo=zoneinfo.ZoneInfo(NY))
    assert local.sel(time=noon).values == 16.0


def test_to_whenever():
    w = pytest.importorskip("whenever")
    da = ar.DataArray(utc("2024-05-11T07:00", "NaT"), dims="t")
    zoned = da.dt.tz_convert(NY).dt.to_whenever()
    assert zoned[0] == w.ZonedDateTime(2024, 5, 11, 3, tz=NY)
    assert zoned[1] is None
    assert da.dt.to_whenever()[0] == w.Instant.from_utc(2024, 5, 11, 7)
    fixed = da.dt.tz_convert("+05:30").dt.to_whenever()[0]
    assert fixed == w.OffsetDateTime(2024, 5, 11, 12, 30, offset=w.TimeDelta(hours=5, minutes=30))


def test_repr_shows_local_times():
    da = ar.DataArray(utc("2024-07-01T02:00"), dims="t").dt.tz_convert(NY)
    text = repr(da)
    assert "2024-06-30 22:00:00-04:00" in text
    assert "Time zone: America/New_York" in text


def test_round_trips(tmp_path):
    da = ar.DataArray(utc("2024-07-01T02:00", "2024-07-01T03:00"), dims="time")
    ds = ar.Dataset({"v": ("time", [1.0, 2.0])}, coords={"time": da.dt.tz_convert(NY)})
    ds.to_netcdf(tmp_path / "z.nc")
    back = ar.open_dataset(tmp_path / "z.nc")
    assert back.time.dt.tz == NY
    assert "time_zone" not in back.time.attrs
    np.testing.assert_array_equal(back.time.values, da.values)
    # Given as an attribute, as files hold it.
    attr = ar.DataArray(da.values, dims="time", attrs={"time_zone": NY})
    assert attr.dt.tz == NY
    assert "time_zone" not in attr.attrs
    pytest.importorskip("xarray")
    x = ds.to_xarray()
    assert x.time.attrs["time_zone"] == NY
    assert ar.from_xarray(x).time.dt.tz == NY


def test_concat_refuses_different_zones():
    da = ar.DataArray(utc("2024-07-01"), dims="time")
    with pytest.raises(ValueError, match="time zone"):
        ar.concat([da.dt.tz_convert(NY), da], dim="time")
    both = ar.concat([da.dt.tz_convert(NY), da.dt.tz_convert(NY)], dim="time")
    assert both.dt.tz == NY
