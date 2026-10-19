"""netCDF-4 write options: honoured exactly as netCDF-C/netCDF4-python honour them (the stored
bits are compared), or refused; never silently dropped."""

from __future__ import annotations

import numpy as np
import pytest

import annoray as ar

nc = pytest.importorskip("netCDF4")
pytestmark = pytest.mark.filterwarnings("ignore:endian-ness of dtype:UserWarning")

RNG = np.random.default_rng(7)
# Finite, non-zero values over a wide range (netCDF-C and annoray differ, deliberately, on
# NaN, ±∞, ±0 and on precisions beyond the type: see annoray.cf.quantize).
WIDE = (RNG.standard_normal(4000) * 10.0 ** RNG.integers(-30, 31, 4000)).reshape(40, 100)


def write_both(tmp_path, data, dtype, enc, kw):
    ours = tmp_path / "ours.nc"
    ar.Dataset({"v": ar.DataArray(data.astype(dtype), dims=("t", "x"))}).to_netcdf(
        ours, encoding={"v": enc}
    )
    ref = tmp_path / "ref.nc"
    with nc.Dataset(ref, "w") as d:
        d.createDimension("t", data.shape[0])
        d.createDimension("x", data.shape[1])
        d.createVariable("v", dtype, ("t", "x"), **kw)[:] = data.astype(dtype)
    return ours, ref


def bits(path):
    with nc.Dataset(path) as d:
        d.set_auto_mask(False)
        v = np.asarray(d["v"][:])
    return v.view(f"u{v.dtype.itemsize}")


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize(
    ("mode", "levels"),
    [("BitGroom", range(1, 7)), ("GranularBitRound", range(1, 7)), ("BitRound", range(1, 23, 4))],
)
def test_quantization_matches_netcdf_c(tmp_path, dtype, mode, levels):
    for n in levels:
        enc = {"significant_digits": n, "quantize_mode": mode, "zlib": True}
        ours, ref = write_both(
            tmp_path,
            WIDE,
            dtype,
            enc,
            {"significant_digits": n, "quantize_mode": mode, "zlib": True},
        )
        np.testing.assert_array_equal(bits(ours), bits(ref), err_msg=f"{mode} {n}")
        with nc.Dataset(ours) as d:
            assert d["v"].quantization() == (n, mode)


def test_least_significant_digit_matches_netcdf4_python(tmp_path):
    data = RNG.standard_normal((10, 100)) * 50
    for lsd in (0, 1, 2, 4):
        ours, ref = write_both(
            tmp_path, data, "f4", {"least_significant_digit": lsd}, {"least_significant_digit": lsd}
        )
        np.testing.assert_array_equal(bits(ours), bits(ref))


def test_checksums_byte_order_and_storage(tmp_path):
    data = RNG.standard_normal((10, 100))
    for enc, kw in [
        ({"fletcher32": True, "chunksizes": (5, 50)}, {"fletcher32": True, "chunksizes": (5, 50)}),
        ({"fletcher32": True}, {"fletcher32": True}),
        ({"endian": "big"}, {"endian": "big"}),
        ({"endian": "big", "zlib": True, "shuffle": True}, {"endian": "big", "zlib": True}),
        ({"contiguous": True}, {"contiguous": True}),
    ]:
        ours, ref = write_both(tmp_path, data, "f8", enc, kw)
        np.testing.assert_array_equal(bits(ours), bits(ref))
        with nc.Dataset(ours) as d:
            v = d["v"]
            assert v.filters()["fletcher32"] == bool(enc.get("fletcher32")), enc
            assert v.endian() == enc.get("endian", "little"), enc
            if enc.get("contiguous"):
                assert v.chunking() == "contiguous"
        back = ar.open_dataset(ours)["v"].encoding
        for k in ("fletcher32", "endian", "contiguous"):
            if k in enc:
                assert back.get(k) == enc[k], (k, back)


def test_no_fill_value(tmp_path):
    p = tmp_path / "f.nc"
    ar.Dataset({"v": ar.DataArray([1.0, np.nan], dims="x")}).to_netcdf(
        p, encoding={"v": {"_FillValue": None}}
    )
    with nc.Dataset(p) as d:
        assert "_FillValue" not in d["v"].ncattrs()


def test_options_survive_a_round_trip(tmp_path):
    p, q = tmp_path / "a.nc", tmp_path / "b.nc"
    enc = {
        "significant_digits": 3,
        "quantize_mode": "GranularBitRound",
        "zlib": True,
        "fletcher32": True,
        "endian": "big",
    }
    ar.Dataset({"v": ar.DataArray(WIDE, dims=("t", "x"))}).to_netcdf(p, encoding={"v": enc})
    ds = ar.open_dataset(p)
    assert "_QuantizeGranularBitRoundNumberOfSignificantDigits" not in ds["v"].attrs
    ds.to_netcdf(q)  # the variable's own encoding
    with nc.Dataset(q) as d:
        v = d["v"]
        assert v.quantization() == (3, "GranularBitRound")
        assert v.filters()["fletcher32"] and v.endian() == "big"
    np.testing.assert_array_equal(bits(p), bits(q))


def test_lazy_writes_quantize_like_eager_ones(tmp_path):
    src = tmp_path / "src.nc"
    ar.Dataset({"v": ar.DataArray(WIDE, dims=("t", "x"))}).to_netcdf(
        src, encoding={"v": {"chunksizes": (7, 30)}}
    )
    enc = {"v": {"significant_digits": 4, "zlib": True, "chunksizes": (7, 30)}}
    lazy, eager = tmp_path / "lazy.nc", tmp_path / "eager.nc"
    ds = ar.open_dataset(src)
    assert ds["v"].is_lazy
    ds.to_netcdf(lazy, encoding=enc)
    ds.load().to_netcdf(eager, encoding=enc)
    np.testing.assert_array_equal(bits(lazy), bits(eager))


@pytest.mark.parametrize(
    ("enc", "fmt", "match"),
    [
        ({"zlibb": True}, "NETCDF4", "unexpected encoding"),
        ({"nonsense": 1}, "NETCDF4", "unexpected encoding"),
        ({"contiguous": True, "zlib": True}, "NETCDF4", "contiguous"),
        ({"endian": "middle"}, "NETCDF4", "endian"),
        ({"quantize_mode": "BitRound"}, "NETCDF4", "significant_digits"),
        ({"significant_digits": 99}, "NETCDF4", "out of range"),
        ({"compression": "szip"}, "NETCDF4", "szip"),
        ({"zlib": True}, "NETCDF3_64BIT", "netCDF-4"),
        ({"endian": "little"}, "NETCDF3_CLASSIC", "big-endian"),
    ],
)
def test_options_that_cannot_be_honoured_are_refused(tmp_path, enc, fmt, match):
    ds = ar.Dataset({"v": ar.DataArray(np.arange(4.0), dims="x")})
    with pytest.raises(ValueError, match=match):
        ds.to_netcdf(tmp_path / "x.nc", format=fmt, encoding={"v": enc})


def test_zarr_refuses_netcdf_options(tmp_path):
    ds = ar.Dataset({"v": ar.DataArray(np.arange(4.0), dims="x")})
    with pytest.raises(ValueError, match="unexpected encoding"):
        ds.to_zarr(tmp_path / "x.zarr", encoding={"v": {"zlib": True}})
    ds.to_zarr(tmp_path / "y.zarr", encoding={"v": {"chunks": (2,)}})


def test_xarray_objects_are_checked_too(tmp_path):
    xr = pytest.importorskip("xarray")
    x = xr.Dataset({"v": ("x", np.arange(4.0))})
    with pytest.raises(ValueError, match="unexpected encoding"):
        ar.to_netcdf(x, tmp_path / "x.nc", encoding={"v": {"zlibb": True}})
    x["v"].encoding.update({"source": "somewhere.nc", "original_shape": (4,)})  # bookkeeping
    ar.to_netcdf(x, tmp_path / "y.nc", encoding={"v": {"fletcher32": True}})
    with nc.Dataset(tmp_path / "y.nc") as d:
        assert d["v"].filters()["fletcher32"]
