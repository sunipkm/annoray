//! Raw (pre-CF) round trips through netCDF-C.
#![cfg(feature = "netcdf-c")]
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use annoray::io::netcdf_c::NetcdfC;
use annoray::io::{
    Backend, Format, IoError, RawDataset, RawDim, RawVariable, Storage, WriteOptions,
};
use annoray::{AttrValue, DType, Name, TypedArray};
use ndarray::{ArrayD, IxDyn, array};

fn var(dims: &[&str], data: TypedArray<annoray::Owned>) -> RawVariable<'static> {
    RawVariable {
        dims: dims.iter().map(|d| Name::from(*d)).collect(),
        data: data.into(),
        is_char: false,
        attrs: Default::default(),
        storage: Storage::default(),
    }
}

fn sample(netcdf4: bool) -> RawDataset<'static> {
    let mut ds = RawDataset::default();
    ds.dims.insert(
        "time".into(),
        RawDim {
            len: 3,
            unlimited: true,
        },
    );
    ds.dims.insert(
        "x".into(),
        RawDim {
            len: 2,
            unlimited: false,
        },
    );
    ds.dims.insert(
        "nchar".into(),
        RawDim {
            len: 4,
            unlimited: false,
        },
    );
    ds.attrs.insert("title".into(), AttrValue::from("raw test"));
    ds.attrs
        .insert("version".into(), AttrValue::from(vec![1i32, 2]));
    let mut t = var(
        &["time", "x"],
        TypedArray::F64(
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ),
    );
    t.attrs
        .insert("_FillValue".into(), AttrValue::from(-999.0f64));
    t.attrs.insert("units".into(), AttrValue::from("K"));
    ds.vars.insert("t".into(), t);
    ds.vars.insert(
        "i16".into(),
        var(&["x"], TypedArray::I16(array![-3i16, 7].into_dyn())),
    );
    ds.vars.insert(
        "i8".into(),
        var(&["x"], TypedArray::I8(array![-1i8, 1].into_dyn())),
    );
    ds.vars.insert(
        "scalar".into(),
        var(&[], TypedArray::I32(ArrayD::from_elem(IxDyn(&[]), 42))),
    );
    let mut c = var(
        &["x", "nchar"],
        TypedArray::U8(ArrayD::from_shape_vec(IxDyn(&[2, 4]), b"ab\0\0wxyz".to_vec()).unwrap()),
    );
    c.is_char = true;
    ds.vars.insert("chars".into(), c);
    if netcdf4 {
        ds.vars.insert(
            "u64".into(),
            var(&["x"], TypedArray::U64(array![u64::MAX, 0].into_dyn())),
        );
        ds.vars.insert(
            "s".into(),
            var(
                &["x"],
                TypedArray::Str(array!["héllo".to_owned(), String::new()].into_dyn()),
            ),
        );
        let mut z = var(
            &["time", "x"],
            TypedArray::F32(ArrayD::from_elem(IxDyn(&[3, 2]), 0.5f32)),
        );
        z.storage.chunks = Some(vec![1, 2]);
        z.storage.deflate = Some(4);
        z.storage.shuffle = true;
        ds.vars.insert("z".into(), z);
        let mut g = RawDataset::default();
        g.dims.insert(
            "y".into(),
            RawDim {
                len: 1,
                unlimited: false,
            },
        );
        g.vars.insert(
            "gy".into(),
            var(
                &["x", "y"],
                TypedArray::F32(array![[1.5f32], [2.5]].into_dyn()),
            ),
        );
        g.attrs.insert(
            "group_attr".into(),
            AttrValue::Strs(vec!["a".into(), "b".into()]),
        );
        ds.groups.insert("sub".into(), g);
    }
    ds
}

fn check(a: &RawDataset, b: &RawDataset) {
    assert_eq!(a.dims, b.dims);
    assert_eq!(a.attrs, b.attrs);
    assert_eq!(
        a.vars.keys().collect::<Vec<_>>(),
        b.vars.keys().collect::<Vec<_>>()
    );
    for (k, v) in &a.vars {
        let w = &b.vars[k];
        assert_eq!(v.dims, w.dims, "{k}");
        assert_eq!(v.attrs, w.attrs, "{k}");
        assert_eq!(v.is_char, w.is_char, "{k}");
        assert!(
            annoray::variables_equal(
                &annoray::Variable::from_typed(
                    v.dims.clone(),
                    v.data.values().unwrap().into_owned()
                )
                .unwrap(),
                &annoray::Variable::from_typed(
                    w.dims.clone(),
                    w.data.values().unwrap().into_owned()
                )
                .unwrap()
            ),
            "{k}: {:?} vs {:?}",
            v.data,
            w.data
        );
    }
    assert_eq!(a.groups.len(), b.groups.len());
    for (k, g) in &a.groups {
        check(g, &b.groups[k]);
    }
}

#[test]
fn netcdf4_round_trip() {
    let dir = std::env::temp_dir().join(format!("annoray-io-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("nc4.nc");
    let ds = sample(true);
    NetcdfC
        .write(&ds, &path, &WriteOptions::format(Format::Netcdf4))
        .unwrap();
    let back = NetcdfC.read(&path).unwrap();
    check(&ds, &back);
    assert_eq!(back.vars["z"].storage.chunks, Some(vec![1, 2]));
    assert_eq!(back.vars["s"].data.dtype(), DType::Str);
}

#[test]
fn netcdf3_round_trip_and_limits() {
    let dir = std::env::temp_dir().join(format!("annoray-io-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (i, f) in [
        Format::Netcdf3Classic,
        Format::Netcdf3Offset64,
        Format::Netcdf4Classic,
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.join(format!("nc3-{i}.nc"));
        let ds = sample(false);
        NetcdfC.write(&ds, &path, &WriteOptions::format(f)).unwrap();
        check(&ds, &NetcdfC.read(&path).unwrap());
    }
    // Classic formats reject netCDF-4-only features with an error.
    let path = dir.join("bad.nc");
    let err = NetcdfC
        .write(
            &sample(true),
            &path,
            &WriteOptions::format(Format::Netcdf3Classic),
        )
        .unwrap_err();
    assert!(matches!(err, IoError::Format { .. }), "{err}");
    assert!(matches!(
        NetcdfC.read(&dir.join("missing.nc")),
        Err(IoError::Backend { .. })
    ));
}
