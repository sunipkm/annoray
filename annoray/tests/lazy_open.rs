//! Lazy opening: data variables are read on access, only the selected part, and agree with an
//! eager read.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use std::path::PathBuf;

use annoray::io::{OpenOptions, open_dataset, open_dataset_with};
use annoray::{Indexer, PySlice, Repr, VariableBase};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name)
}

/// Values compared through `Debug`, so NaNs compare equal (strides of empty arrays ignored).
fn values<S: Repr>(v: &VariableBase<S>) -> String {
    let s = format!("{:?}", v.typed().unwrap());
    s.split(", strides=").next().unwrap_or_default().to_owned()
}

fn same<S: Repr, R: Repr>(a: &VariableBase<S>, b: &VariableBase<R>, what: &str) {
    assert_eq!(a.dims(), b.dims(), "{what}: dims");
    assert_eq!(a.dtype(), b.dtype(), "{what}: dtype");
    assert_eq!(a.unit(), b.unit(), "{what}: unit");
    assert_eq!(a.attrs(), b.attrs(), "{what}: attrs");
    assert_eq!(values(a), values(b), "{what}: values");
}

#[test]
fn lazy_matches_eager() {
    for f in [
        "nc/xr_nc3.nc",
        "nc/xr_nc4.nc",
        "nc/raw_nc4.nc",
        "h5/latest.h5",
        "h5/earliest_be.h5",
    ] {
        let lazy = open_dataset(fixture(f)).unwrap();
        let eager = open_dataset_with(fixture(f), &OpenOptions::default().lazy(false))
            .unwrap()
            .dataset;
        assert_eq!(
            lazy.data_vars().keys().collect::<Vec<_>>(),
            eager.data_vars().keys().collect::<Vec<_>>(),
            "{f}"
        );
        for (name, v) in lazy.data_vars() {
            let e = &eager.data_vars()[name];
            assert!(!e.is_lazy());
            assert_eq!(
                format!("{:?}", v.encoding()),
                format!("{:?}", e.encoding()),
                "{f}:{name}"
            );

            // Selections compose without reading and give the same values.
            for (axis, dim) in v.dims().iter().enumerate() {
                let n = v.shape()[axis];
                let picks: [Indexer; 3] = [
                    PySlice::new(Some(1), None, Some(-2)).into(),
                    PySlice::new(None, None, Some(2)).into(),
                    Indexer::Int(-1),
                ];
                for p in picks {
                    if n == 0 {
                        continue;
                    }
                    let a = v.isel([(dim.as_str(), p)]).unwrap();
                    let b = e.isel([(dim.as_str(), p)]).unwrap();
                    assert!(a.is_lazy() || !v.is_lazy());
                    same(&a, &b, &format!("{f}:{name}[{dim}]"));
                }
            }
            same(v, e, &format!("{f}:{name}"));
        }
        for (name, c) in lazy.coords() {
            let e = &eager.coords()[name];
            same(c, e, &format!("{f}:{name}"));
            if c.dims().len() == 1 && c.dims()[0] == name.as_str() {
                assert!(!c.is_lazy(), "{f}:{name}: index coordinates load eagerly");
            }
        }
        let mut loaded = lazy.clone();
        loaded.load().unwrap();
        assert!(loaded.data_vars().values().all(|v| !v.is_lazy()));
    }
}

#[test]
fn opening_is_lazy() {
    let ds = open_dataset(fixture("nc/xr_nc4.nc")).unwrap();
    assert!(ds.data_vars().values().any(VariableBase::is_lazy));
    // Display never reads.
    let shown = format!("{ds}");
    assert!(shown.contains("..."), "{shown}");
    assert!(ds.data_vars().values().any(VariableBase::is_lazy));
}
