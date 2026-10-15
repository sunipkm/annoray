//! netCDF-4 files (written by netCDF-C via Python) read through the plain HDF5 reader must
//! give the same variable values and attributes as netCDF-C itself.
#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use std::path::PathBuf;

use annoray_io::{Backend, RawDataset};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/nc")
        .join(name)
}

fn same_values(ours: &RawDataset, c: &RawDataset, ctx: &str) {
    for (name, v) in &c.vars {
        let Some(w) = ours.vars.get(name) else {
            panic!("{ctx}: {name} missing; skipped = {:?}", ours.skipped)
        };
        let a = format!("{:?}", v.data);
        let b = format!("{:?}", w.data);
        // NC_CHAR arrays are plain bytes to HDF5.
        if !v.is_char {
            assert_eq!(a, b, "{ctx}: {name}");
        }
        for (k, attr) in &v.attrs {
            if k.as_str() == "_FillValue" {
                continue; // netCDF-C reports the fill value property; HDF5 stores it as a property too
            }
            assert_eq!(
                Some(format!("{attr:?}")),
                w.attrs.get(k).map(|x| format!("{x:?}")),
                "{ctx}: {name}@{k}"
            );
        }
    }
    for (k, attr) in &c.attrs {
        assert_eq!(
            Some(format!("{attr:?}")),
            ours.attrs.get(k).map(|x| format!("{x:?}")),
            "{ctx}: global @{k}"
        );
    }
    for (g, child) in &c.groups {
        same_values(&ours.groups[g], child, &format!("{ctx}/{g}"));
    }
}

#[test]
fn netcdf4_files_match_netcdf_c() {
    for name in ["xr_nc4.nc", "raw_nc4.nc"] {
        let c = annoray_io_c::NetcdfC.read(&fixture(name)).unwrap();
        let ours = annoray_hdf5::Hdf5.read(&fixture(name)).unwrap();
        same_values(&ours, &c, name);
    }
}
