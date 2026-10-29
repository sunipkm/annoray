//! A netCDF-3, netCDF-4 or HDF5 file in memory, opened under `ReadLimits::untrusted()` and
//! read: lazily (a corner of every variable, then the whole), and eagerly. Every input must
//! give a dataset or an error; panics, aborts and allocations past the limits are bugs.
#![no_main]

use annoray::io::{OpenOptions, ReadLimits, open_dataset_from_bytes, open_datatree_from_bytes};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let limits = ReadLimits::untrusted()
        .max_read_bytes(16 << 20)
        .max_total_bytes(64 << 20);
    let lazy = OpenOptions::default().limits(limits);
    if let Ok(opened) = open_dataset_from_bytes(data.to_vec(), &lazy) {
        let ds = &opened.dataset;
        for v in ds.coords().values().chain(ds.data_vars().values()) {
            let corner: Vec<_> = v.dims().iter().map(|d| (d.clone(), 0..1)).collect();
            if let Ok(view) = v.isel(corner) {
                let _ = view.to_owned().load();
            }
            let _ = v.clone().load();
        }
    }
    let _ = open_dataset_from_bytes(data.to_vec(), &lazy.clone().lazy(false));
    let _ = open_datatree_from_bytes(data.to_vec(), &lazy);
});
