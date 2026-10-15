//! Writing lazy data streams it: values are read (and CF-encoded) a slab or chunk at a time,
//! never as a whole.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::arithmetic_side_effects
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use annoray::io::{Format, OpenOptions, ToNetcdf, WriteOptions, open_dataset_with};
use annoray::ndarray::{ArrayD, IxDyn};
use annoray::{AttrValue, DType, Dataset, LazyArray, LazySource, Owned, TypedArray, Variable};

const ROWS: usize = 5000;
const COLS: usize = 1000;

fn value(i: usize, j: usize) -> f64 {
    ((i * 7 + j * 3) % 1000) as f64 / 10.0
}

/// Values computed on demand, remembering the largest read.
#[derive(Debug)]
struct Synthetic {
    shape: Vec<usize>,
    largest: AtomicUsize,
}

impl LazySource for Synthetic {
    fn dtype(&self) -> DType {
        DType::F64
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> annoray::Result<TypedArray<Owned>> {
        self.largest
            .fetch_max(count.iter().product(), Ordering::SeqCst);
        Ok(TypedArray::F64(ArrayD::from_shape_fn(IxDyn(count), |ix| {
            value(start[0] + ix[0], start[1] + ix[1])
        })))
    }
}

fn lazy_dataset(packed: bool) -> (Dataset, Arc<Synthetic>) {
    let src = Arc::new(Synthetic {
        shape: vec![ROWS, COLS],
        largest: AtomicUsize::new(0),
    });
    let mut v = Variable::from_lazy(["t", "x"], LazyArray::new(src.clone())).unwrap();
    v.encoding_mut().chunks = Some(vec![250, 500]);
    if packed {
        // CF packing, done slice by slice: 0.1 steps in 16-bit integers.
        v.encoding_mut().dtype = Some(DType::I16);
        v.encoding_mut().scale_factor = Some(AttrValue::F64(vec![0.1]));
        v.encoding_mut().fill_value = Some(AttrValue::I16(vec![-1]));
    }
    let mut ds = Dataset::new().with_var("v", v).unwrap();
    ds.encoding_mut().unlimited_dims = vec!["t".into()];
    (ds, src)
}

fn check(path: &std::path::Path) {
    let back = open_dataset_with(path, &OpenOptions::default())
        .unwrap()
        .dataset;
    let v = &back.data_vars()["v"];
    assert_eq!(v.shape(), [ROWS, COLS]);
    for (i, j) in [(0, 0), (1, 999), (2499, 3), (4999, 999), (3210, 456)] {
        let x = v.isel([("t", i), ("x", j)]).unwrap();
        let got = x.data::<f64>().unwrap()[[]];
        assert!((got - value(i, j)).abs() < 1e-9, "({i}, {j}): {got}");
    }
}

#[test]
fn netcdf4_writes_chunk_by_chunk() {
    let (ds, src) = lazy_dataset(true);
    let dir = std::env::temp_dir().join(format!("annoray-stream-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("s4.nc");
    ds.to_netcdf(&path).unwrap();
    // One chunk at a time (plus the one-element probe of the encoder).
    assert_eq!(src.largest.load(Ordering::SeqCst), 250 * 500);
    let raw = annoray::io::default_backend().read(&path).unwrap();
    assert_eq!(raw.vars["v"].data.dtype(), DType::I16, "packed on the way");
    check(&path);
}

#[test]
fn netcdf3_writes_records_in_batches() {
    let (ds, src) = lazy_dataset(false);
    let dir = std::env::temp_dir().join(format!("annoray-stream3-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("s3.nc");
    ds.to_netcdf_with(&path, &WriteOptions::format(Format::Netcdf3Offset64))
        .unwrap();
    // 40 MB of values, read in batches of records under 32 MB.
    let largest = src.largest.load(Ordering::SeqCst);
    assert!(largest < ROWS * COLS && largest > 0, "{largest}");
    check(&path);
}
