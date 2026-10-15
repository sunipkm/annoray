//! Computing on lazy (on-disk) data without loading it: elementwise operations stay lazy and
//! read only the boxes asked for; reductions stream slab by slab within a budget. Results
//! must equal the in-memory computation bit for bit.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{ArrayD, Dimension, IxDyn};
use xr_core::{
    DType, DataArray, LazyArray, LazySource, Name, Owned, ReduceOptions, Reduction, TypedArray,
    Variable,
};

const BUDGET: usize = 64 << 10;

/// Values computed on demand (with NaNs), remembering the largest read (in elements).
#[derive(Debug)]
struct Synthetic {
    shape: Vec<usize>,
    f32: bool,
    largest: AtomicUsize,
    /// Chunk length along the first axis (reported through `chunking`).
    chunk0: Option<usize>,
    /// (start, len) along the first axis of every read.
    reads: std::sync::Mutex<Vec<(usize, usize)>>,
}

fn value(ix: &[usize]) -> f64 {
    let k = ix.iter().fold(0usize, |a, i| a * 1000 + i);
    if k % 11 == 0 {
        f64::NAN
    } else {
        ((k * 7919) % 1000) as f64 / 7.0 - 50.0
    }
}

impl LazySource for Synthetic {
    fn dtype(&self) -> DType {
        if self.f32 { DType::F32 } else { DType::F64 }
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn chunking(&self, axis: usize) -> Option<(usize, usize)> {
        (axis == 0).then_some(self.chunk0?).map(|c| (c, 0))
    }
    fn read_slice(&self, start: &[usize], count: &[usize]) -> xr_core::Result<TypedArray<Owned>> {
        self.largest
            .fetch_max(count.iter().product(), Ordering::SeqCst);
        if let Ok(mut r) = self.reads.lock() {
            r.push((start[0], count[0]));
        }
        let a = ArrayD::from_shape_fn(IxDyn(count), |ix| {
            let g: Vec<usize> = ix.slice().iter().zip(start).map(|(i, s)| i + s).collect();
            value(&g)
        });
        Ok(if self.f32 {
            TypedArray::F32(a.mapv(|v| v as f32))
        } else {
            TypedArray::F64(a)
        })
    }
}

fn lazy(dims: &[&str], shape: &[usize], f32: bool) -> (DataArray, Arc<Synthetic>) {
    lazy_chunked(dims, shape, f32, None)
}

fn lazy_chunked(
    dims: &[&str],
    shape: &[usize],
    f32: bool,
    chunk0: Option<usize>,
) -> (DataArray, Arc<Synthetic>) {
    let src = Arc::new(Synthetic {
        shape: shape.to_vec(),
        f32,
        largest: AtomicUsize::new(0),
        chunk0,
        reads: std::sync::Mutex::new(Vec::new()),
    });
    let v = Variable::from_lazy(dims.to_vec(), LazyArray::new(src.clone()))
        .unwrap()
        .with_unit("m".parse().unwrap());
    (DataArray::new(v), src)
}

/// The same data, in memory (read through a fresh source: loading a clone would fill the
/// lazy cache it shares with `a`).
fn loaded(a: &DataArray) -> DataArray {
    let src = a.variable().as_lazy_source().unwrap();
    let data = src.read_slice(&vec![0; a.ndim()], a.shape()).unwrap();
    let mut v = Variable::from_typed(a.dims().to_vec(), data).unwrap();
    v.set_unit(a.unit().cloned());
    DataArray::new(v)
}

fn bits(t: &TypedArray<xr_core::View<'_>>) -> Vec<u64> {
    match t {
        TypedArray::F64(a) => a.iter().map(|x| x.to_bits()).collect(),
        TypedArray::F32(a) => a.iter().map(|x| u64::from(x.to_bits())).collect(),
        other => panic!("{other:?}"),
    }
}

fn same(a: &DataArray, b: &DataArray, what: &str) {
    assert_eq!(a.dims(), b.dims(), "{what}");
    assert_eq!(a.unit(), b.unit(), "{what}");
    let (x, y) = (a.variable().typed().unwrap(), b.variable().typed().unwrap());
    assert_eq!(x.dtype(), y.dtype(), "{what}");
    assert_eq!(bits(&x), bits(&y), "{what}");
}

/// Equal to rounding (streamed variances merge per-slab moments; in memory they are
/// two-pass): NaN where the other is NaN, otherwise within `rtol`.
fn close(a: &DataArray, b: &DataArray, what: &str) {
    assert_eq!(a.dims(), b.dims(), "{what}");
    assert_eq!(a.unit(), b.unit(), "{what}");
    let f = |d: &DataArray| -> (Vec<f64>, f64) {
        match d.variable().typed().unwrap() {
            TypedArray::F64(x) => (x.iter().copied().collect(), 1e-12),
            TypedArray::F32(x) => (x.iter().map(|v| f64::from(*v)).collect(), 1e-5),
            other => panic!("{other:?}"),
        }
    };
    let ((x, rtol), (y, _)) = (f(a), f(b));
    assert_eq!(x.len(), y.len(), "{what}");
    for (p, q) in x.iter().zip(&y) {
        assert!(
            (p.is_nan() && q.is_nan()) || (p - q).abs() <= rtol * q.abs().max(1.0),
            "{what}: {p} vs {q}"
        );
    }
}

/// Bit-identical, except variances (which agree to rounding).
fn same_or_close(op: Reduction, a: &DataArray, b: &DataArray, what: &str) {
    if matches!(op, Reduction::Var | Reduction::Std) {
        close(a, b, what);
    } else {
        same(a, b, what);
    }
}

#[test]
fn arithmetic_stays_lazy_and_reads_boxes() {
    let (a, sa) = lazy(&["t", "y", "x"], &[40, 30, 20], false);
    let (b, sb) = lazy(&["y"], &[30], false);
    let two = DataArray::new(
        Variable::new(Vec::<Name>::new(), ArrayD::from_elem(IxDyn(&[]), 2.0)).unwrap(),
    );
    let expr = ((&a * &two).unwrap().try_add(&b).unwrap())
        .try_sub(&a)
        .unwrap();
    assert!(expr.variable().is_lazy(), "nothing computed yet");
    assert_eq!(sa.largest.load(Ordering::SeqCst), 0, "nothing read yet");
    // A small selection reads only its box.
    let part = expr
        .isel([
            ("t", xr_core::Indexer::Int(3)),
            ("x", xr_core::Indexer::Int(7)),
        ])
        .unwrap()
        .to_owned();
    let got = loaded(&DataArray::new(part.variable().to_owned()));
    assert_eq!(
        sa.largest.load(Ordering::SeqCst),
        30,
        "one (t, x) column of a"
    );
    assert_eq!(sb.largest.load(Ordering::SeqCst), 30);
    // Same values as in memory.
    let (am, bm) = (loaded(&a), loaded(&b));
    let want = ((&am * &two).unwrap().try_add(&bm).unwrap())
        .try_sub(&am)
        .unwrap();
    same(&loaded(&expr), &want, "a * 2 + b - a");
    let wpart = want
        .isel([
            ("t", xr_core::Indexer::Int(3)),
            ("x", xr_core::Indexer::Int(7)),
        ])
        .unwrap()
        .to_owned();
    same(
        &got,
        &DataArray::new(wpart.variable().to_owned()),
        "selection",
    );
    // Unit conversion too.
    let cm = a.to_unit(&"cm".parse().unwrap()).unwrap();
    assert!(cm.variable().is_lazy());
    same(
        &loaded(&cm),
        &am.to_unit(&"cm".parse().unwrap()).unwrap(),
        "to cm",
    );
}

#[test]
fn reductions_stream_within_budget() {
    xr_core::set_stream_bytes(BUDGET);
    let ops = [
        Reduction::Sum,
        Reduction::Mean,
        Reduction::Var,
        Reduction::Std,
        Reduction::Min,
        Reduction::Max,
    ];
    for f32 in [false, true] {
        // 60 × 40 × 50 values: well over the budget.
        let (a, src) = lazy(&["t", "y", "x"], &[60, 40, 50], f32);
        let am = loaded(&a);
        let esize = if f32 { 4 } else { 8 };
        let row = 40 * 50;
        for dims in [
            vec!["t"],
            vec!["y"],
            vec!["t", "x"],
            vec!["t", "y", "x"],
            vec!["y", "x"],
        ] {
            for op in ops {
                for skipna in [true, false] {
                    let opts = ReduceOptions::default().skipna(skipna).ddof(1);
                    let names: Vec<Name> = dims.iter().map(|d| Name::from(*d)).collect();
                    src.largest.store(0, Ordering::SeqCst);
                    let got = a.reduce(op, names.clone(), &opts).unwrap();
                    let largest = src.largest.load(Ordering::SeqCst);
                    let want = am.reduce(op, names, &opts).unwrap();
                    let what = format!("f32={f32} {op:?} over {dims:?} skipna={skipna}");
                    same_or_close(op, &got, &want, &what);
                    let slab = (BUDGET / (row * esize)).max(1) * row;
                    assert!(
                        largest <= slab,
                        "{what}: read {largest} values at once (budget {slab})"
                    );
                    assert!(largest > 0, "{what}: streamed");
                }
            }
        }
    }
}

#[test]
fn streamed_slabs_follow_chunks() {
    xr_core::set_stream_bytes(BUDGET);
    // Chunks of 7 rows; a row is 40 * 50 * 8 = 16000 bytes, so the budget is 4 rows: each
    // slab is one whole chunk.
    let (a, src) = lazy_chunked(&["t", "y", "x"], &[60, 40, 50], false, Some(7));
    let am = loaded(&a);
    for op in [Reduction::Mean, Reduction::Std] {
        src.reads.lock().unwrap().clear();
        let got = a
            .reduce(op, vec![Name::from("t")], &ReduceOptions::default())
            .unwrap();
        same_or_close(
            op,
            &got,
            &am.reduce(op, vec![Name::from("t")], &ReduceOptions::default())
                .unwrap(),
            "t",
        );
        let reads = src.reads.lock().unwrap().clone();
        assert_eq!(
            reads.iter().map(|(_, n)| n).sum::<usize>(),
            60,
            "each row read once"
        );
        for (s, n) in &reads {
            assert_eq!(s % 7, 0, "{reads:?}");
            assert!((s + n) % 7 == 0 || s + n == 60, "{reads:?}");
        }
    }
    // A selection starting mid-chunk shifts the boundaries.
    let sub = a
        .isel([(
            "t",
            xr_core::Indexer::Slice(xr_core::PySlice::new(Some(3), None, None)),
        )])
        .unwrap()
        .to_owned();
    src.reads.lock().unwrap().clear();
    let got = sub
        .reduce(
            Reduction::Sum,
            vec![Name::from("t")],
            &ReduceOptions::default(),
        )
        .unwrap();
    let want = loaded(&sub)
        .reduce(
            Reduction::Sum,
            vec![Name::from("t")],
            &ReduceOptions::default(),
        )
        .unwrap();
    same(&got, &want, "sub");
    for (s, n) in src.reads.lock().unwrap().iter() {
        assert!(
            *s == 3 || s % 7 == 0,
            "starts on a boundary (or the selection start)"
        );
        assert!(
            (s + n) % 7 == 0 || s + n == 60,
            "ends on a boundary (or the end)"
        );
    }
}

#[test]
fn missing_values_stay_lazy() {
    let (a, src) = lazy(&["t", "y", "x"], &[20, 10, 8], false);
    let (c, _) = lazy(&["y"], &[10], false);
    let am = loaded(&a);
    let cond_lazy = c.notnull().unwrap();
    assert!(cond_lazy.variable().is_lazy());
    let cond_mem = loaded(&c).notnull().unwrap();
    for (name, got, want) in [
        ("isnull", a.isnull().unwrap(), am.isnull().unwrap()),
        ("notnull", a.notnull().unwrap(), am.notnull().unwrap()),
        ("fillna", a.fillna(-1.0).unwrap(), am.fillna(-1.0).unwrap()),
        (
            "where",
            a.where_(&cond_lazy).unwrap(),
            am.where_(&cond_mem).unwrap(),
        ),
        (
            "where_or",
            a.where_or(&cond_lazy, 7.0).unwrap(),
            am.where_or(&cond_mem, 7.0).unwrap(),
        ),
    ] {
        assert!(got.variable().is_lazy(), "{name}");
        let g = loaded(&got);
        assert_eq!(g.dims(), want.dims(), "{name}");
        assert_eq!(g.unit(), want.unit(), "{name}");
        match (
            g.variable().typed().unwrap(),
            want.variable().typed().unwrap(),
        ) {
            (TypedArray::Bool(x), TypedArray::Bool(y)) => assert_eq!(x, y, "{name}"),
            (x, y) => assert_eq!(bits(&x), bits(&y), "{name}"),
        }
        // One element reads one element.
        src.largest.store(0, Ordering::SeqCst);
        let one = got
            .isel([
                ("t", xr_core::Indexer::Int(2)),
                ("y", xr_core::Indexer::Int(3)),
                ("x", xr_core::Indexer::Int(4)),
            ])
            .unwrap()
            .to_owned();
        let _ = loaded(&DataArray::new(one.variable().to_owned()));
        assert_eq!(src.largest.load(Ordering::SeqCst), 1, "{name}");
    }
}

#[test]
fn take_reads_only_the_chosen_positions() {
    let (a, src) = lazy(&["t", "y", "x"], &[200, 10, 8], false);
    let am = loaded(&a);
    // A few runs: each read is one run.
    let idx: Vec<isize> = vec![5, 6, 7, 50, 51, 120, 3];
    let got = a.take("t", &idx).unwrap();
    assert!(got.variable().is_lazy());
    src.largest.store(0, Ordering::SeqCst);
    let g = loaded(&got);
    assert_eq!(
        src.largest.load(Ordering::SeqCst),
        3 * 10 * 8,
        "the longest run"
    );
    same(&g, &am.take("t", &idx).unwrap(), "runs");
    // Many scattered positions: one covering read, same values.
    let scattered: Vec<isize> = (0..100).map(|i| (i * 37 % 199) as isize).collect();
    same(
        &loaded(&a.take("t", &scattered).unwrap()),
        &am.take("t", &scattered).unwrap(),
        "scattered",
    );
    // Group-by over non-contiguous groups streams through lazy takes.
    let labels: Vec<i64> = (0..200).map(|i| i % 3).collect();
    let lab = Variable::new(
        ["t"],
        ArrayD::from_shape_vec(IxDyn(&[200]), labels).unwrap(),
    )
    .unwrap();
    let ga = a.clone().with_coord("k", lab.clone()).unwrap();
    let gm = am.clone().with_coord("k", lab).unwrap();
    xr_core::set_stream_bytes(BUDGET);
    src.largest.store(0, Ordering::SeqCst);
    let got = ga.groupby("k").unwrap().mean().unwrap();
    // A slab of a group (every third row) reads the rows its positions span: at most
    // four times as many as it keeps, plus slack.
    let slab_rows = (BUDGET / (10 * 8 * 8)).max(1);
    assert!(src.largest.load(Ordering::SeqCst) <= (slab_rows * 4 + 16) * 10 * 8);
    // Far-apart positions do not read what lies between them.
    src.largest.store(0, Ordering::SeqCst);
    let far: Vec<isize> = (0..70)
        .map(|i| if i % 2 == 0 { i } else { 199 - i })
        .collect();
    same(
        &loaded(&a.take("t", &far).unwrap()),
        &am.take("t", &far).unwrap(),
        "far apart",
    );
    assert!(
        src.largest.load(Ordering::SeqCst) < 200 * 10 * 8 / 2,
        "not the whole span"
    );
    same(
        &got,
        &gm.groupby("k").unwrap().mean().unwrap(),
        "groupby mean",
    );
}

#[test]
fn groupby_streams_in_one_pass() {
    xr_core::set_stream_bytes(BUDGET);
    for f32 in [false, true] {
        let (a, src) = lazy_chunked(&["t", "y", "x"], &[120, 10, 8], f32, Some(7));
        let am = loaded(&a);
        let labels: Vec<i64> = (0..120).map(|i| i % 12).collect();
        let lab = Variable::new(
            ["t"],
            ArrayD::from_shape_vec(IxDyn(&[120]), labels).unwrap(),
        )
        .unwrap();
        let ga = a.clone().with_coord("k", lab.clone()).unwrap();
        let gm = am.clone().with_coord("k", lab.clone()).unwrap();
        let mut ds_lazy = xr_core::Dataset::new();
        ds_lazy.add_array("v", ga.clone()).unwrap();
        let mut ds_mem = xr_core::Dataset::new();
        ds_mem.add_array("v", gm.clone()).unwrap();
        // An integer variable along the grouped dimension: reduced group by group.
        let n = Variable::new(
            ["t"],
            ArrayD::from_shape_fn(IxDyn(&[120]), |ix| ix[0] as i64 * 3),
        )
        .unwrap();
        ds_lazy.add_var("n", n.clone()).unwrap();
        ds_mem.add_var("n", n).unwrap();
        for op in [
            Reduction::Mean,
            Reduction::Sum,
            Reduction::Std,
            Reduction::Min,
            Reduction::Max,
        ] {
            src.reads.lock().unwrap().clear();
            let got = ga
                .groupby("k")
                .unwrap()
                .reduce(op, &ReduceOptions::default())
                .unwrap();
            let rows: usize = src.reads.lock().unwrap().iter().map(|(_, n)| n).sum();
            assert_eq!(rows, 120, "{op:?}: each row read once");
            let want = gm
                .groupby("k")
                .unwrap()
                .reduce(op, &ReduceOptions::default())
                .unwrap();
            same_or_close(op, &got, &want, &format!("f32={f32} {op:?}"));
            let dg = ds_lazy
                .groupby("k")
                .unwrap()
                .reduce(op, &ReduceOptions::default())
                .unwrap();
            let dw = ds_mem
                .groupby("k")
                .unwrap()
                .reduce(op, &ReduceOptions::default())
                .unwrap();
            same_or_close(
                op,
                &dg.get("v").unwrap().to_owned(),
                &dw.get("v").unwrap().to_owned(),
                &format!("dataset f32={f32} {op:?}"),
            );
            assert!(
                xr_core::variables_equal(
                    dg.get("n").unwrap().variable(),
                    dw.get("n").unwrap().variable()
                ),
                "dataset integer variable {op:?}"
            );
        }
    }
}
