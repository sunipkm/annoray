//! Stores written by zarr-python (`fixtures/gen_zarr.py`): every array reads back as
//! zarr-python wrote it, whole and by region.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::arithmetic_side_effects
)]

use std::path::PathBuf;

use annoray_zarr::{Array, ArrayData, Group, Node, Store, Value, ZarrFormat};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/zarr")
}

fn expected() -> serde_json::Map<String, Value> {
    let text = std::fs::read_to_string(fixtures().join("expected.json")).unwrap();
    serde_json::from_str::<Value>(&text)
        .unwrap()
        .as_object()
        .unwrap()
        .clone()
}

/// The values as JSON, NaN as "NaN", as the manifest has them.
fn as_json(d: &ArrayData) -> Vec<Value> {
    as_json_opt(d).unwrap()
}

fn as_json_opt(d: &ArrayData) -> Option<Vec<Value>> {
    fn f(x: f64) -> Value {
        if x.is_nan() {
            Value::String("NaN".into())
        } else {
            serde_json::json!(x)
        }
    }
    Some(match d {
        ArrayData::Bool(a) => a.iter().map(|v| Value::Bool(*v)).collect(),
        ArrayData::I8(a) => a.iter().map(|v| serde_json::json!(v)).collect(),
        ArrayData::I16(a) => a.iter().map(|v| serde_json::json!(v)).collect(),
        ArrayData::I32(a) => a.iter().map(|v| serde_json::json!(v)).collect(),
        ArrayData::I64(a) => a.iter().map(|v| serde_json::json!(v)).collect(),
        ArrayData::U8(a) => a.iter().map(|v| serde_json::json!(v)).collect(),
        ArrayData::U16(a) => a.iter().map(|v| serde_json::json!(v)).collect(),
        ArrayData::U32(a) => a.iter().map(|v| serde_json::json!(v)).collect(),
        ArrayData::U64(a) => a.iter().map(|v| serde_json::json!(v)).collect(),
        ArrayData::F32(a) => a.iter().map(|v| f(f64::from(*v))).collect(),
        ArrayData::F64(a) => a.iter().map(|v| f(*v)).collect(),
        ArrayData::Str(a) => a.iter().map(|v| Value::String(v.clone())).collect(),
        _ => return None,
    })
}

fn same(got: &[Value], want: &[Value], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (g, w) in got.iter().zip(want) {
        let ok = match (g.as_f64(), w.as_f64()) {
            (Some(a), Some(b)) => a == b || (a - b).abs() <= 1e-6 * b.abs(),
            _ => g == w,
        };
        assert!(ok, "{what}: {g} vs {w}");
    }
}

#[test]
fn every_fixture_array_reads_back() {
    let want = expected();
    let mut seen = 0;
    for (key, spec) in &want {
        let (fmt, path) = key.split_once('/').unwrap();
        let store = Store::new(fixtures().join(format!("v{fmt}.zarr")));
        let a = Array::open(&store, path).unwrap_or_else(|e| panic!("{key}: {e}"));
        let shape: Vec<usize> = serde_json::from_value(spec["shape"].clone()).unwrap();
        assert_eq!(a.shape(), shape.as_slice(), "{key}");
        let data = a.read().unwrap_or_else(|e| panic!("{key}: {e}"));
        same(&as_json(&data), spec["values"].as_array().unwrap(), key);
        seen += 1;
    }
    assert_eq!(seen, want.len());
}

#[test]
fn regions_decode_only_what_they_need() {
    let want = expected();
    for fmt in ["2", "3"] {
        let store = Store::new(fixtures().join(format!("v{fmt}.zarr")));
        let a = Array::open(&store, "comp_zstd").unwrap();
        let all = as_json(&a.read().unwrap());
        // Rows 2..6, columns 1..4 of a 7×5 array chunked 3×2: crosses chunk edges.
        let part = as_json(&a.read_region(&[2, 1], &[4, 3]).unwrap());
        let mut expect = Vec::new();
        for r in 2..6 {
            for c in 1..4 {
                expect.push(all[r * 5 + c].clone());
            }
        }
        same(&part, &expect, "region");
        assert!(a.read_region(&[5, 0], &[3, 1]).is_err());
        let _ = &want;
    }
}

#[test]
fn hierarchy_attributes_and_names() {
    let v2 = Group::open(&Store::new(fixtures().join("v2.zarr")), "").unwrap();
    assert_eq!(v2.format(), ZarrFormat::V2);
    assert_eq!(v2.attrs()["title"], "v2 fixtures");
    assert!(v2.members().unwrap().iter().any(|(n, _)| n == "vlen"));
    let v3 = Group::open(&Store::new(fixtures().join("v3.zarr")), "").unwrap();
    assert_eq!(v3.format(), ZarrFormat::V3);
    let sub = v3.group("sub").unwrap();
    assert_eq!(sub.attrs()["level"], 1);
    assert!(matches!(&sub.members().unwrap()[0].1, Node::Array(a) if a.shape() == [3]));
    let a = v3.array("float64").unwrap();
    assert_eq!(a.dimension_names().unwrap(), ["y", "x"]);
    assert!(Group::open(&Store::new(fixtures().join("nope.zarr")), "").is_err());
    assert!(Array::open(&Store::new(fixtures().join("v3.zarr")), "../v2.zarr/vlen").is_err());
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(200))]

    /// Corrupted chunks give errors, never panics.
    #[test]
    fn corrupted_chunks_never_panic(
        which in 0usize..6,
        edits in proptest::collection::vec((proptest::prelude::any::<usize>(), proptest::prelude::any::<u8>()), 1..12),
        cut in proptest::prelude::any::<usize>(),
    ) {
        let arrays = [
            ("v2", "comp_blosc_lz4_shuffle"),
            ("v2", "comp_bz2"),
            ("v2", "vlen"),
            ("v3", "sharded"),
            ("v3", "comp_crc32c"),
            ("v3", "string"),
        ];
        let (fmt, name) = arrays[which];
        let src = fixtures().join(format!("{fmt}.zarr"));
        let tmp = std::env::temp_dir().join(format!("annoray-zarr-fuzz-{}-{which}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        copy_dir(&src.join(name), &tmp.join(name));
        let store = Store::new(&tmp);
        let a = Array::open(&store, name).unwrap();
        let key = a.chunk_key(&vec![0; a.shape().len()]);
        if let Some(mut bytes) = store.get(&key).unwrap() {
            for (at, b) in &edits {
                if !bytes.is_empty() {
                    let n = bytes.len();
                    bytes[at % n] ^= b | 1;
                }
            }
            bytes.truncate(cut % (bytes.len() + 1));
            store.set(&key, &bytes).unwrap();
            let _ = a.read();
        }
        std::fs::remove_dir_all(&tmp).ok();
    }
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let p = e.path();
        if p.is_dir() {
            copy_dir(&p, &to.join(e.file_name()));
        } else {
            std::fs::copy(&p, to.join(e.file_name())).unwrap();
        }
    }
}
