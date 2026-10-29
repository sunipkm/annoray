//! Schemas: every problem a dataset has, and typed views of one that has none.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::panic,
    clippy::arithmetic_side_effects
)]

use annoray::dims::{Length, Time};
use annoray::io::{ToNetcdf, open_dataset};
use annoray::ndarray::{ArrayD, ArrayView1, ArrayView2, ArrayViewD, IxDyn};
use annoray::schema::{Problem, ProblemKind, Schema, Var};
use annoray::{AttrValue, DType, Dataset, Variable};

fn map() -> Dataset {
    let x = Variable::new(["x"], ArrayD::from_shape_fn(IxDyn(&[4]), |i| i[0] as f64))
        .unwrap()
        .with_unit("km".parse().unwrap());
    let y = Variable::new(["y"], ArrayD::from_shape_fn(IxDyn(&[3]), |i| i[0] as f64))
        .unwrap()
        .with_unit("m".parse().unwrap());
    let intensity = Variable::new(
        ["y", "x"],
        ArrayD::from_shape_fn(IxDyn(&[3, 4]), |i| (i[0] * 4 + i[1]) as f32),
    )
    .unwrap()
    .with_unit("1".parse().unwrap())
    .with_attrs([("long_name", "intensity")]);
    Dataset::new()
        .with_coord("x", x)
        .unwrap()
        .with_coord("y", y)
        .unwrap()
        .with_var("intensity", intensity)
        .unwrap()
        .with_attrs([("format", "prism-map"), ("version", "2")])
}

fn shown(problems: &[Problem]) -> Vec<String> {
    problems.iter().map(ToString::to_string).collect()
}

#[test]
fn every_problem_is_reported() {
    let schema = Schema::new()
        .attr_eq("format", "prism-spectrum")
        .attr_in("version", ["1", "3"])
        .attr("author")
        .coord("x", Var::of::<f64>(["x"]).dimension::<Time>())
        .coord("y", Var::of::<f64>(["y"]).convertible("km").size("y", 3))
        .var(
            "intensity",
            Var::of::<f64>(["x", "y"])
                .unit("%")
                .min_size("x", 10)
                .attr_eq("long_name", "I"),
        )
        .var("x", Var::any())
        .var("counts", Var::of::<i64>(["x"]))
        .optional("mask", Var::of::<bool>(["y", "x"]))
        .var("bad", Var::any().unit("furlongs per"));
    let ds = map();
    assert_eq!(
        shown(&schema.problems(&ds)),
        [
            "attribute format: \"prism-map\", expected \"prism-spectrum\"",
            "attribute version: \"2\", expected one of \"1\", \"3\"",
            "attribute author: missing",
            "x: unit km is not of dimension s",
            "intensity: dimensions (y, x), expected (x, y)",
            "intensity: element type float32, expected float64",
            "intensity: unit 1, expected unit %",
            "intensity: dimension x of length 4, expected at least 10",
            "intensity attribute long_name: \"intensity\", expected \"I\"",
            "x: is a coordinate, expected a data variable",
            "counts: missing",
            "bad: missing",
        ]
    );
    let err = schema.check(&ds).unwrap_err();
    assert_eq!(err.problems.len(), 12);
    assert!(
        err.to_string()
            .starts_with("the dataset does not match the schema:\n  attribute")
    );
    // A unit in the schema that does not parse is the schema's problem.
    let bad = Schema::new().var("intensity", Var::any().unit("furlongs per"));
    let p = bad.problems(&ds);
    assert!(matches!(&p[0].kind, ProblemKind::BadSchema(_)), "{p:?}");
}

#[test]
fn matching_datasets_pass() {
    let schema = Schema::new()
        .attr_eq("format", "prism-map")
        .attr_in("version", ["1", "2"])
        .coord(
            "x",
            Var::of::<f64>(["x"]).dimension::<Length>().size("x", 4),
        )
        .coord("y", Var::of::<f64>(["y"]).convertible("km"))
        .var(
            "intensity",
            Var::of::<f32>(["y", "x"]).unit("1").attr("long_name"),
        )
        .optional("mask", Var::of::<bool>(["y", "x"]));
    schema.check(&map()).unwrap();
    // Units compare as units: km is 1000 m.
    Schema::new()
        .coord("x", Var::any().unit("1000 m"))
        .check(&map())
        .unwrap();
    // Extra variables are allowed unless the schema is closed.
    let open = Schema::new().var("intensity", Var::any());
    open.check(&map()).unwrap();
    assert_eq!(
        shown(&open.clone().closed().problems(&map())),
        ["x: not in the schema", "y: not in the schema"]
    );
    // No unit is not `1`.
    let ds = Dataset::new()
        .with_var(
            "n",
            Variable::new(["x"], ArrayD::from_elem(IxDyn(&[2]), 1i64)).unwrap(),
        )
        .unwrap();
    Schema::new()
        .var("n", Var::any().no_unit())
        .check(&ds)
        .unwrap();
    assert_eq!(
        shown(&Schema::new().var("n", Var::any().unit("1")).problems(&ds)),
        ["n: no unit, expected unit 1"]
    );
    assert_eq!(
        shown(
            &Schema::new()
                .var("n", Var::any().ndim(2).dtype(DType::I32))
                .problems(&ds)
        ),
        [
            "n: 1 dimensions, expected 2",
            "n: element type int64, expected int32"
        ]
    );
}

#[derive(annoray::schema::Schema)]
#[schema(attr_eq("format", "prism-map"), attr_in("version", "1", "2"))]
struct MapData<'a> {
    #[schema(coord, dims = ["x"], dimension = Length)]
    x: ArrayView1<'a, f64>,
    #[schema(coord, dims = ["y"])]
    y: &'a annoray::VariableBase<annoray::Owned>,
    #[schema(dims = ["y", "x"], unit = "1")]
    intensity: ArrayView2<'a, f32>,
    #[schema(dims = ["y", "x"])]
    mask: Option<ArrayView2<'a, bool>>,
    #[schema(attr)]
    version: &'a str,
    #[schema(attr, name = "format")]
    kind: String,
    #[schema(attr)]
    resolution: Option<f64>,
}

#[test]
fn derived_views_borrow_the_values() {
    let ds = map();
    let m = MapData::view(&ds).unwrap();
    assert_eq!(m.intensity[[2, 3]], 11.0);
    assert_eq!(m.x[3], 3.0);
    assert_eq!(m.y.unit().unwrap().to_string(), "m");
    assert!(m.mask.is_none() && m.resolution.is_none());
    assert_eq!((m.version, m.kind.as_str()), ("2", "prism-map"));
    // Borrowed, not copied.
    let original: ArrayViewD<'_, f32> = ds.data_vars()["intensity"].data().unwrap();
    assert_eq!(m.intensity.as_ptr(), original.as_ptr());

    // The schema the struct states, and every problem of a dataset that does not match.
    let schema = MapData::schema();
    assert!(schema.check(&ds).is_ok());
    let mut wrong = map().with_attrs([("version", "3")]);
    wrong.attrs_mut().shift_remove("format");
    wrong
        .add_var(
            "mask",
            Variable::new(["y", "x"], ArrayD::from_elem(IxDyn(&[3, 4]), 1u8)).unwrap(),
        )
        .unwrap();
    let e = MapData::view(&wrong).err().unwrap();
    let annoray::Error::Schema(e) = e else {
        panic!("{e}")
    };
    assert_eq!(
        shown(&e.problems),
        [
            "attribute format: missing",
            "attribute version: \"3\", expected one of \"1\", \"2\"",
            "mask: element type uint8, expected bool",
        ]
    );
}

#[derive(annoray::schema::Schema)]
#[schema(closed)]
struct Series<'a> {
    #[schema(coord, dims = ["time"])]
    time: ArrayViewD<'a, f64>,
    #[schema(dims = ["time"], convertible = "m s-1")]
    speed: ArrayView1<'a, f64>,
    #[schema(attr)]
    count: Option<i64>,
    #[schema(attr)]
    note: Option<&'a AttrValue>,
}

#[test]
fn lazy_values_are_read_by_the_view() {
    let time = Variable::new(
        ["time"],
        ArrayD::from_shape_fn(IxDyn(&[5]), |i| i[0] as f64),
    )
    .unwrap();
    let speed = Variable::new(["time"], ArrayD::from_elem(IxDyn(&[5]), 2.5))
        .unwrap()
        .with_unit("km h-1".parse().unwrap());
    let ds = Dataset::new()
        .with_coord("time", time)
        .unwrap()
        .with_var("speed", speed)
        .unwrap()
        .with_attrs([("count", AttrValue::I32(vec![5]))]);
    let path = std::env::temp_dir().join(format!("annoray-schema-{}.nc", std::process::id()));
    ds.to_netcdf(&path).unwrap();
    let opened = open_dataset(&path).unwrap();
    assert!(opened.data_vars()["speed"].is_lazy());
    let s = Series::view(&opened).unwrap();
    assert_eq!(s.speed.sum(), 12.5);
    assert_eq!(s.time.len(), 5);
    assert_eq!(s.count, Some(5));
    assert!(s.note.is_none());
    // Views of datasets that are themselves views.
    let part = opened.isel([("time", 1..3)]).unwrap();
    assert_eq!(Series::view(&part).unwrap().speed.len(), 2);
    // Closed: another variable is a problem.
    let more = ds
        .with_var(
            "extra",
            Variable::new(["time"], ArrayD::from_elem(IxDyn(&[5]), 0.0)).unwrap(),
        )
        .unwrap();
    let e = Series::view(&more).err().unwrap().to_string();
    assert!(e.contains("extra: not in the schema"), "{e}");
}
