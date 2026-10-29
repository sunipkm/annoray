use annoray::ndarray::ArrayView1;

#[derive(annoray::schema::Schema)]
struct Bad<'a> {
    #[schema(dims = ["x"], units = "m")]
    x: ArrayView1<'a, f64>,
}

fn main() {}
