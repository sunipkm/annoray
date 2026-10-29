#[derive(annoray::schema::Schema)]
struct Bad<'a> {
    values: Vec<f64>,
    #[schema(attr)]
    title: &'a str,
}

fn main() {}
