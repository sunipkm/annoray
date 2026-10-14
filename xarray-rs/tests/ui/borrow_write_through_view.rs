// Shared views do not expose mutable data access.
use xarray_rs::VariableView;

fn f(mut v: VariableView<'_>) {
    let _ = v.data_mut::<f64>();
}

fn main() {}
