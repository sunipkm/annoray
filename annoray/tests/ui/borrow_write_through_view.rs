// Shared views do not expose mutable data access.
use annoray::VariableView;

fn f(mut v: VariableView<'_>) {
    let _ = v.data_mut::<f64>();
}

fn main() {}
