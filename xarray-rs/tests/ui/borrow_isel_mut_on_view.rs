// `isel!(mut ...)` needs a mutable representation; a shared view is not one.
use xarray_rs::{VariableView, isel};

fn f(mut v: VariableView<'_>) {
    let _ = isel!(mut v[0]);
}

fn main() {}
