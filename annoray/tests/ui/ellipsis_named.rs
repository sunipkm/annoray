use annoray::{Variable, isel};

fn f(v: &Variable) {
    let _ = isel!(v[time = ..]);
}

fn main() {}
