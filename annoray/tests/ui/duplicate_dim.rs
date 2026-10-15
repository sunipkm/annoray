use annoray::{Variable, isel};

fn f(v: &Variable) {
    let _ = isel!(v[time = 0, time = 1]);
}

fn main() {}
