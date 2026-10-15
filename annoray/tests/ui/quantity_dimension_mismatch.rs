// Adding a length to a time does not compile.
use annoray::units::Quantity;
use annoray::units::dims::{Length, Time};

fn main() {
    let d = Quantity::<Length>::new(1.0, "m").unwrap();
    let t = Quantity::<Time>::new(1.0, "s").unwrap();
    let _ = d + t;
}
