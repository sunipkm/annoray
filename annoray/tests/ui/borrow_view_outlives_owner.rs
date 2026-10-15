// A view cannot outlive the variable it borrows from.
use annoray::ndarray::{ArrayD, IxDyn};
use annoray::{Variable, VariableView};

fn make() -> VariableView<'static> {
    let v = Variable::new(["x"], ArrayD::<f64>::zeros(IxDyn(&[3]))).unwrap();
    v.view()
}

fn main() {}
