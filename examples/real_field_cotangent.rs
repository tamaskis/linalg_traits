use linalg_traits::RealField;

fn cot<T: RealField>(x: T) -> T {
    let y1: T = x.cot();
    let y2: T = T::cot(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(cot(1_f64), 1_f64.cos() / 1_f64.sin());
}
