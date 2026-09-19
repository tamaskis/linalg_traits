use linalg_traits::RealField;

fn cosh<T: RealField>(x: T) -> T {
    let y1: T = x.cosh();
    let y2: T = T::cosh(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(cosh(0_f64), 1_f64);
}
