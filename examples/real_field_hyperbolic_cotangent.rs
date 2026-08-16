use linalg_traits::RealField;

fn coth<T: RealField>(x: T) -> T {
    let y1: T = x.coth();
    let y2: T = T::coth(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(coth(1_f64), 1_f64.cosh() / 1_f64.sinh());
}
