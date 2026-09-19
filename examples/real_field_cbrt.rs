use linalg_traits::RealField;

fn cbrt<T: RealField>(x: T) -> T {
    let y1: T = x.cbrt();
    let y2: T = T::cbrt(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(cbrt(8_f64), 2_f64);
}
