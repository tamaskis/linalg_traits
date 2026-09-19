use linalg_traits::RealField;

fn acosh<T: RealField>(x: T) -> T {
    let y1: T = x.acosh();
    let y2: T = T::acosh(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(acosh(1_f64), 0_f64);
}
