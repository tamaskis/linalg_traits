use linalg_traits::RealField;

fn floor<T: RealField>(x: T) -> T {
    let y1: T = x.floor();
    let y2: T = T::floor(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(floor(1.7_f64), 1_f64);
}
