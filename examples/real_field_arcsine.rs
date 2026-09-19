use linalg_traits::RealField;

fn asin<T: RealField>(x: T) -> T {
    let y1: T = x.asin();
    let y2: T = T::asin(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(asin(0_f64), 0_f64);
}
