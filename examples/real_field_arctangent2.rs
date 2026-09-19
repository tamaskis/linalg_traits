use linalg_traits::RealField;

fn atan2<T: RealField>(y: T, x: T) -> T {
    let z1: T = y.atan2(x);
    let z2: T = T::atan2(y, x);

    assert_eq!(z1, z2);

    z1
}

fn main() {
    assert_eq!(atan2(0_f64, 1_f64), 0_f64);
}
