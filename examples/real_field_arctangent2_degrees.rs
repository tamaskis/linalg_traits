use linalg_traits::RealField;

fn atan2d<T: RealField>(y: T, x: T) -> T {
    let z1: T = y.atan2d(x);
    let z2: T = T::atan2d(y, x);

    assert_eq!(z1, z2);

    z1
}

fn main() {
    assert_eq!(atan2d(0_f64, 1_f64), 0_f64);
}
