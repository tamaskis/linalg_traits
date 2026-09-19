use linalg_traits::RealField;

fn hypot<T: RealField>(x: T, y: T) -> T {
    let z1: T = x.hypot(y);
    let z2: T = T::hypot(x, y);

    assert_eq!(z1, z2);

    z1
}

fn main() {
    assert_eq!(hypot(3_f64, 4_f64), 5_f64);
}
