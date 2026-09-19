use linalg_traits::RealField;

fn min<T: RealField>(x: T, y: T) -> T {
    let z1: T = x.min(y);
    let z2: T = T::min(x, y);

    assert_eq!(z1, z2);

    z1
}

fn main() {
    assert_eq!(min(1_f64, 2_f64), 1_f64);
}
