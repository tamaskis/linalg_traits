use linalg_traits::RealField;

fn max<T: RealField>(x: T, y: T) -> T {
    let z1: T = x.max(y);
    let z2: T = T::max(x, y);

    assert_eq!(z1, z2);

    z1
}

fn main() {
    assert_eq!(max(1_f64, 2_f64), 2_f64);
}
