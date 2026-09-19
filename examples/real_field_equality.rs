use linalg_traits::RealField;

fn eq<T: RealField>(x: T, y: T) -> (bool, bool) {
    (x == y, x != y)
}

fn main() {
    assert_eq!(eq(1_f64, 1_f64), (true, false));
    assert_eq!(eq(1_f64, 2_f64), (false, true));
}
