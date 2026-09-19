use linalg_traits::RealField;

fn ord<T: RealField>(x: T, y: T) -> (bool, bool, bool, bool) {
    (x < y, x <= y, x > y, x >= y)
}

fn main() {
    assert_eq!(ord(1_f64, 2_f64), (true, true, false, false));
    assert_eq!(ord(2_f64, 2_f64), (false, true, false, true));
    assert_eq!(ord(3_f64, 2_f64), (false, false, true, true));
}
