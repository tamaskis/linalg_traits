use linalg_traits::RealField;

fn is_sign_positive<T: RealField>(x: T) -> bool {
    let y1: bool = x.is_sign_positive();
    let y2: bool = T::is_sign_positive(&x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert!(is_sign_positive(1_f64));
    assert!(!is_sign_positive(-1_f64));
}
