use linalg_traits::RealField;

fn is_sign_negative<T: RealField>(x: T) -> bool {
    let y1: bool = x.is_sign_negative();
    let y2: bool = T::is_sign_negative(&x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert!(is_sign_negative(-1_f64));
    assert!(!is_sign_negative(1_f64));
}
