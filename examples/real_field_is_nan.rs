use linalg_traits::RealField;

fn is_nan<T: RealField>(x: T) -> bool {
    let y1: bool = x.is_nan();
    let y2: bool = T::is_nan(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert!(is_nan(f64::NAN));
    assert!(!is_nan(1_f64));
}
