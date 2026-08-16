use linalg_traits::RealField;

fn is_finite<T: RealField>(x: T) -> bool {
    let y1: bool = x.is_finite();
    let y2: bool = T::is_finite(&x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert!(is_finite(1_f64));
    assert!(!is_finite(f64::INFINITY));
}
