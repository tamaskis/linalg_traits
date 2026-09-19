use linalg_traits::RealField;

fn is_normal<T: RealField>(x: T) -> bool {
    let y1: bool = x.is_normal();
    let y2: bool = T::is_normal(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert!(is_normal(1_f64));
    assert!(!is_normal(f64::MIN_POSITIVE / 2_f64));
}
