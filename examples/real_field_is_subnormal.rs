use linalg_traits::RealField;

fn is_subnormal<T: RealField>(x: T) -> bool {
    let y1: bool = x.is_subnormal();
    let y2: bool = T::is_subnormal(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert!(is_subnormal(f64::MIN_POSITIVE / 2_f64));
    assert!(!is_subnormal(1_f64));
}
