use linalg_traits::RealField;

fn is_infinite<T: RealField>(x: T) -> bool {
    let y1: bool = x.is_infinite();
    let y2: bool = T::is_infinite(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert!(is_infinite(f64::INFINITY));
    assert!(!is_infinite(1_f64));
}
