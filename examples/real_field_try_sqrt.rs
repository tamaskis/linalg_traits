use linalg_traits::RealField;

fn try_sqrt<T: RealField>(x: T) -> Option<T> {
    let y1: Option<T> = x.try_sqrt();
    let y2: Option<T> = T::try_sqrt(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(try_sqrt(4_f64), Some(2_f64));
    assert_eq!(try_sqrt(-1_f64), None);
}
