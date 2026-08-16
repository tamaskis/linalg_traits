use linalg_traits::RealField;

fn round<T: RealField>(x: T) -> T {
    let y1: T = x.round();
    let y2: T = T::round(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(round(1.5_f64), 2_f64);
}
