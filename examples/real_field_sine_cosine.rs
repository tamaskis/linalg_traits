use linalg_traits::RealField;

fn sin_cos<T: RealField>(x: T) -> (T, T) {
    let y1: (T, T) = x.sin_cos();
    let y2: (T, T) = T::sin_cos(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(sin_cos(0_f64), (0_f64, 1_f64));
}
