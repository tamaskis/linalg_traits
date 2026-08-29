use linalg_traits::RealField;

fn sinh_cosh<T: RealField>(x: T) -> (T, T) {
    let y1: (T, T) = x.sinh_cosh();
    let y2: (T, T) = T::sinh_cosh(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(sinh_cosh(0_f64), (0_f64, 1_f64));
}
