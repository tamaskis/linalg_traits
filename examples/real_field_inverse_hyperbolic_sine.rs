use linalg_traits::RealField;

fn asinh<T: RealField>(x: T) -> T {
    let y1: T = x.asinh();
    let y2: T = T::asinh(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(asinh(0_f64), 0_f64);
}
