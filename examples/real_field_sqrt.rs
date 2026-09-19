use linalg_traits::RealField;

fn sqrt<T: RealField>(x: T) -> T {
    let y1: T = x.sqrt();
    let y2: T = T::sqrt(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(sqrt(4_f64), 2_f64);
}
