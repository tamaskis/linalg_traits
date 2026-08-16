use linalg_traits::RealField;

fn fract<T: RealField>(x: T) -> T {
    let y1: T = x.fract();
    let y2: T = T::fract(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(fract(1.25_f64), 0.25_f64);
}
