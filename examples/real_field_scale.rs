use linalg_traits::RealField;

fn scale<T: RealField>(x: T, factor: T) -> T {
    let y1: T = x.scale(factor);
    let y2: T = T::scale(x, factor);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(scale(2_f64, 3_f64), 6_f64);
}
