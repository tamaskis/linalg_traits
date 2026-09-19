use linalg_traits::RealField;

fn clamp<T: RealField>(x: T, min: T, max: T) -> T {
    let y1: T = x.clamp(min, max);
    let y2: T = T::clamp(x, min, max);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(clamp(5_f64, 0_f64, 3_f64), 3_f64);
}
