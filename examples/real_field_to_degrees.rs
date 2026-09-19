use linalg_traits::RealField;

fn to_degrees<T: RealField>(x: T) -> T {
    let y1: T = x.to_degrees();
    let y2: T = T::to_degrees(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(to_degrees(0_f64), 0_f64);
}
