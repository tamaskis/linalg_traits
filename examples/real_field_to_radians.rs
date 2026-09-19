use linalg_traits::RealField;

fn to_radians<T: RealField>(x: T) -> T {
    let y1: T = x.to_radians();
    let y2: T = T::to_radians(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(to_radians(0_f64), 0_f64);
}
