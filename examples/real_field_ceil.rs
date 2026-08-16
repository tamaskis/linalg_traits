use linalg_traits::RealField;

fn ceil<T: RealField>(x: T) -> T {
    let y1: T = x.ceil();
    let y2: T = T::ceil(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(ceil(1.2_f64), 2_f64);
}
