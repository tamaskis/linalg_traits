use linalg_traits::RealField;

fn sin<T: RealField>(x: T) -> T {
    let y1: T = x.sin();
    let y2: T = T::sin(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(sin(0_f64), 0_f64);
}
