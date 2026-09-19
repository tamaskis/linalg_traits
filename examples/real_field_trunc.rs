use linalg_traits::RealField;

fn trunc<T: RealField>(x: T) -> T {
    let y1: T = x.trunc();
    let y2: T = T::trunc(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(trunc(1.7_f64), 1_f64);
}
