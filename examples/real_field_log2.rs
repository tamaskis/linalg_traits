use linalg_traits::RealField;

fn log2<T: RealField>(x: T) -> T {
    let y1: T = x.log2();
    let y2: T = T::log2(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(log2(8_f64), 3_f64);
}
