use linalg_traits::RealField;

fn log<T: RealField>(x: T, base: T) -> T {
    let y1: T = x.log(base);
    let y2: T = T::log(x, base);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(log(8_f64, 2_f64), 3_f64);
}
