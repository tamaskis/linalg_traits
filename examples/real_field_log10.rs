use linalg_traits::RealField;

fn log10<T: RealField>(x: T) -> T {
    let y1: T = x.log10();
    let y2: T = T::log10(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(log10(1000_f64), 3_f64);
}
