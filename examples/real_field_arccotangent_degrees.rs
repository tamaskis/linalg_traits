use linalg_traits::RealField;

fn acotd<T: RealField>(x: T) -> T {
    let y1: T = x.acotd();
    let y2: T = T::acotd(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(acotd(2_f64), 2_f64.acot().to_degrees());
}
