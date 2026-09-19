use linalg_traits::RealField;

fn acsch<T: RealField>(x: T) -> T {
    let y1: T = x.acsch();
    let y2: T = T::acsch(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(acsch(1_f64), 1_f64.recip().asinh());
}
