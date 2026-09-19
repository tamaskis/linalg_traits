use linalg_traits::RealField;

fn acsc<T: RealField>(x: T) -> T {
    let y1: T = x.acsc();
    let y2: T = T::acsc(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(acsc(1_f64), 1_f64.recip().asin());
}
