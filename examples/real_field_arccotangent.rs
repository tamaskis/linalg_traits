use linalg_traits::RealField;

fn acot<T: RealField>(x: T) -> T {
    let y1: T = x.acot();
    let y2: T = T::acot(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(acot(1_f64), 1_f64.recip().atan());
}
