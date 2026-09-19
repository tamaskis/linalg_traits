use linalg_traits::RealField;

fn acoth<T: RealField>(x: T) -> T {
    let y1: T = x.acoth();
    let y2: T = T::acoth(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(acoth(2_f64), 2_f64.recip().atanh());
}
