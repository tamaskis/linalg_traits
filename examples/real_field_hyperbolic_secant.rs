use linalg_traits::RealField;

fn sech<T: RealField>(x: T) -> T {
    let y1: T = x.sech();
    let y2: T = T::sech(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(sech(1_f64), 1_f64.cosh().recip());
}
