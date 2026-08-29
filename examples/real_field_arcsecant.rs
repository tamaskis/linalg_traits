use linalg_traits::RealField;

fn asec<T: RealField>(x: T) -> T {
    let y1: T = x.asec();
    let y2: T = T::asec(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(asec(1_f64), 1_f64.recip().acos());
}
