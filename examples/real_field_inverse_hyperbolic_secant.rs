use linalg_traits::RealField;

fn asech<T: RealField>(x: T) -> T {
    let y1: T = x.asech();
    let y2: T = T::asech(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(asech(1_f64), 0_f64);
}
