use linalg_traits::RealField;

fn sinh<T: RealField>(x: T) -> T {
    let y1: T = x.sinh();
    let y2: T = T::sinh(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(sinh(0_f64), 0_f64);
}
