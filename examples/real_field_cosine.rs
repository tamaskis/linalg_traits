use linalg_traits::RealField;

fn cos<T: RealField>(x: T) -> T {
    let y1: T = x.cos();
    let y2: T = T::cos(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(cos(0_f64), 1_f64);
}
