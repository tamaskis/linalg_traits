use linalg_traits::RealField;

fn ln<T: RealField>(x: T) -> T {
    let y1: T = x.ln();
    let y2: T = T::ln(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(ln(1_f64), 0_f64);
}
