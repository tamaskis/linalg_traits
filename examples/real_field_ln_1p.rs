use linalg_traits::RealField;

fn ln_1p<T: RealField>(x: T) -> T {
    let y1: T = x.ln_1p();
    let y2: T = T::ln_1p(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(ln_1p(0_f64), 0_f64);
}
