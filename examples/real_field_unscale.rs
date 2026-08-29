use linalg_traits::RealField;

fn unscale<T: RealField>(x: T, factor: T) -> T {
    let y1: T = x.unscale(factor);
    let y2: T = T::unscale(x, factor);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(unscale(6_f64, 3_f64), 2_f64);
}
