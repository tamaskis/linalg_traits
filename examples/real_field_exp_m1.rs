use linalg_traits::RealField;

fn exp_m1<T: RealField>(x: T) -> T {
    let y1: T = x.exp_m1();
    let y2: T = T::exp_m1(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(exp_m1(0_f64), 0_f64);
}
