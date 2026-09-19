use linalg_traits::RealField;

fn csc<T: RealField>(x: T) -> T {
    let y1: T = x.csc();
    let y2: T = T::csc(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(csc(1_f64), 1_f64.sin().recip());
}
