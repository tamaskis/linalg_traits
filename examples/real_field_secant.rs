use linalg_traits::RealField;

fn sec<T: RealField>(x: T) -> T {
    let y1: T = x.sec();
    let y2: T = T::sec(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(sec(1_f64), 1_f64.cos().recip());
}
