use linalg_traits::RealField;

fn next_down<T: RealField>(x: T) -> T {
    let y1: T = x.next_down();
    let y2: T = T::next_down(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(next_down(1_f64), 1_f64.next_down());
}
