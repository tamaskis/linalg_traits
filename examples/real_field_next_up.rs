use linalg_traits::RealField;

fn next_up<T: RealField>(x: T) -> T {
    let y1: T = x.next_up();
    let y2: T = T::next_up(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(next_up(1_f64), 1_f64.next_up());
}
