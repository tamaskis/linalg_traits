use linalg_traits::RealField;

fn convert<T: RealField>(x: T, y: f64) -> (f64, T) {
    let a: f64 = x.into();
    let b: f64 = T::into(x);
    let c: f64 = f64::from(x);

    assert_eq!(a.to_bits(), b.to_bits());
    assert_eq!(b.to_bits(), c.to_bits());

    let d: T = y.into();
    let e: T = f64::into(y);
    let f: T = T::from(y);

    assert_eq!(d, e);
    assert_eq!(e, f);

    (a, d)
}

fn main() {
    assert_eq!(convert(1_f64, 2_f64), (1_f64, 2_f64));
}
