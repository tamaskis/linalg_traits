use linalg_traits::RealField;

// When the `nalgebra` feature is active, the `abs` method always has to be disambiguated because
// `num_traits::Signed` and `simba::scalar::ComplexField` (both of which are supertraits of
// `nalgebra::RealField`) define an `abs` method.
#[cfg(feature = "nalgebra")]
fn abs<T: RealField>(x: T) -> T {
    let y1: T = x.abs();
    let y2: T = <T as num_traits::Signed>::abs(&x);

    assert_eq!(y1, y2);

    y1
}

#[cfg(not(feature = "nalgebra"))]
fn abs<T: RealField>(x: T) -> T {
    let y1: T = x.abs();
    let y2: T = T::abs(&x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(abs(-2_f64), 2_f64);
}
