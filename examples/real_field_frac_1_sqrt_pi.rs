use linalg_traits::RealField;

fn frac_1_sqrt_pi<T: RealField>() -> T {
    T::frac_1_sqrt_pi()
}

#[allow(clippy::excessive_precision)]
fn main() {
    assert_eq!(
        frac_1_sqrt_pi::<f64>(),
        0.564_189_583_547_756_286_948_079_451_560_772_586
    );
}
