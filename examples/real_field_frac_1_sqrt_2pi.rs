use linalg_traits::RealField;

fn frac_1_sqrt_2pi<T: RealField>() -> T {
    T::frac_1_sqrt_2pi()
}

#[allow(clippy::excessive_precision)]
fn main() {
    assert_eq!(
        frac_1_sqrt_2pi::<f64>(),
        0.398_942_280_401_432_677_939_946_059_934_381_868
    );
}
