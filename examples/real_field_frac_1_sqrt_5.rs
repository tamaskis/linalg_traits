use linalg_traits::RealField;

fn frac_1_sqrt_5<T: RealField>() -> T {
    T::frac_1_sqrt_5()
}

#[allow(clippy::excessive_precision)]
fn main() {
    assert_eq!(
        frac_1_sqrt_5::<f64>(),
        0.447_213_595_499_957_939_281_834_733_746_255_24
    );
}
