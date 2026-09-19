use linalg_traits::RealField;

fn frac_1_sqrt_3<T: RealField>() -> T {
    T::frac_1_sqrt_3()
}

#[allow(clippy::excessive_precision)]
fn main() {
    assert_eq!(
        frac_1_sqrt_3::<f64>(),
        0.577_350_269_189_625_764_509_148_780_501_957_456
    );
}
