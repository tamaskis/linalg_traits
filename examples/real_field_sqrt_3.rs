use linalg_traits::RealField;

fn sqrt_3<T: RealField>() -> T {
    T::sqrt_3()
}

#[allow(clippy::excessive_precision)]
fn main() {
    assert_eq!(
        sqrt_3::<f64>(),
        1.732_050_807_568_877_293_527_446_341_505_872_367
    );
}
