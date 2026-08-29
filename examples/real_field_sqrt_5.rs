use linalg_traits::RealField;

fn sqrt_5<T: RealField>() -> T {
    T::sqrt_5()
}

#[allow(clippy::excessive_precision)]
fn main() {
    assert_eq!(
        sqrt_5::<f64>(),
        2.236_067_977_499_789_696_409_173_668_731_276_23
    );
}
