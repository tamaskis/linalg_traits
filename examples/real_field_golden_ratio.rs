use linalg_traits::RealField;

fn golden_ratio<T: RealField>() -> T {
    T::golden_ratio()
}

#[allow(clippy::excessive_precision)]
fn main() {
    assert_eq!(
        golden_ratio::<f64>(),
        1.618_033_988_749_894_848_204_586_834_365_638_118
    );
}
