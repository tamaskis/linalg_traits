use linalg_traits::RealField;
use std::num::FpCategory;

fn classify<T: RealField>(x: T) -> FpCategory {
    let y1: FpCategory = x.classify();
    let y2: FpCategory = T::classify(x);

    assert_eq!(y1, y2);

    y1
}

fn main() {
    assert_eq!(classify(0_f64), FpCategory::Zero);
    assert_eq!(classify(1_f64), FpCategory::Normal);
    assert_eq!(classify(f64::MIN_POSITIVE / 2_f64), FpCategory::Subnormal);
    assert_eq!(classify(f64::INFINITY), FpCategory::Infinite);
    assert_eq!(classify(f64::NAN), FpCategory::Nan);
}
