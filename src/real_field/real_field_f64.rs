use crate::real_field::real_field::RealField;
use crate::real_field::real_field_base::RealFieldBase;
use crate::verify_trait_implemented;

// Verify at compile time that the `RealField` trait is successfully implemented for `f64` via the
// blanket implementation provided by this crate.
const _: bool = verify_trait_implemented!(f64: RealField);

impl RealFieldBase for f64 {
    #[inline]
    fn nan() -> Self {
        f64::NAN
    }

    #[inline]
    fn infinity() -> Self {
        f64::INFINITY
    }

    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }

    #[inline]
    fn recip(self) -> Self {
        f64::recip(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }

    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline]
    fn cbrt(self) -> Self {
        f64::cbrt(self)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }

    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }

    #[inline]
    fn exp2(self) -> Self {
        f64::exp2(self)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        f64::exp_m1(self)
    }

    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        f64::ln_1p(self)
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        f64::log(self, base)
    }

    #[inline]
    fn log2(self) -> Self {
        f64::log2(self)
    }

    #[inline]
    fn log10(self) -> Self {
        f64::log10(self)
    }

    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        f64::sin_cos(self)
    }

    #[inline]
    fn tan(self) -> Self {
        f64::tan(self)
    }

    #[inline]
    fn asin(self) -> Self {
        f64::asin(self)
    }

    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }

    #[inline]
    fn atan(self) -> Self {
        f64::atan(self)
    }

    #[inline]
    fn sinh(self) -> Self {
        f64::sinh(self)
    }

    #[inline]
    fn cosh(self) -> Self {
        f64::cosh(self)
    }

    #[inline]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }

    #[inline]
    fn asinh(self) -> Self {
        f64::asinh(self)
    }

    #[inline]
    fn acosh(self) -> Self {
        f64::acosh(self)
    }

    #[inline]
    fn atanh(self) -> Self {
        f64::atanh(self)
    }

    #[inline]
    fn floor(self) -> Self {
        f64::floor(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        f64::ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        f64::round(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        f64::trunc(self)
    }

    #[inline]
    fn fract(self) -> Self {
        f64::fract(self)
    }

    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    #[inline]
    fn is_infinite(self) -> bool {
        f64::is_infinite(self)
    }

    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    #[inline]
    fn is_subnormal(self) -> bool {
        f64::is_subnormal(self)
    }

    #[inline]
    fn is_normal(self) -> bool {
        f64::is_normal(self)
    }

    #[inline]
    fn classify(self) -> std::num::FpCategory {
        f64::classify(self)
    }

    #[inline]
    fn is_sign_positive(self) -> bool {
        f64::is_sign_positive(self)
    }

    #[inline]
    fn is_sign_negative(self) -> bool {
        f64::is_sign_negative(self)
    }

    #[inline]
    fn epsilon() -> Self {
        f64::EPSILON
    }

    #[inline]
    fn next_up(self) -> Self {
        f64::next_up(self)
    }

    #[inline]
    fn next_down(self) -> Self {
        f64::next_down(self)
    }

    #[inline]
    fn bits() -> usize {
        64
    }

    #[inline]
    fn min_positive() -> Self {
        f64::MIN_POSITIVE
    }

    #[inline]
    fn max_positive() -> Self {
        f64::MAX
    }

    #[inline]
    fn min_value() -> Option<Self> {
        Some(f64::MIN)
    }

    #[inline]
    fn max_value() -> Option<Self> {
        Some(f64::MAX)
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f64::copysign(self, sign)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        f64::clamp(self, min, max)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }

    #[inline]
    fn as_slice(&self) -> &[f64] {
        std::slice::from_ref(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_field_base_f64_constants() {
        assert!(f64::nan().is_nan());
        assert_eq!(f64::infinity(), f64::INFINITY);
        assert_eq!(f64::epsilon(), f64::EPSILON);
        assert_eq!(f64::bits(), 64);
        assert_eq!(f64::min_positive(), f64::MIN_POSITIVE);
        assert_eq!(f64::max_positive(), f64::MAX);
        assert_eq!(f64::pi(), std::f64::consts::PI);
        assert_eq!(f64::two_pi(), std::f64::consts::TAU);
    }

    #[test]
    fn test_real_field_base_f64_operations() {
        assert_eq!(2.0_f64.powi(3), 8.0);
        assert_eq!(2.0_f64.powf(3.0), 8.0);
        assert_eq!(4.0_f64.sqrt(), 2.0);
        assert_eq!(8.0_f64.cbrt(), 2.0);
        assert_eq!(2.0_f64.exp2(), 4.0);
        assert_eq!(2.0_f64.ln(), std::f64::consts::LN_2);
        assert_eq!(2.0_f64.sin_cos(), (2.0_f64).sin_cos());
        assert_eq!(3.0_f64.hypot(4.0), 5.0);
        assert_eq!((-2.0_f64).abs(), 2.0);
        assert!((-2.0_f64).is_sign_negative());
    }
}
