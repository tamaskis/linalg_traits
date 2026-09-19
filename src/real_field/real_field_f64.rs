use crate::real_field::real_field::RealField;
use crate::real_field::real_field_base::RealFieldBase;
use crate::verify_trait_implemented;

// Verify at compile time that the `RealField` trait is successfully implemented for `f64` via the
// blanket implementation provided by this crate.
const _: bool = verify_trait_implemented!(f64: RealField);

// Implement the `RealFieldBase` trait for `f64`.
//
// Note that we never call `impl_real_field!` on `f64` becuase it would reimplement traits that have
// already been implement (e.g. `std::ops::Add`). Therefore, some of the methods implemented here
// are never used, but we need to define them anyways so that `RealFieldBase`, and therefore
// `RealField`, are implemented for `f64`.
impl RealFieldBase for f64 {
    #[inline]
    fn _from_f64(value: f64) -> Self {
        value
    }

    #[inline]
    fn _to_f64(&self) -> f64 {
        *self
    }

    #[inline]
    fn _neg(self) -> Self {
        -self
    }

    #[inline]
    fn _add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline]
    fn _sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline]
    fn _mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn _div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline]
    fn _rem(self, rhs: Self) -> Self {
        self % rhs
    }

    #[inline]
    fn _eq(self, rhs: Self) -> bool {
        self == rhs
    }

    #[inline]
    fn _partial_cmp(self, rhs: Self) -> Option<std::cmp::Ordering> {
        <f64 as PartialOrd>::partial_cmp(&self, &rhs)
    }

    #[inline]
    fn _zero() -> Self {
        0.0
    }

    #[inline]
    fn _one() -> Self {
        1.0
    }

    #[inline]
    fn _abs(&self) -> Self {
        f64::abs(*self)
    }

    #[inline]
    fn _hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }

    #[inline]
    fn _recip(self) -> Self {
        f64::recip(self)
    }

    #[inline]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }

    #[inline]
    fn _sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline]
    fn _cbrt(self) -> Self {
        f64::cbrt(self)
    }

    #[inline]
    fn _powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }

    #[inline]
    fn _powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }

    #[inline]
    fn _exp(self) -> Self {
        f64::exp(self)
    }

    #[inline]
    fn _exp2(self) -> Self {
        f64::exp2(self)
    }

    #[inline]
    fn _exp_m1(self) -> Self {
        f64::exp_m1(self)
    }

    #[inline]
    fn _ln(self) -> Self {
        f64::ln(self)
    }

    #[inline]
    fn _ln_1p(self) -> Self {
        f64::ln_1p(self)
    }

    #[inline]
    fn _log(self, base: Self) -> Self {
        f64::log(self, base)
    }

    #[inline]
    fn _log2(self) -> Self {
        f64::log2(self)
    }

    #[inline]
    fn _log10(self) -> Self {
        f64::log10(self)
    }

    #[inline]
    fn _sin(self) -> Self {
        f64::sin(self)
    }

    #[inline]
    fn _cos(self) -> Self {
        f64::cos(self)
    }

    #[inline]
    fn _sin_cos(self) -> (Self, Self) {
        f64::sin_cos(self)
    }

    #[inline]
    fn _tan(self) -> Self {
        f64::tan(self)
    }

    #[inline]
    fn _asin(self) -> Self {
        f64::asin(self)
    }

    #[inline]
    fn _acos(self) -> Self {
        f64::acos(self)
    }

    #[inline]
    fn _atan(self) -> Self {
        f64::atan(self)
    }

    #[inline]
    fn _atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }

    #[inline]
    fn _sinh(self) -> Self {
        f64::sinh(self)
    }

    #[inline]
    fn _cosh(self) -> Self {
        f64::cosh(self)
    }

    #[inline]
    fn _tanh(self) -> Self {
        f64::tanh(self)
    }

    #[inline]
    fn _asinh(self) -> Self {
        f64::asinh(self)
    }

    #[inline]
    fn _acosh(self) -> Self {
        f64::acosh(self)
    }

    #[inline]
    fn _atanh(self) -> Self {
        f64::atanh(self)
    }

    #[inline]
    fn _floor(self) -> Self {
        f64::floor(self)
    }

    #[inline]
    fn _ceil(self) -> Self {
        f64::ceil(self)
    }

    #[inline]
    fn _round(self) -> Self {
        f64::round(self)
    }

    #[inline]
    fn _trunc(self) -> Self {
        f64::trunc(self)
    }

    #[inline]
    fn _fract(self) -> Self {
        f64::fract(self)
    }

    #[inline]
    fn _copysign(self, sign: Self) -> Self {
        f64::copysign(self, sign)
    }

    #[inline]
    fn _min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    #[inline]
    fn _max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    #[inline]
    fn _clamp(self, min: Self, max: Self) -> Self {
        f64::clamp(self, min, max)
    }

    #[inline]
    fn _is_nan(self) -> bool {
        f64::is_nan(self)
    }

    #[inline]
    fn _is_infinite(self) -> bool {
        f64::is_infinite(self)
    }

    #[inline]
    fn _is_finite(&self) -> bool {
        f64::is_finite(*self)
    }

    #[inline]
    fn _is_subnormal(self) -> bool {
        f64::is_subnormal(self)
    }

    #[inline]
    fn _is_normal(self) -> bool {
        f64::is_normal(self)
    }

    #[inline]
    fn _classify(self) -> std::num::FpCategory {
        f64::classify(self)
    }

    #[inline]
    fn _is_sign_positive(&self) -> bool {
        f64::is_sign_positive(*self)
    }

    #[inline]
    fn _is_sign_negative(&self) -> bool {
        f64::is_sign_negative(*self)
    }

    #[inline]
    fn _next_up(self) -> Self {
        f64::next_up(self)
    }

    #[inline]
    fn _next_down(self) -> Self {
        f64::next_down(self)
    }

    #[inline]
    fn _epsilon() -> Self {
        f64::EPSILON
    }

    #[inline]
    fn _bits() -> usize {
        64
    }

    #[inline]
    fn _min_positive() -> Self {
        f64::MIN_POSITIVE
    }

    #[inline]
    fn _max_positive() -> Self {
        f64::MAX
    }

    #[inline]
    fn _min_value() -> Option<Self> {
        Some(f64::MIN)
    }

    #[inline]
    fn _max_value() -> Option<Self> {
        Some(f64::MAX)
    }

    #[inline]
    fn _nan() -> Self {
        f64::NAN
    }

    #[inline]
    fn _infinity() -> Self {
        f64::INFINITY
    }

    #[inline]
    fn _as_slice(&self) -> &[f64] {
        std::slice::from_ref(self)
    }
}

#[cfg(test)]
#[allow(clippy::used_underscore_items)]
mod tests {
    use super::*;

    #[test]
    fn test_real_field_base_f64_constants() {
        assert!(f64::_nan()._is_nan());
        assert_eq!(f64::_infinity(), f64::INFINITY);
        assert_eq!(f64::_epsilon(), f64::EPSILON);
        assert_eq!(f64::_bits(), 64);
        assert_eq!(f64::_min_positive(), f64::MIN_POSITIVE);
        assert_eq!(f64::_max_positive(), f64::MAX);
        assert_eq!(f64::_pi(), std::f64::consts::PI);
        assert_eq!(f64::_two_pi(), std::f64::consts::TAU);
    }

    #[test]
    fn test_real_field_base_f64_operations() {
        assert_eq!(2.0_f64._powi(3), 8.0);
        assert_eq!(2.0_f64._powf(3.0), 8.0);
        assert_eq!(4.0_f64._sqrt(), 2.0);
        assert_eq!(8.0_f64._cbrt(), 2.0);
        assert_eq!(2.0_f64._exp2(), 4.0);
        assert_eq!(2.0_f64._ln(), std::f64::consts::LN_2);
        assert_eq!(2.0_f64._sin_cos(), (2.0_f64).sin_cos());
        assert_eq!(3.0_f64._hypot(4.0), 5.0);
        assert_eq!((-2.0_f64)._abs(), 2.0);
        assert!((-2.0_f64)._is_sign_negative());
    }
}
