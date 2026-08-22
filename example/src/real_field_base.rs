use crate::my_scalar::MyScalar;
use linalg_traits::real_field::RealFieldBase;
use linalg_traits::verify_trait_implemented;

const _: bool = verify_trait_implemented!(MyScalar: RealFieldBase);

impl RealFieldBase for MyScalar {
    #[inline]
    fn nan() -> Self {
        Self::new(f64::NAN)
    }

    #[inline]
    fn infinity() -> Self {
        Self::new(f64::INFINITY)
    }

    #[inline]
    fn abs(self) -> Self {
        Self::new(self.x.abs())
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self::new(self.x.hypot(other.x))
    }

    #[inline]
    fn recip(self) -> Self {
        Self::new(self.x.recip())
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::new(self.x.mul_add(a.x, b.x))
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::new(self.x.sqrt())
    }

    #[inline]
    fn cbrt(self) -> Self {
        Self::new(self.x.cbrt())
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Self::new(self.x.powi(n))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        Self::new(self.x.powf(n.x))
    }

    #[inline]
    fn exp(self) -> Self {
        Self::new(self.x.exp())
    }

    #[inline]
    fn exp2(self) -> Self {
        Self::new(self.x.exp2())
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Self::new(self.x.exp_m1())
    }

    #[inline]
    fn ln(self) -> Self {
        Self::new(self.x.ln())
    }

    #[inline]
    fn ln_1p(self) -> Self {
        Self::new(self.x.ln_1p())
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        Self::new(self.x.log(base.x))
    }

    #[inline]
    fn log2(self) -> Self {
        Self::new(self.x.log2())
    }

    #[inline]
    fn log10(self) -> Self {
        Self::new(self.x.log10())
    }

    #[inline]
    fn sin(self) -> Self {
        Self::new(self.x.sin())
    }

    #[inline]
    fn cos(self) -> Self {
        Self::new(self.x.cos())
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.x.sin_cos();
        (Self::new(sin), Self::new(cos))
    }

    #[inline]
    fn tan(self) -> Self {
        Self::new(self.x.tan())
    }

    #[inline]
    fn asin(self) -> Self {
        Self::new(self.x.asin())
    }

    #[inline]
    fn acos(self) -> Self {
        Self::new(self.x.acos())
    }

    #[inline]
    fn atan(self) -> Self {
        Self::new(self.x.atan())
    }

    #[inline]
    fn sinh(self) -> Self {
        Self::new(self.x.sinh())
    }

    #[inline]
    fn cosh(self) -> Self {
        Self::new(self.x.cosh())
    }

    #[inline]
    fn tanh(self) -> Self {
        Self::new(self.x.tanh())
    }

    #[inline]
    fn asinh(self) -> Self {
        Self::new(self.x.asinh())
    }

    #[inline]
    fn acosh(self) -> Self {
        Self::new(self.x.acosh())
    }

    #[inline]
    fn atanh(self) -> Self {
        Self::new(self.x.atanh())
    }

    #[inline]
    fn floor(self) -> Self {
        Self::new(self.x.floor())
    }

    #[inline]
    fn ceil(self) -> Self {
        Self::new(self.x.ceil())
    }

    #[inline]
    fn round(self) -> Self {
        Self::new(self.x.round())
    }

    #[inline]
    fn trunc(self) -> Self {
        Self::new(self.x.trunc())
    }

    #[inline]
    fn fract(self) -> Self {
        Self::new(self.x.fract())
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.x.is_nan()
    }

    #[inline]
    fn is_infinite(self) -> bool {
        self.x.is_infinite()
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.x.is_finite()
    }

    #[inline]
    fn is_subnormal(self) -> bool {
        self.x.is_subnormal()
    }

    #[inline]
    fn is_normal(self) -> bool {
        self.x.is_normal()
    }

    #[inline]
    fn classify(self) -> std::num::FpCategory {
        self.x.classify()
    }

    #[inline]
    fn is_sign_positive(self) -> bool {
        self.x.is_sign_positive()
    }

    #[inline]
    fn is_sign_negative(self) -> bool {
        self.x.is_sign_negative()
    }

    #[inline]
    fn next_up(self) -> Self {
        Self::new(self.x.next_up())
    }

    #[inline]
    fn next_down(self) -> Self {
        Self::new(self.x.next_down())
    }

    #[inline]
    fn epsilon() -> Self {
        Self::new(f64::EPSILON)
    }

    #[inline]
    fn bits() -> usize {
        64
    }

    #[inline]
    fn min_positive() -> Self {
        Self::new(f64::MIN_POSITIVE)
    }

    #[inline]
    fn max_positive() -> Self {
        Self::new(f64::MAX)
    }

    #[inline]
    fn min_value() -> Option<Self> {
        Some(Self::new(f64::MIN))
    }

    #[inline]
    fn max_value() -> Option<Self> {
        Some(Self::new(f64::MAX))
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        Self::new(self.x.copysign(sign.x))
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x))
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        Self::new(self.x.clamp(min.x, max.x))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self::new(self.x.atan2(other.x))
    }

    #[inline]
    fn as_slice(&self) -> &[f64] {
        std::slice::from_ref(&self.x)
    }
}
