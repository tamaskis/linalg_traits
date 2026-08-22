use crate::my_scalar::MyScalar;
use linalg_traits::real_field::RealFieldBase;
use linalg_traits::verify_trait_implemented;

const _: bool = verify_trait_implemented!(MyScalar: RealFieldBase);

impl RealFieldBase for MyScalar {
    #[inline]
    fn _nan() -> Self {
        Self::new(f64::NAN)
    }

    #[inline]
    fn _infinity() -> Self {
        Self::new(f64::INFINITY)
    }

    #[inline]
    fn _abs(self) -> Self {
        Self::new(self.x.abs())
    }

    #[inline]
    fn _hypot(self, other: Self) -> Self {
        Self::new(self.x.hypot(other.x))
    }

    #[inline]
    fn _recip(self) -> Self {
        Self::new(self.x.recip())
    }

    #[inline]
    fn _mul_add(self, a: Self, b: Self) -> Self {
        Self::new(self.x.mul_add(a.x, b.x))
    }

    #[inline]
    fn _sqrt(self) -> Self {
        Self::new(self.x.sqrt())
    }

    #[inline]
    fn _cbrt(self) -> Self {
        Self::new(self.x.cbrt())
    }

    #[inline]
    fn _powi(self, n: i32) -> Self {
        Self::new(self.x.powi(n))
    }

    #[inline]
    fn _powf(self, n: Self) -> Self {
        Self::new(self.x.powf(n.x))
    }

    #[inline]
    fn _exp(self) -> Self {
        Self::new(self.x.exp())
    }

    #[inline]
    fn _exp2(self) -> Self {
        Self::new(self.x.exp2())
    }

    #[inline]
    fn _exp_m1(self) -> Self {
        Self::new(self.x.exp_m1())
    }

    #[inline]
    fn _ln(self) -> Self {
        Self::new(self.x.ln())
    }

    #[inline]
    fn _ln_1p(self) -> Self {
        Self::new(self.x.ln_1p())
    }

    #[inline]
    fn _log(self, base: Self) -> Self {
        Self::new(self.x.log(base.x))
    }

    #[inline]
    fn _log2(self) -> Self {
        Self::new(self.x.log2())
    }

    #[inline]
    fn _log10(self) -> Self {
        Self::new(self.x.log10())
    }

    #[inline]
    fn _sin(self) -> Self {
        Self::new(self.x.sin())
    }

    #[inline]
    fn _cos(self) -> Self {
        Self::new(self.x.cos())
    }

    #[inline]
    fn _sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.x.sin_cos();
        (Self::new(sin), Self::new(cos))
    }

    #[inline]
    fn _tan(self) -> Self {
        Self::new(self.x.tan())
    }

    #[inline]
    fn _asin(self) -> Self {
        Self::new(self.x.asin())
    }

    #[inline]
    fn _acos(self) -> Self {
        Self::new(self.x.acos())
    }

    #[inline]
    fn _atan(self) -> Self {
        Self::new(self.x.atan())
    }

    #[inline]
    fn _sinh(self) -> Self {
        Self::new(self.x.sinh())
    }

    #[inline]
    fn _cosh(self) -> Self {
        Self::new(self.x.cosh())
    }

    #[inline]
    fn _tanh(self) -> Self {
        Self::new(self.x.tanh())
    }

    #[inline]
    fn _asinh(self) -> Self {
        Self::new(self.x.asinh())
    }

    #[inline]
    fn _acosh(self) -> Self {
        Self::new(self.x.acosh())
    }

    #[inline]
    fn _atanh(self) -> Self {
        Self::new(self.x.atanh())
    }

    #[inline]
    fn _floor(self) -> Self {
        Self::new(self.x.floor())
    }

    #[inline]
    fn _ceil(self) -> Self {
        Self::new(self.x.ceil())
    }

    #[inline]
    fn _round(self) -> Self {
        Self::new(self.x.round())
    }

    #[inline]
    fn _trunc(self) -> Self {
        Self::new(self.x.trunc())
    }

    #[inline]
    fn _fract(self) -> Self {
        Self::new(self.x.fract())
    }

    #[inline]
    fn _is_nan(self) -> bool {
        self.x.is_nan()
    }

    #[inline]
    fn _is_infinite(self) -> bool {
        self.x.is_infinite()
    }

    #[inline]
    fn _is_finite(self) -> bool {
        self.x.is_finite()
    }

    #[inline]
    fn _is_subnormal(self) -> bool {
        self.x.is_subnormal()
    }

    #[inline]
    fn _is_normal(self) -> bool {
        self.x.is_normal()
    }

    #[inline]
    fn _classify(self) -> std::num::FpCategory {
        self.x.classify()
    }

    #[inline]
    fn _is_sign_positive(self) -> bool {
        self.x.is_sign_positive()
    }

    #[inline]
    fn _is_sign_negative(self) -> bool {
        self.x.is_sign_negative()
    }

    #[inline]
    fn _next_up(self) -> Self {
        Self::new(self.x.next_up())
    }

    #[inline]
    fn _next_down(self) -> Self {
        Self::new(self.x.next_down())
    }

    #[inline]
    fn _epsilon() -> Self {
        Self::new(f64::EPSILON)
    }

    #[inline]
    fn _bits() -> usize {
        64
    }

    #[inline]
    fn _min_positive() -> Self {
        Self::new(f64::MIN_POSITIVE)
    }

    #[inline]
    fn _max_positive() -> Self {
        Self::new(f64::MAX)
    }

    #[inline]
    fn _min_value() -> Option<Self> {
        Some(Self::new(f64::MIN))
    }

    #[inline]
    fn _max_value() -> Option<Self> {
        Some(Self::new(f64::MAX))
    }

    #[inline]
    fn _copysign(self, sign: Self) -> Self {
        Self::new(self.x.copysign(sign.x))
    }

    #[inline]
    fn _min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x))
    }

    #[inline]
    fn _max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x))
    }

    #[inline]
    fn _clamp(self, min: Self, max: Self) -> Self {
        Self::new(self.x.clamp(min.x, max.x))
    }

    #[inline]
    fn _atan2(self, other: Self) -> Self {
        Self::new(self.x.atan2(other.x))
    }

    #[inline]
    fn _as_slice(&self) -> &[f64] {
        std::slice::from_ref(&self.x)
    }
}
