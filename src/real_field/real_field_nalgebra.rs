/// Additional requirements to add on top of [`crate::real_field::RealFieldBase`] to make a type
/// compatible with [`nalgebra`] when the `nalgebra` feature is enabled.
///
/// When the `nalgebra` feature is _NOT_ enabled, this trait is simply an empty marker trait.
///
/// # Blanket Implementations
///
/// * When the `nalgebra` feature is disabled, this trait is automatically implemented for any type
///   that implements [`crate::real_field::RealFieldBase`].
/// * When the `nalgebra` feature is enabled, this trait is automatically implemented for any type
///   that implements [`crate::real_field::RealFieldBase`] and [`nalgebra::RealField`].
///     * If a type already implements [`crate::real_field::RealFieldBase`], the
///       [`nalgebra::RealField`] trait can be implemented using
///       [`crate::impl_nalgebra_real_field`].
#[cfg(feature = "nalgebra")]
pub trait RealFieldNalgebra: nalgebra::RealField {}
#[allow(missing_docs)]
#[cfg(not(feature = "nalgebra"))]
pub trait RealFieldNalgebra {}

// Blanket implementations.
#[cfg(feature = "nalgebra")]
impl<T> RealFieldNalgebra for T where T: crate::real_field::RealFieldBase + nalgebra::RealField {}
#[allow(missing_docs)]
#[cfg(not(feature = "nalgebra"))]
impl<T> RealFieldNalgebra for T where T: crate::real_field::RealFieldBase {}

/// Implement the [`nalgebra::RealField`] trait for a type that has already implemented
/// [`crate::real_field::RealFieldBase`].
///
/// # Generic Arguments
///
/// * `$t` - The type for which the [`nalgebra::RealField`] trait is being implemented.
#[cfg(feature = "nalgebra")]
#[macro_export]
macro_rules! impl_nalgebra_real_field {
    ($t:ty) => {
        impl approx::AbsDiffEq for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Epsilon = Self;

            fn default_epsilon() -> Self::Epsilon {
                Self::default()
            }

            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                let self_slice: &[f64] = <$t as $crate::real_field::RealFieldBase>::_as_slice(self);
                let other_slice: &[f64] =
                    <$t as $crate::real_field::RealFieldBase>::_as_slice(other);
                self_slice
                    .iter()
                    .zip(other_slice)
                    .all(|(a, b)| a.abs_diff_eq(b, epsilon.into()))
            }
        }

        impl approx::RelativeEq for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            fn default_max_relative() -> Self::Epsilon {
                Self::default()
            }

            fn relative_eq(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                let self_slice: &[f64] = <$t as $crate::real_field::RealFieldBase>::_as_slice(self);
                let other_slice: &[f64] =
                    <$t as $crate::real_field::RealFieldBase>::_as_slice(other);
                self_slice
                    .iter()
                    .zip(other_slice)
                    .all(|(a, b)| a.relative_eq(b, epsilon.into(), max_relative.into()))
            }
        }

        impl approx::UlpsEq for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            fn default_max_ulps() -> u32 {
                f64::default_max_ulps()
            }

            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                let self_slice: &[f64] = <$t as $crate::real_field::RealFieldBase>::_as_slice(self);
                let other_slice: &[f64] =
                    <$t as $crate::real_field::RealFieldBase>::_as_slice(other);
                self_slice
                    .iter()
                    .zip(other_slice)
                    .all(|(a, b)| a.ulps_eq(b, epsilon.into(), max_ulps))
            }
        }

        impl simba::simd::SimdValue for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            const LANES: usize = 1;
            type Element = $t;
            type SimdBool = bool;

            #[inline(always)]
            fn splat(val: Self::Element) -> Self {
                val
            }

            #[inline(always)]
            fn extract(&self, _: usize) -> Self::Element {
                *self
            }

            #[inline(always)]
            unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
                *self
            }

            #[inline(always)]
            fn replace(&mut self, _: usize, val: Self::Element) {
                *self = val;
            }

            #[inline(always)]
            unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
                *self = val;
            }

            #[inline(always)]
            fn select(self, cond: Self::SimdBool, other: Self) -> Self {
                if cond { self } else { other }
            }
        }

        impl simba::scalar::SubsetOf<$t> for $t {
            #[inline]
            fn to_superset(&self) -> $t {
                *self
            }

            #[inline]
            fn from_superset_unchecked(element: &$t) -> $t {
                *element
            }

            #[inline]
            fn is_in_subset(_: &$t) -> bool {
                true
            }
        }

        impl simba::scalar::SubsetOf<$t> for f32 {
            #[inline]
            fn to_superset(&self) -> $t {
                <$t as From<f64>>::from((*self).into())
            }

            #[inline]
            #[allow(clippy::cast_possible_truncation)]
            fn from_superset_unchecked(element: &$t) -> f32 {
                <$t as Into<f64>>::into(*element) as f32
            }

            #[inline]
            fn is_in_subset(_: &$t) -> bool {
                true
            }
        }

        impl simba::scalar::SubsetOf<$t> for f64 {
            #[inline]
            fn to_superset(&self) -> $t {
                <$t as From<f64>>::from(*self)
            }

            #[inline]
            fn from_superset_unchecked(element: &$t) -> f64 {
                <$t as Into<f64>>::into(*element)
            }

            #[inline]
            fn is_in_subset(_: &$t) -> bool {
                true
            }
        }

        impl simba::scalar::Field for $t {}

        impl nalgebra::ComplexField for $t {
            type RealField = $t;

            #[inline]
            fn from_real(re: Self::RealField) -> Self {
                re
            }

            #[inline]
            fn real(self) -> Self::RealField {
                self
            }

            #[inline]
            fn imaginary(self) -> Self::RealField {
                <$t as num_traits::Zero>::zero()
            }

            #[inline]
            fn modulus(self) -> Self::RealField {
                <$t as $crate::real_field::RealFieldBase>::_abs(self)
            }

            #[inline]
            fn modulus_squared(self) -> Self::RealField {
                self * self // https://docs.rs/simba/0.10.0/src/simba/scalar/complex.rs.html#749
            }

            #[inline]
            fn argument(self) -> Self::RealField {
                <$t as $crate::real_field::RealFieldBase>::_atan2(
                    self,
                    <$t as num_traits::Zero>::zero(),
                )
            }

            #[inline]
            fn norm1(self) -> Self::RealField {
                <$t as $crate::real_field::RealFieldBase>::_abs(self) // https://docs.rs/simba/0.10.0/src/simba/scalar/complex.rs.html#768
            }

            #[inline]
            fn scale(self, factor: Self::RealField) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_scale(self, factor)
            }

            #[inline]
            fn unscale(self, factor: Self::RealField) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_unscale(self, factor)
            }

            #[inline]
            fn floor(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_floor(self)
            }

            #[inline]
            fn ceil(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_ceil(self)
            }

            #[inline]
            fn round(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_round(self)
            }

            #[inline]
            fn trunc(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_trunc(self)
            }

            #[inline]
            fn fract(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_fract(self)
            }

            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_mul_add(self, a, b)
            }

            #[inline]
            fn abs(self) -> Self::RealField {
                <$t as $crate::real_field::RealFieldBase>::_abs(self)
            }

            #[inline]
            fn hypot(self, other: Self) -> Self::RealField {
                <$t as $crate::real_field::RealFieldBase>::_hypot(self, other)
            }

            #[inline]
            fn recip(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_recip(self)
            }

            #[inline]
            fn conjugate(self) -> Self {
                self // https://docs.rs/simba/0.10.0/src/simba/scalar/complex.rs.html#806
            }

            #[inline]
            fn sin(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_sin(self)
            }

            #[inline]
            fn cos(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_cos(self)
            }

            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                <$t as $crate::real_field::RealFieldBase>::_sin_cos(self)
            }

            #[inline]
            fn tan(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_tan(self)
            }

            #[inline]
            fn asin(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_asin(self)
            }

            #[inline]
            fn acos(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_acos(self)
            }

            #[inline]
            fn atan(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_atan(self)
            }

            #[inline]
            fn sinh(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_sinh(self)
            }

            #[inline]
            fn cosh(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_cosh(self)
            }

            #[inline]
            fn tanh(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_tanh(self)
            }

            #[inline]
            fn asinh(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_asinh(self)
            }

            #[inline]
            fn acosh(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_acosh(self)
            }

            #[inline]
            fn atanh(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_atanh(self)
            }

            #[inline]
            fn log(self, base: Self::RealField) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_log(self, base)
            }

            #[inline]
            fn log2(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_log2(self)
            }

            #[inline]
            fn log10(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_log10(self)
            }

            #[inline]
            fn ln(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_ln(self)
            }

            #[inline]
            fn ln_1p(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_ln_1p(self)
            }

            #[inline]
            fn sqrt(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_sqrt(self)
            }

            #[inline]
            fn exp(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_exp(self)
            }

            #[inline]
            fn exp2(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_exp2(self)
            }

            #[inline]
            fn exp_m1(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_exp_m1(self)
            }

            #[inline]
            fn powi(self, n: i32) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_powi(self, n)
            }

            #[inline]
            fn powf(self, n: Self::RealField) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_powf(self, n)
            }

            #[inline]
            fn powc(self, n: Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_powf(self, n) // https://docs.rs/simba/0.10.0/src/simba/scalar/complex.rs.html#872
            }

            #[inline]
            fn cbrt(self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_cbrt(self)
            }

            #[inline]
            fn is_finite(&self) -> bool {
                <$t as $crate::real_field::RealFieldBase>::_is_finite(*self)
            }

            #[inline]
            fn try_sqrt(self) -> Option<Self> {
                <$t as $crate::real_field::RealFieldBase>::_try_sqrt(self)
            }
        }

        impl nalgebra::RealField for $t {
            #[inline]
            fn is_sign_positive(&self) -> bool {
                <$t as $crate::real_field::RealFieldBase>::_is_sign_positive(*self)
            }

            #[inline]
            fn is_sign_negative(&self) -> bool {
                <$t as $crate::real_field::RealFieldBase>::_is_sign_negative(*self)
            }

            #[inline]
            fn copysign(self, sign: Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_copysign(self, sign)
            }

            #[inline]
            fn max(self, other: Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_max(self, other)
            }

            #[inline]
            fn min(self, other: Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_min(self, other)
            }

            #[inline]
            fn clamp(self, min: Self, max: Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_clamp(self, min, max)
            }

            #[inline]
            fn atan2(self, other: Self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_atan2(self, other)
            }

            #[inline]
            fn min_value() -> Option<Self> {
                <$t as $crate::real_field::RealFieldBase>::_min_value()
            }

            #[inline]
            fn max_value() -> Option<Self> {
                <$t as $crate::real_field::RealFieldBase>::_max_value()
            }

            #[inline]
            fn pi() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_pi()
            }

            #[inline]
            fn two_pi() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_two_pi()
            }

            #[inline]
            fn frac_pi_2() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_frac_pi_2()
            }

            #[inline]
            fn frac_pi_3() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_frac_pi_3()
            }

            #[inline]
            fn frac_pi_4() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_frac_pi_4()
            }

            #[inline]
            fn frac_pi_6() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_frac_pi_6()
            }

            #[inline]
            fn frac_pi_8() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_frac_pi_8()
            }

            #[inline]
            fn frac_1_pi() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_frac_1_pi()
            }

            #[inline]
            fn frac_2_pi() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_frac_2_pi()
            }

            #[inline]
            fn frac_2_sqrt_pi() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_frac_2_sqrt_pi()
            }

            #[inline]
            fn e() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_e()
            }

            #[inline]
            fn log2_e() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_log2_e()
            }

            #[inline]
            fn log10_e() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_log10_e()
            }

            #[inline]
            fn ln_2() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_ln_2()
            }

            #[inline]
            fn ln_10() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_ln_10()
            }
        }
    };
}
