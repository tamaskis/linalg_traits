#![allow(clippy::used_underscore_items)]

use crate::real_field::real_field_base::RealFieldBase;
use crate::real_field::real_field_faer::RealFieldFaer;
use crate::real_field::real_field_nalgebra::RealFieldNalgebra;
use crate::real_field::real_field_ndarray::RealFieldNdarray;
use crate::real_field::real_field_operations::real_field_operations::RealFieldOperations;

// TODO: test that all simd methods behave identically to corresponding non-simd methods

/// Trait defining a generic real number type.
///
/// # Overview
///
/// This trait defines the core functionality expected from any real number type. Please refer to
/// the [documentation in the `linalg-traits` book](https://tamaskis.github.io/linalg_traits/real_field.html)
/// for:
///
/// * a background on what this trait is trying to achieve
/// * a full description of the operators and methods provided by this trait
/// * A discussion on the interoperability with [`f64`]s
/// * How to implement this trait for a custom real number type
/// * How to test implementations of this trait
pub trait RealField:
    RealFieldBase + RealFieldOperations + RealFieldFaer + RealFieldNalgebra + RealFieldNdarray
{
    // -----
    // Zero.
    // -----

    /// Returns the additive identity element of `Self`, `0`.
    #[cfg(not(any(feature = "nalgebra", feature = "faer", feature = "ndarray")))]
    #[must_use]
    #[inline]
    fn zero() -> Self {
        Self::_zero()
    }

    /// Returns `true` if `self` is equal to the additive identity.
    #[cfg(not(any(feature = "nalgebra", feature = "faer", feature = "ndarray")))]
    #[inline]
    fn is_zero(&self) -> bool {
        self._is_zero()
    }

    /// Sets self to the additive identity element of `Self`, `0`.
    #[cfg(not(any(feature = "nalgebra", feature = "faer", feature = "ndarray")))]
    #[inline]
    fn set_zero(&mut self) {
        self._set_zero();
    }

    // ----
    // One.
    // ----

    /// Returns the multiplicative identity element of `Self`, `1`.
    #[cfg(not(any(feature = "nalgebra", feature = "faer", feature = "ndarray")))]
    #[must_use]
    #[inline]
    fn one() -> Self {
        Self::_one()
    }

    /// Returns `true` if `self` is equal to the multiplicative identity.
    #[cfg(not(any(feature = "nalgebra", feature = "faer", feature = "ndarray")))]
    #[inline]
    fn is_one(&self) -> bool {
        self._is_one()
    }

    /// Sets self to the multiplicative identity element of `Self`, `1`.
    #[cfg(not(any(feature = "nalgebra", feature = "faer", feature = "ndarray")))]
    #[inline]
    fn set_one(&mut self) {
        self._set_one();
    }

    // ----------
    // Constants.
    // ----------

    /// Euler's number (e).
    ///
    /// See [`RealFieldBase::_e`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn e() -> Self {
        Self::_e()
    }

    /// `π`
    ///
    /// See [`RealFieldBase::_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn pi() -> Self {
        Self::_pi()
    }

    /// The full circle constant (`τ`).
    ///
    /// Equal to `2π`.
    ///
    /// See [`RealFieldBase::_tau`].
    #[must_use]
    #[inline]
    fn tau() -> Self {
        Self::_tau()
    }

    /// `2π`
    ///
    /// See [`RealFieldBase::_two_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn two_pi() -> Self {
        Self::_two_pi()
    }

    /// The Euler-Mascheroni constant (γ).
    ///
    /// See [`RealFieldBase::_euler_gamma`].
    #[must_use]
    #[inline]
    fn euler_gamma() -> Self {
        Self::_euler_gamma()
    }

    /// The golden ratio (φ).
    ///
    /// See [`RealFieldBase::_golden_ratio`].
    #[must_use]
    #[inline]
    fn golden_ratio() -> Self {
        Self::_golden_ratio()
    }

    /// `1/π`
    ///
    /// See [`RealFieldBase::_frac_1_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_1_pi() -> Self {
        Self::_frac_1_pi()
    }

    /// `2/π`
    ///
    /// See [`RealFieldBase::_frac_2_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_2_pi() -> Self {
        Self::_frac_2_pi()
    }

    /// `π/2`
    ///
    /// See [`RealFieldBase::_frac_pi_2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_2() -> Self {
        Self::_frac_pi_2()
    }

    /// `π/3`
    ///
    /// See [`RealFieldBase::_frac_pi_3`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_3() -> Self {
        Self::_frac_pi_3()
    }

    /// `π/4`
    ///
    /// See [`RealFieldBase::_frac_pi_4`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_4() -> Self {
        Self::_frac_pi_4()
    }

    /// `π/6`
    ///
    /// See [`RealFieldBase::_frac_pi_6`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_6() -> Self {
        Self::_frac_pi_6()
    }

    /// `π/8`
    ///
    /// See [`RealFieldBase::_frac_pi_8`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_8() -> Self {
        Self::_frac_pi_8()
    }

    /// `2/√π`
    ///
    /// See [`RealFieldBase::_frac_2_sqrt_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self::_frac_2_sqrt_pi()
    }

    /// `1/√(π)`
    ///
    /// See [`RealFieldBase::_frac_1_sqrt_pi`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_pi() -> Self {
        Self::_frac_1_sqrt_pi()
    }

    /// `ln(2)`
    ///
    /// See [`RealFieldBase::_ln_2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ln_2() -> Self {
        Self::_ln_2()
    }

    /// `ln(10)`
    ///
    /// See [`RealFieldBase::_ln_10`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ln_10() -> Self {
        Self::_ln_10()
    }

    /// `log₂(e)`
    ///
    /// See [`RealFieldBase::_log2_e`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log2_e() -> Self {
        Self::_log2_e()
    }

    /// `log₁₀(e)`
    ///
    /// See [`RealFieldBase::_log10_e`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log10_e() -> Self {
        Self::_log10_e()
    }

    /// `log₂(10)`
    ///
    /// See [`RealFieldBase::_log2_10`].
    #[must_use]
    #[inline]
    fn log2_10() -> Self {
        Self::_log2_10()
    }

    /// `log₁₀(2)`
    ///
    /// See [`RealFieldBase::_log10_2`].
    #[must_use]
    #[inline]
    fn log10_2() -> Self {
        Self::_log10_2()
    }

    /// `√(2)`
    ///
    /// See [`RealFieldBase::_sqrt_2`].
    #[must_use]
    #[inline]
    fn sqrt_2() -> Self {
        Self::_sqrt_2()
    }

    /// `√(3)`
    ///
    /// See [`RealFieldBase::_sqrt_3`].
    #[must_use]
    #[inline]
    fn sqrt_3() -> Self {
        Self::_sqrt_3()
    }

    /// `√(5)`
    ///
    /// See [`RealFieldBase::_sqrt_5`].
    #[must_use]
    #[inline]
    fn sqrt_5() -> Self {
        Self::_sqrt_5()
    }

    /// `1/√(2)`
    ///
    /// See [`RealFieldBase::_frac_1_sqrt_2`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_2() -> Self {
        Self::_frac_1_sqrt_2()
    }

    /// `1/√(3)`
    ///
    /// See [`RealFieldBase::_frac_1_sqrt_3`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_3() -> Self {
        Self::_frac_1_sqrt_3()
    }

    /// `1/√(5)`
    ///
    /// See [`RealFieldBase::_frac_1_sqrt_5`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_5() -> Self {
        Self::_frac_1_sqrt_5()
    }

    /// `1/√(2π)`
    ///
    /// See [`RealFieldBase::_frac_1_sqrt_2pi`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_2pi() -> Self {
        Self::_frac_1_sqrt_2pi()
    }

    // ----------
    // Magnitude.
    // ----------

    /// Computes the absolute value of `self`.
    ///
    /// See [`RealFieldBase::_abs`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn abs(&self) -> Self {
        self._abs()
    }

    /// Compute the distance between the origin and a point `(x, y)` on the Euclidean plane.
    /// Equivalently, compute the length of the hypotenuse of a right-angle triangle with other
    /// sides having length `x._abs()` and `y._abs()`.
    ///
    /// See [`RealFieldBase::_hypot`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn hypot(self, other: Self) -> Self {
        self._hypot(other)
    }

    // ---------------------
    // Scaling / arithmetic.
    // ---------------------

    /// Multiplies this number by `factor`.
    ///
    /// See [`RealFieldBase::_scale`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn scale(self, factor: Self) -> Self {
        self._scale(factor)
    }

    /// Divides this number by `factor`.
    ///
    /// See [`RealFieldBase::_unscale`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn unscale(self, factor: Self) -> Self {
        self._unscale(factor)
    }

    /// Take the reciprocal (inverse) of a number, `1/x`.
    ///
    /// See [`RealFieldBase::_recip`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn recip(self) -> Self {
        self._recip()
    }

    /// Fused multiply-add. `Computes (self * a) + b`.
    ///
    /// See [`RealFieldBase::_mul_add`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self._mul_add(a, b)
    }

    // ------
    // Roots.
    // ------

    /// Returns the square root of a number.
    ///
    /// See [`RealFieldBase::_sqrt`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn sqrt(self) -> Self {
        self._sqrt()
    }

    /// Take the square root of a number, returning None if the number is negative.
    ///
    /// See [`RealFieldBase::_try_sqrt`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn try_sqrt(self) -> Option<Self> {
        self._try_sqrt()
    }

    /// Returns the cube root of a number.
    ///
    /// See [`RealFieldBase::_cbrt`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn cbrt(self) -> Self {
        self._cbrt()
    }

    // -------
    // Powers.
    // -------

    /// Raises a number to an integer power.
    ///
    /// See [`RealFieldBase::_powi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn powi(self, n: i32) -> Self {
        self._powi(n)
    }

    /// Raises a number to a floating-point power.
    ///
    /// See [`RealFieldBase::_powf`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn powf(self, n: Self) -> Self {
        self._powf(n)
    }

    // --------------------------
    // Exponential / logarithmic.
    // --------------------------

    /// Returns `e^(self)`, (the exponential function).
    ///
    /// See [`RealFieldBase::_exp`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn exp(self) -> Self {
        self._exp()
    }

    /// Returns `2^(self)`.
    ///
    /// See [`RealFieldBase::_exp2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn exp2(self) -> Self {
        self._exp2()
    }

    /// Returns `e^(self) - 1`.
    ///
    /// See [`RealFieldBase::_exp_m1`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn exp_m1(self) -> Self {
        self._exp_m1()
    }

    /// Returns the natural logarithm of the number.
    ///
    /// See [`RealFieldBase::_ln`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ln(self) -> Self {
        self._ln()
    }

    /// Returns `ln(1+n)` (natural logarithm)
    ///
    /// See [`RealFieldBase::_ln_1p`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ln_1p(self) -> Self {
        self._ln_1p()
    }

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// See [`RealFieldBase::_log`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log(self, base: Self) -> Self {
        self._log(base)
    }

    /// Returns the base 2 logarithm of the number.
    ///
    /// See [`RealFieldBase::_log2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log2(self) -> Self {
        self._log2()
    }

    /// Returns the base 10 logarithm of the number.
    ///
    /// See [`RealFieldBase::_log10`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log10(self) -> Self {
        self._log10()
    }

    // ------------------
    // Angle conversions.
    // ------------------

    /// Converts radians to degrees.
    ///
    /// See [`RealFieldBase::_to_degrees`].
    #[must_use]
    #[inline]
    fn to_degrees(self) -> Self {
        self._to_degrees()
    }

    /// Converts degrees to radians.
    ///
    /// See [`RealFieldBase::_to_radians`].
    #[must_use]
    #[inline]
    fn to_radians(self) -> Self {
        self._to_radians()
    }

    // ------------------------
    // Trigonometric functions.
    // ------------------------

    /// Computes the sine of a number (in radians).
    ///
    /// See [`RealFieldBase::_sin`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn sin(self) -> Self {
        self._sin()
    }

    /// Computes the cosine of a number (in radians).
    ///
    /// See [`RealFieldBase::_cos`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn cos(self) -> Self {
        self._cos()
    }

    /// Simultaneously computes the sine and cosine of the number, `x` (in radians). Returns
    /// `(sin(x), cos(x))`.
    ///
    /// See [`RealFieldBase::_sin_cos`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        self._sin_cos()
    }

    /// Computes the tangent of a number (in radians).
    ///
    /// See [`RealFieldBase::_tan`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn tan(self) -> Self {
        self._tan()
    }

    /// Computes the cosecant of a number (in radians).
    ///
    /// See [`RealFieldBase::_csc`].
    #[must_use]
    #[inline]
    fn csc(self) -> Self {
        self._csc()
    }

    /// Computes the secant of a number (in radians).
    ///
    /// See [`RealFieldBase::_sec`].
    #[must_use]
    #[inline]
    fn sec(self) -> Self {
        self._sec()
    }

    /// Computes the cotangent of a number (in radians).
    ///
    /// See [`RealFieldBase::_cot`].
    #[must_use]
    #[inline]
    fn cot(self) -> Self {
        self._cot()
    }

    /// Computes the arcsine of a number (return value is in radians).
    ///
    /// See [`RealFieldBase::_asin`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn asin(self) -> Self {
        self._asin()
    }

    /// Computes the arccosine of a number (return value is in radians).
    ///
    /// See [`RealFieldBase::_acos`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn acos(self) -> Self {
        self._acos()
    }

    /// Computes the arctangent of a number (return value is in radians).
    ///
    /// See [`RealFieldBase::_atan`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn atan(self) -> Self {
        self._atan()
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in radians.
    ///
    /// See [`RealFieldBase::_atan2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn atan2(self, other: Self) -> Self {
        self._atan2(other)
    }

    /// Computes the arccosecant of a number (return value is in radians).
    ///
    /// See [`RealFieldBase::_acsc`].
    #[must_use]
    #[inline]
    fn acsc(self) -> Self {
        self._acsc()
    }

    /// Computes the arcsecant of a number (return value is in radians).
    ///
    /// See [`RealFieldBase::_asec`].
    #[must_use]
    #[inline]
    fn asec(self) -> Self {
        self._asec()
    }

    /// Computes the arccotangent of a number (return value is in radians).
    ///
    /// See [`RealFieldBase::_acot`].
    #[must_use]
    #[inline]
    fn acot(self) -> Self {
        self._acot()
    }

    /// Computes the sine of a number (in degrees).
    ///
    /// See [`RealFieldBase::_sind`].
    #[must_use]
    #[inline]
    fn sind(self) -> Self {
        self._sind()
    }

    /// Computes the cosine of a number (in degrees).
    ///
    /// See [`RealFieldBase::_cosd`].
    #[must_use]
    #[inline]
    fn cosd(self) -> Self {
        self._cosd()
    }

    /// Simultaneously computes the sine and cosine of a number (in degrees).
    ///
    /// See [`RealFieldBase::_sind_cosd`].
    #[must_use]
    #[inline]
    fn sind_cosd(self) -> (Self, Self) {
        self._sind_cosd()
    }

    /// Computes the tangent of a number (in degrees).
    ///
    /// See [`RealFieldBase::_tand`].
    #[must_use]
    #[inline]
    fn tand(self) -> Self {
        self._tand()
    }

    /// Computes the cosecant of a number (in degrees).
    ///
    /// See [`RealFieldBase::_cscd`].
    #[must_use]
    #[inline]
    fn cscd(self) -> Self {
        self._cscd()
    }

    /// Computes the secant of a number (in degrees).
    ///
    /// See [`RealFieldBase::_secd`].
    #[must_use]
    #[inline]
    fn secd(self) -> Self {
        self._secd()
    }

    /// Computes the cotangent of a number (in degrees).
    ///
    /// See [`RealFieldBase::_cotd`].
    #[must_use]
    #[inline]
    fn cotd(self) -> Self {
        self._cotd()
    }

    /// Computes the arcsine of a number (in degrees).
    ///
    /// See [`RealFieldBase::_asind`].
    #[must_use]
    #[inline]
    fn asind(self) -> Self {
        self._asind()
    }

    /// Computes the arccosine of a number (in degrees).
    ///
    /// See [`RealFieldBase::_acosd`].
    #[must_use]
    #[inline]
    fn acosd(self) -> Self {
        self._acosd()
    }

    /// Computes the arctangent of a number (in degrees).
    ///
    /// See [`RealFieldBase::_atand`].
    #[must_use]
    #[inline]
    fn atand(self) -> Self {
        self._atand()
    }

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in degrees.
    ///
    /// See [`RealFieldBase::_atan2d`].
    #[must_use]
    #[inline]
    fn atan2d(self, other: Self) -> Self {
        self._atan2d(other)
    }

    /// Computes the arccosecant of a number (in degrees).
    ///
    /// See [`RealFieldBase::_acscd`].
    #[must_use]
    #[inline]
    fn acscd(self) -> Self {
        self._acscd()
    }

    /// Computes the arcsecant of a number (in degrees).
    ///
    /// See [`RealFieldBase::_asecd`].
    #[must_use]
    #[inline]
    fn asecd(self) -> Self {
        self._asecd()
    }

    /// Computes the arccotangent of a number (in degrees).
    ///
    /// See [`RealFieldBase::_acotd`].
    #[must_use]
    #[inline]
    fn acotd(self) -> Self {
        self._acotd()
    }

    // ---------------------
    // Hyperbolic functions.
    // ---------------------

    /// Hyperbolic sine function.
    ///
    /// See [`RealFieldBase::_sinh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn sinh(self) -> Self {
        self._sinh()
    }

    /// Hyperbolic cosine function.
    ///
    /// See [`RealFieldBase::_cosh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn cosh(self) -> Self {
        self._cosh()
    }

    /// Simultaneously computes the hyperbolic sine and hyperbolic cosine of the number, `x`.
    /// Returns `(sinh(x), cosh(x))`.
    ///
    /// See [`RealFieldBase::_sinh_cosh`].
    #[cfg(not(feature = "nalgebra"))]
    #[inline]
    fn sinh_cosh(self) -> (Self, Self) {
        self._sinh_cosh()
    }

    /// Hyperbolic tangent function.
    ///
    /// See [`RealFieldBase::_tanh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn tanh(self) -> Self {
        self._tanh()
    }

    /// Hyperbolic cosecant function.
    ///
    /// See [`RealFieldBase::_csch`].
    #[must_use]
    #[inline]
    fn csch(self) -> Self {
        self._csch()
    }

    /// Hyperbolic secant function.
    ///
    /// See [`RealFieldBase::_sech`].
    #[must_use]
    #[inline]
    fn sech(self) -> Self {
        self._sech()
    }

    /// Hyperbolic cotangent function.
    ///
    /// See [`RealFieldBase::_coth`].
    #[must_use]
    #[inline]
    fn coth(self) -> Self {
        self._coth()
    }

    /// Inverse hyperbolic sine function.
    ///
    /// See [`RealFieldBase::_asinh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn asinh(self) -> Self {
        self._asinh()
    }

    /// Inverse hyperbolic cosine function.
    ///
    /// See [`RealFieldBase::_acosh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn acosh(self) -> Self {
        self._acosh()
    }

    /// Inverse hyperbolic tangent function.
    ///
    /// See [`RealFieldBase::_atanh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn atanh(self) -> Self {
        self._atanh()
    }

    /// Inverse hyperbolic cosecant function.
    ///
    /// See [`RealFieldBase::_acsch`].
    #[must_use]
    #[inline]
    fn acsch(self) -> Self {
        self._acsch()
    }

    /// Inverse hyperbolic secant function.
    ///
    /// See [`RealFieldBase::_asech`].
    #[must_use]
    #[inline]
    fn asech(self) -> Self {
        self._asech()
    }

    /// Inverse hyperbolic cotangent function.
    ///
    /// See [`RealFieldBase::_acoth`].
    #[must_use]
    #[inline]
    fn acoth(self) -> Self {
        self._acoth()
    }

    // ---------
    // Rounding.
    // ---------

    /// Returns the largest integer that is less than or equal to `self`.
    ///
    /// See [`RealFieldBase::_floor`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn floor(self) -> Self {
        self._floor()
    }

    /// Returns the smallest integer that is greater than or equal to `self`.
    ///
    /// See [`RealFieldBase::_ceil`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ceil(self) -> Self {
        self._ceil()
    }

    /// Returns the nearest integer to `self`. If a value is half-way between two integers, round
    /// away from `0.0`.
    ///
    /// See [`RealFieldBase::_round`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn round(self) -> Self {
        self._round()
    }

    /// Returns the integer part of `self`. This means that non-integer numbers are always truncated
    /// towards zero.
    ///
    /// See [`RealFieldBase::_trunc`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn trunc(self) -> Self {
        self._trunc()
    }

    /// Returns the fractional part of `self`.
    ///
    /// See [`RealFieldBase::_fract`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn fract(self) -> Self {
        self._fract()
    }

    // ------------------------------
    // Magnitude-specific operations.
    // ------------------------------

    /// Returns a number composed of the magnitude of `self` and the sign of `sign`.
    ///
    /// See [`RealFieldBase::_copysign`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        self._copysign(sign)
    }

    /// Returns the minimum of the two numbers, ignoring NaN.
    ///
    /// See [`RealFieldBase::_min`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn min(self, other: Self) -> Self {
        self._min(other)
    }

    /// Returns the maximum of the two numbers, ignoring NaN.
    ///
    /// See [`RealFieldBase::_max`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn max(self, other: Self) -> Self {
        self._max(other)
    }

    /// Restrict a value to a certain interval unless it is NaN.
    ///
    /// Returns `max` if `self` is greater than `max`, and `min` if `self` is less than `min`.
    /// Otherwise this returns `self`.
    ///
    /// Note that this function returns NaN if the initial value was NaN as well. If the result is
    /// zero and among the three inputs `self`, `min`, and `max` there are zeros with different
    /// sign, either 0.0 or -0.0 is returned non-deterministically.
    ///
    /// # Panics
    ///
    /// Panics if `min` > `max`, `min` is NaN, or `max` is NaN.
    ///
    /// See [`RealFieldBase::_clamp`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        self._clamp(min, max)
    }

    // -----------------------------------
    // Checking floating point properties.
    // -----------------------------------

    /// Returns `true` if the number is NaN.
    ///
    /// See [`RealFieldBase::_is_nan`].
    #[inline]
    fn is_nan(self) -> bool {
        self._is_nan()
    }

    /// Returns `true` if this number is infinite.
    ///
    /// See [`RealFieldBase::_is_infinite`].
    #[inline]
    fn is_infinite(self) -> bool {
        self._is_infinite()
    }

    /// Returns `true` if this number is neither infinite nor `NaN`.
    ///
    /// See [`RealFieldBase::_is_finite`].
    #[cfg(not(feature = "nalgebra"))]
    #[inline]
    fn is_finite(&self) -> bool {
        self._is_finite()
    }

    /// Returns `true` if this number is
    /// [subnormal](https://en.wikipedia.org/wiki/Subnormal_number).
    ///
    /// See [`RealFieldBase::_is_subnormal`].
    #[inline]
    fn is_subnormal(self) -> bool {
        self._is_subnormal()
    }

    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal](https://en.wikipedia.org/wiki/Subnormal_number), or NaN.
    ///
    /// See [`RealFieldBase::_is_normal`].
    #[inline]
    fn is_normal(self) -> bool {
        self._is_normal()
    }

    /// Returns the floating point category of the number.
    ///
    /// See [`RealFieldBase::_classify`].
    #[inline]
    fn classify(self) -> std::num::FpCategory {
        self._classify()
    }

    /// Returns `true` if `self` has a positive sign.
    ///
    /// See [`RealFieldBase::_is_sign_positive`].
    #[cfg(not(feature = "nalgebra"))]
    #[inline]
    fn is_sign_positive(&self) -> bool {
        self._is_sign_positive()
    }

    /// Returns `true` if `self` has a negative sign.
    ///
    /// See [`RealFieldBase::_is_sign_negative`].
    #[cfg(not(feature = "nalgebra"))]
    #[inline]
    fn is_sign_negative(&self) -> bool {
        self._is_sign_negative()
    }

    /// Returns the least number greater than `self`.
    ///
    /// See [`RealFieldBase::_next_up`].
    #[must_use]
    #[inline]
    fn next_up(self) -> Self {
        self._next_up()
    }

    /// Returns the greatest number less than `self`.
    ///
    /// See [`RealFieldBase::_next_down`].
    #[must_use]
    #[inline]
    fn next_down(self) -> Self {
        self._next_down()
    }

    // --------------------------
    // Floating point properties.
    // --------------------------

    /// Returns epsilon, a small positive value.
    ///
    /// For types like [`f64`], this is the
    /// [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon).
    ///
    /// See [`RealFieldBase::_epsilon`].
    #[must_use]
    #[inline]
    fn epsilon() -> Self {
        Self::_epsilon()
    }

    /// The size of this type in bits.
    ///
    /// See [`RealFieldBase::_bits`].
    #[must_use]
    #[inline]
    fn bits() -> usize {
        Self::_bits()
    }

    /// Smallest positive normal value.
    ///
    /// See [`RealFieldBase::_min_positive`].
    #[must_use]
    #[inline]
    fn min_positive() -> Self {
        Self::_min_positive()
    }

    /// Largest positive finite value.
    ///
    /// See [`RealFieldBase::_max_positive`].
    #[must_use]
    #[inline]
    fn max_positive() -> Self {
        Self::_max_positive()
    }

    /// Square root of the smallest positive normal value.
    ///
    /// See [`RealFieldBase::_sqrt_min_positive`].
    #[must_use]
    #[inline]
    fn sqrt_min_positive() -> Self {
        Self::_sqrt_min_positive()
    }

    /// Square root of the largest positive finite value.
    ///
    /// See [`RealFieldBase::_sqrt_max_positive`].
    #[must_use]
    #[inline]
    fn sqrt_max_positive() -> Self {
        Self::_sqrt_max_positive()
    }

    /// The smallest finite negative (i.e. "most negative") value representable using this type.
    ///
    /// # Warning
    ///
    /// Ref. \[1\] incorrectly documents this as "the smallest finite positive value representable
    /// using this type". However, from a simple test, it is clear that they mean "the smallest
    /// negative finite value representable using this type".
    ///
    /// ```
    /// # #[cfg(feature = "nalgebra")]
    /// # {
    /// let x: f64 = nalgebra::RealField::min_value().unwrap();
    /// assert_eq!(x, -1.7976931348623157e308);
    /// # }
    /// ```
    ///
    /// See [`RealFieldBase::_min_value`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn min_value() -> Option<Self> {
        Self::_min_value()
    }

    /// The largest finite positive value representable using this type.
    ///
    /// See [`RealFieldBase::_max_value`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn max_value() -> Option<Self> {
        Self::_max_value()
    }

    /// Returns the `NaN` value.
    ///
    /// See [`RealFieldBase::_nan`].
    #[must_use]
    #[inline]
    fn nan() -> Self {
        Self::_nan()
    }

    /// Returns the infinite value.
    ///
    /// See [`RealFieldBase::_infinity`].
    #[must_use]
    #[inline]
    fn infinity() -> Self {
        Self::_infinity()
    }

    // ------------------
    // Interface methods.
    // ------------------

    /// Express this real field type as a slice of `f64`.
    ///
    /// See [`RealFieldBase::_as_slice`].
    #[inline]
    fn as_slice(&self) -> &[f64] {
        self._as_slice()
    }
}

// Blanket implementation.
impl<T> RealField for T where
    T: RealFieldBase + RealFieldOperations + RealFieldFaer + RealFieldNalgebra + RealFieldNdarray
{
}
