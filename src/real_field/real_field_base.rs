use crate::real_field::base::Base;
use crate::real_field_operations::real_field_operations::RealFieldOperations;
use num_traits::Num;

// TODO everything in this crate has to be unit tested, and we should check that in a generic
// context we don't have to disambiguate

/// Trait defining most functionality of a real number.
///
/// This is a separate trait because the fully-fledged [`crate::RealField`] trait includes
/// additional optional features that are conditionally compiled depending on which features (e.g.
/// `nalgebra`, `faer`, `ndarray`) are enabled.
///
/// # Note
///
/// We do not offer default implementations for any methods provided directly by [`f64`]. This is to
/// (a) ensure no difference between the behavior of this trait and the native [`f64`] methods, and
/// (b) avoid potential performance overhead from a default implementation that would be provided by
/// this trait.
pub trait RealFieldBase: Base + Num + RealFieldOperations + PartialOrd {
    // ----------
    // Constants.
    // ----------

    /// Euler's number (e).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.E.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.e>
    #[must_use]
    #[inline]
    fn _e() -> Self {
        Self::from(std::f64::consts::E)
    }

    /// The Euler-Mascheroni constant (γ).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.EULER_GAMMA.html>
    #[rustversion::since(1.94)]
    #[must_use]
    #[inline]
    fn _euler_gamma() -> Self {
        Self::from(std::f64::consts::EULER_GAMMA)
    }
    #[rustversion::before(1.94)]
    #[must_use]
    #[inline]
    #[allow(missing_docs)]
    fn _euler_gamma() -> Self {
        Self::from(0.5772156649015329)
    }

    /// `1/π`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_1_PI.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.frac_1_pi>
    #[must_use]
    #[inline]
    fn _frac_1_pi() -> Self {
        Self::from(std::f64::consts::FRAC_1_PI)
    }

    /// `1/√(2)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_1_SQRT_2.html>
    #[must_use]
    #[inline]
    fn _frac_1_sqrt_2() -> Self {
        Self::from(std::f64::consts::FRAC_1_SQRT_2)
    }

    /// `1/√(2π)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_1_SQRT_2PI.html>
    #[must_use]
    #[inline]
    #[allow(clippy::excessive_precision)]
    fn _frac_1_sqrt_2pi() -> Self {
        Self::from(0.398942280401432677939946059934381868) // `std:f64::const::FRAC_1_SQRT_2PI` is in `nightly`
    }

    /// `1/√(3)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_1_SQRT_3.html>
    #[must_use]
    #[inline]
    #[allow(clippy::excessive_precision)]
    fn _frac_1_sqrt_3() -> Self {
        Self::from(0.577350269189625764509148780501957456) // `std:f64::const::FRAC_1_SQRT_3` is in `nightly`
    }

    /// `1/√(5)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_1_SQRT_5.html>
    #[must_use]
    #[inline]
    #[allow(clippy::excessive_precision)]
    fn _frac_1_sqrt_5() -> Self {
        Self::from(0.44721359549995793928183473374625524) // `std:f64::const::FRAC_1_SQRT_5` is in `nightly`
    }

    /// `1/√(2π)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_1_SQRT_2PI.html>
    #[must_use]
    #[inline]
    #[allow(clippy::excessive_precision)]
    fn _frac_1_sqrt_pi() -> Self {
        Self::from(0.564189583547756286948079451560772586) // `std:f64::const::FRAC_1_SQRT_PI` is in `nightly`
    }

    /// `2/π`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_2_PI.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.frac_2_pi>
    #[must_use]
    #[inline]
    fn _frac_2_pi() -> Self {
        Self::from(std::f64::consts::FRAC_2_PI)
    }

    /// `2/√π`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_2_SQRT_PI.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.frac_2_sqrt_pi>
    #[must_use]
    #[inline]
    fn _frac_2_sqrt_pi() -> Self {
        Self::from(std::f64::consts::FRAC_2_SQRT_PI)
    }

    /// `π/2`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_PI_2.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.frac_pi_2>
    #[must_use]
    #[inline]
    fn _frac_pi_2() -> Self {
        Self::from(std::f64::consts::FRAC_PI_2)
    }

    /// `π/3`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_PI_3.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.frac_pi_3>
    #[must_use]
    #[inline]
    fn _frac_pi_3() -> Self {
        Self::from(std::f64::consts::FRAC_PI_3)
    }

    /// `π/4`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_PI_4.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.frac_pi_4>
    #[must_use]
    #[inline]
    fn _frac_pi_4() -> Self {
        Self::from(std::f64::consts::FRAC_PI_4)
    }

    /// `π/6`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_PI_6.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.frac_pi_6>
    #[must_use]
    #[inline]
    fn _frac_pi_6() -> Self {
        Self::from(std::f64::consts::FRAC_PI_6)
    }

    /// `π/8`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.FRAC_PI_8.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.frac_pi_8>
    #[must_use]
    #[inline]
    fn _frac_pi_8() -> Self {
        Self::from(std::f64::consts::FRAC_PI_8)
    }

    /// The golden ratio (φ).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.GOLDEN_RATIO.html>
    #[rustversion::since(1.94)]
    #[must_use]
    #[inline]
    fn _golden_ratio() -> Self {
        Self::from(std::f64::consts::GOLDEN_RATIO)
    }
    #[rustversion::before(1.94)]
    #[must_use]
    #[inline]
    #[allow(missing_docs, clippy::excessive_precision)]
    fn _golden_ratio() -> Self {
        Self::from(1.618033988749894848204586834365638118)
    }

    /// `ln(10)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.LN_10.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.ln_10>
    #[must_use]
    #[inline]
    fn _ln_10() -> Self {
        Self::from(std::f64::consts::LN_10)
    }

    /// `ln(2)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.LN_2.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.ln_2>
    #[must_use]
    #[inline]
    fn _ln_2() -> Self {
        Self::from(std::f64::consts::LN_2)
    }

    /// `log₁₀(2)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.LOG10_2.html>
    #[must_use]
    #[inline]
    fn _log10_2() -> Self {
        Self::from(std::f64::consts::LOG10_2)
    }

    /// `log₁₀(e)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.LOG10_E.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.log10_e>
    #[must_use]
    #[inline]
    fn _log10_e() -> Self {
        Self::from(std::f64::consts::LOG10_E)
    }

    /// `log₂(10)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.LOG2_10.html>
    #[must_use]
    #[inline]
    fn _log2_10() -> Self {
        Self::from(std::f64::consts::LOG2_10)
    }

    /// `log₂(e)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.LOG2_E.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.log2_e>
    #[must_use]
    #[inline]
    fn _log2_e() -> Self {
        Self::from(std::f64::consts::LOG2_E)
    }

    /// `π`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.PI.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.pi>
    #[must_use]
    #[inline]
    fn _pi() -> Self {
        Self::from(std::f64::consts::PI)
    }

    /// `√(2)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.SQRT_2.html>
    #[must_use]
    #[inline]
    fn _sqrt_2() -> Self {
        Self::from(std::f64::consts::SQRT_2)
    }

    /// `√(3)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.SQRT_3.html>
    #[must_use]
    #[inline]
    #[allow(clippy::excessive_precision)]
    fn _sqrt_3() -> Self {
        Self::from(1.732050807568877293527446341505872367) // `std:f64::const::SQRT_3` is in `nightly`
    }

    /// `√(5)`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.SQRT_5.html>
    #[must_use]
    #[inline]
    #[allow(clippy::excessive_precision)]
    fn _sqrt_5() -> Self {
        Self::from(2.23606797749978969640917366873127623) // `std:f64::const::SQRT_5` is in `nightly`
    }

    /// The full circle constant (`τ`).
    ///
    /// Equal to `2π`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.TAU.html>
    #[must_use]
    #[inline]
    fn _tau() -> Self {
        Self::from(std::f64::consts::TAU)
    }

    /// `2π`
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/f64/consts/constant.TAU.html>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.two_pi>
    #[must_use]
    #[inline]
    fn _two_pi() -> Self {
        Self::_tau()
    }

    // ----------
    // Magnitude.
    // ----------

    /// Computes the absolute value of `self`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.abs>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.abs>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.abs>
    /// * \[4\] <https://docs.rs/faer-traits/latest/faer_traits/trait.ComplexField.html#tymethod.abs_impl>
    #[must_use]
    fn _abs(self) -> Self;

    /// Compute the distance between the origin and a point `(x, y)` on the Euclidean plane.
    /// Equivalently, compute the length of the hypotenuse of a right-angle triangle with other
    /// sides having length `x._abs()` and `y._abs()`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.hypot>
    #[must_use]
    fn _hypot(self, other: Self) -> Self;

    // ---------------------
    // Scaling / arithmetic.
    // ---------------------

    /// Multiplies this number by `factor`.
    ///
    /// # References
    ///
    /// * \[1\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.scale>
    /// * \[2\] <https://docs.rs/simba/0.10.0/src/simba/scalar/complex.rs.html#1076>
    #[must_use]
    #[inline]
    fn _scale(self, factor: Self) -> Self {
        self * factor
    }

    /// Divides this number by `factor`.
    ///
    /// # References
    ///
    /// * \[1\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.unscale>
    /// * \[2\] <https://docs.rs/simba/0.10.0/src/simba/scalar/complex.rs.html#1081>
    #[must_use]
    #[inline]
    fn _unscale(self, factor: Self) -> Self {
        self / factor
    }

    /// Take the reciprocal (inverse) of a number, `1/x`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.recip>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.recip>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.recip>
    /// * \[4\] <https://docs.rs/faer-traits/latest/faer_traits/trait.ComplexField.html#tymethod.recip_impl>
    #[must_use]
    fn _recip(self) -> Self;

    /// Fused multiply-add. `Computes (self * a) + b`.
    ///
    ///  # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.mul_add>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.mul_add>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.mul_add>
    #[must_use]
    fn _mul_add(self, a: Self, b: Self) -> Self;

    // ------
    // Roots.
    // ------

    /// Returns the square root of a number.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.sqrt>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.sqrt>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.sqrt>
    /// * \[4\] <https://docs.rs/faer-traits/latest/faer_traits/trait.ComplexField.html#tymethod.sqrt_impl>
    #[must_use]
    fn _sqrt(self) -> Self;

    /// Take the square root of a number, returning None if the number is negative.
    ///
    /// # References
    ///
    /// * \[1\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.try_sqrt>
    /// * \[2\] <https://docs.rs/simba/0.10.0/src/simba/scalar/complex.rs.html#883>
    #[must_use]
    #[inline]
    fn _try_sqrt(self) -> Option<Self> {
        if self >= Self::zero() {
            Some(self._sqrt())
        } else {
            None
        }
    }

    /// Returns the cube root of a number.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.cbrt>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.cbrt>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.cbrt>
    #[must_use]
    fn _cbrt(self) -> Self;

    // -------
    // Powers.
    // -------

    /// Raises a number to an integer power.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.powi>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.powi>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.powi>
    #[must_use]
    fn _powi(self, n: i32) -> Self;

    /// Raises a number to a floating-point power.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.powf>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.powf>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.powf>
    #[must_use]
    fn _powf(self, n: Self) -> Self;

    // --------------------------
    // Exponential / logarithmic.
    // --------------------------

    /// Returns `e^(self)`, (the exponential function).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.exp>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.exp>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.exp>
    #[must_use]
    fn _exp(self) -> Self;

    /// Returns `2^(self)`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.exp2>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.exp2>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.exp2>
    #[must_use]
    fn _exp2(self) -> Self;

    /// Returns `e^(self) - 1`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.exp_m1>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.exp_m1>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.exp_m1>
    #[must_use]
    fn _exp_m1(self) -> Self;

    /// Returns the natural logarithm of the number.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.ln>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.ln>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.ln>
    #[must_use]
    fn _ln(self) -> Self;

    /// Returns `ln(1+n)` (natural logarithm)
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.ln_1p>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.ln_1p>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.ln_1p>
    #[must_use]
    fn _ln_1p(self) -> Self;

    /// Returns the logarithm of the number with respect to an arbitrary base.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.log>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.log>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.log>
    #[must_use]
    fn _log(self, base: Self) -> Self;

    /// Returns the base 2 logarithm of the number.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.log2>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.log2>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.log2>
    #[must_use]
    fn _log2(self) -> Self;

    /// Returns the base 10 logarithm of the number.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.log10>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.log10>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.log10>
    #[must_use]
    fn _log10(self) -> Self;

    // ------------------
    // Angle conversions.
    // ------------------

    /// Converts radians to degrees.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.to_degrees>
    /// * \[2\] <https://doc.rust-lang.org/src/core/num/f64.rs.html#991-997>
    /// * \[3\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.to_degrees>
    #[must_use]
    #[inline]
    fn _to_degrees(self) -> Self {
        const PIS_IN_180: f64 = 180.0 / std::f64::consts::PI;
        self * PIS_IN_180
    }

    /// Converts degrees to radians.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.to_radians>
    /// * \[2\] <https://doc.rust-lang.org/src/core/num/f64.rs.html#1020-1026>
    /// * \[3\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.to_radians>
    #[must_use]
    #[inline]
    fn _to_radians(self) -> Self {
        const RADS_PER_DEG: f64 = std::f64::consts::PI / 180.0;
        self * RADS_PER_DEG
    }

    // ------------------------
    // Trigonometric functions.
    // ------------------------

    /// Computes the sine of a number (in radians).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.sin>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.sin>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.sin>
    #[must_use]
    fn _sin(self) -> Self;

    /// Computes the cosine of a number (in radians).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.cos>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.cos>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.cos>
    #[must_use]
    fn _cos(self) -> Self;

    /// Simultaneously computes the sine and cosine of the number, `x`. Returns `(sin(x), cos(x))`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.sin_cos>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.sin_cos>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.sin_cos>
    fn _sin_cos(self) -> (Self, Self);

    /// Computes the tangent of a number (in radians).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.tan>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.tan>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.tan>
    #[must_use]
    fn _tan(self) -> Self;

    /// Computes the cosecant of a number (in radians).
    #[must_use]
    #[inline]
    fn _csc(self) -> Self {
        self._sin()._recip()
    }

    /// Computes the secant of a number (in radians).
    #[must_use]
    #[inline]
    fn _sec(self) -> Self {
        self._cos()._recip()
    }

    /// Computes the cotangent of a number (in radians).
    #[must_use]
    #[inline]
    fn _cot(self) -> Self {
        self._cos() / self._sin()
    }

    /// Computes the arcsine of a number (return value is in radians).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.asin>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.asin>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.asin>
    #[must_use]
    fn _asin(self) -> Self;

    /// Computes the arccosine of a number (return value is in radians).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.acos>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.acos>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.acos>
    #[must_use]
    fn _acos(self) -> Self;

    /// Computes the arctangent of a number (return value is in radians).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.atan>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.atan>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.atan>
    #[must_use]
    fn _atan(self) -> Self;

    /// Computes the four quadrant arctangent of `self` (`y`) and `other` (`x`) in radians.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.atan2>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.atan2>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.atan2>
    #[must_use]
    fn _atan2(self, other: Self) -> Self;

    /// Computes the arccosecant of a number (return value is in radians).
    #[must_use]
    #[inline]
    fn _acsc(self) -> Self {
        self._recip()._asin()
    }

    /// Computes the arcsecant of a number (return value is in radians).
    #[must_use]
    #[inline]
    fn _asec(self) -> Self {
        self._recip()._acos()
    }

    /// Computes the arccotangent of a number (return value is in radians).
    #[must_use]
    #[inline]
    fn _acot(self) -> Self {
        self._recip()._atan()
    }

    /// Computes the sine of a number (in degrees).
    #[must_use]
    #[inline]
    fn _sind(self) -> Self {
        self._to_radians()._sin()
    }

    /// Computes the cosine of a number (in degrees).
    #[must_use]
    #[inline]
    fn _cosd(self) -> Self {
        self._to_radians()._cos()
    }

    /// Simultaneously computes the sine and cosine of a number (in degrees).
    #[must_use]
    #[inline]
    fn _sind_cosd(self) -> (Self, Self) {
        let radians = self._to_radians();
        (radians._sin(), radians._cos())
    }

    /// Computes the tangent of a number (in degrees).
    #[must_use]
    #[inline]
    fn _tand(self) -> Self {
        self._to_radians()._tan()
    }

    /// Computes the cosecant of a number (in degrees).
    #[must_use]
    #[inline]
    fn _cscd(self) -> Self {
        self._to_radians()._csc()
    }

    /// Computes the secant of a number (in degrees).
    #[must_use]
    #[inline]
    fn _secd(self) -> Self {
        self._to_radians()._sec()
    }

    /// Computes the cotangent of a number (in degrees).
    #[must_use]
    #[inline]
    fn _cotd(self) -> Self {
        self._to_radians()._cot()
    }

    /// Computes the arcsine of a number (in degrees).
    #[must_use]
    #[inline]
    fn _asind(self) -> Self {
        self._asin()._to_degrees()
    }

    /// Computes the arccosine of a number (in degrees).
    #[must_use]
    #[inline]
    fn _acosd(self) -> Self {
        self._acos()._to_degrees()
    }

    /// Computes the arctangent of a number (in degrees).
    #[must_use]
    #[inline]
    fn _atand(self) -> Self {
        self._atan()._to_degrees()
    }

    /// Computes the arccosecant of a number (in degrees).
    #[must_use]
    #[inline]
    fn _acscd(self) -> Self {
        self._acsc()._to_degrees()
    }

    /// Computes the arcsecant of a number (in degrees).
    #[must_use]
    #[inline]
    fn _asecd(self) -> Self {
        self._asec()._to_degrees()
    }

    /// Computes the arccotangent of a number (in degrees).
    #[must_use]
    #[inline]
    fn _acotd(self) -> Self {
        self._acot()._to_degrees()
    }

    // ---------------------
    // Hyperbolic functions.
    // ---------------------

    /// Hyperbolic sine function.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.sinh>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.sinh>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.sinh>
    #[must_use]
    fn _sinh(self) -> Self;

    /// Hyperbolic cosine function.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.cosh>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.cosh>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.cosh>
    #[must_use]
    fn _cosh(self) -> Self;

    /// Hyperbolic tangent function.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.tanh>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.tanh>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.tanh>
    #[must_use]
    fn _tanh(self) -> Self;

    /// Hyperbolic cosecant function.
    #[must_use]
    #[inline]
    fn _csch(self) -> Self {
        Self::one() / self._sinh()
    }

    /// Hyperbolic secant function.
    #[must_use]
    #[inline]
    fn _sech(self) -> Self {
        Self::one() / self._cosh()
    }

    /// Hyperbolic cotangent function.
    #[must_use]
    #[inline]
    fn _coth(self) -> Self {
        self._cosh() / self._sinh()
    }

    /// Inverse hyperbolic sine function.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.asinh>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.asinh>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.asinh>
    #[must_use]
    fn _asinh(self) -> Self;

    /// Inverse hyperbolic cosine function.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.acosh>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.acosh>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.acosh>
    #[must_use]
    fn _acosh(self) -> Self;

    /// Inverse hyperbolic tangent function.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.atanh>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.atanh>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.atanh>
    #[must_use]
    fn _atanh(self) -> Self;

    /// Inverse hyperbolic cosecant function.
    #[must_use]
    #[inline]
    fn _acsch(self) -> Self {
        self._recip()._asinh()
    }

    /// Inverse hyperbolic secant function.
    #[must_use]
    #[inline]
    fn _asech(self) -> Self {
        self._recip()._acosh()
    }

    /// Inverse hyperbolic cotangent function.
    #[must_use]
    #[inline]
    fn _acoth(self) -> Self {
        self._recip()._atanh()
    }

    // ---------
    // Rounding.
    // ---------

    /// Returns the largest integer that is less than or equal to `self`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.floor>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.floor>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.floor>
    #[must_use]
    fn _floor(self) -> Self;

    /// Returns the smallest integer that is greater than or equal to `self`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.ceil>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.ceil>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.ceil>
    #[must_use]
    fn _ceil(self) -> Self;

    /// Returns the nearest integer to `self`. If a value is half-way between two integers, round
    /// away from `0.0`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.round>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.round>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.round>
    #[must_use]
    fn _round(self) -> Self;

    /// Returns the integer part of `self`. This means that non-integer numbers are always truncated
    /// towards zero.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.trunc>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.trunc>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.trunc>
    #[must_use]
    fn _trunc(self) -> Self;

    /// Returns the fractional part of `self`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.fract>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.fract>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.fract>
    #[must_use]
    fn _fract(self) -> Self;

    // ------------------------------
    // Magnitude-specific operations.
    // ------------------------------

    /// Returns a number composed of the magnitude of `self` and the sign of `sign`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.copysign>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.copysign>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.copysign>
    #[must_use]
    fn _copysign(self, sign: Self) -> Self;

    /// Returns the minimum of the two numbers, ignoring NaN.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.min>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.min>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.min>
    #[must_use]
    fn _min(self, other: Self) -> Self;

    /// Returns the maximum of the two numbers, ignoring NaN.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.max>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.max>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.max>
    #[must_use]
    fn _max(self, other: Self) -> Self;

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
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.clamp>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.clamp>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.clamp>
    #[must_use]
    fn _clamp(self, min: Self, max: Self) -> Self;

    // -----------------------------------
    // Checking floating point properties.
    // -----------------------------------

    /// Returns `true` if the number is NaN.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.is_nan>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.is_nan>
    fn _is_nan(self) -> bool;

    /// Returns `true` if this number is infinite.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.is_infinite>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.is_infinite>
    fn _is_infinite(self) -> bool;

    /// Returns `true` if this number is neither infinite nor `NaN`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.is_finite>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.is_finite>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.is_finite>
    fn _is_finite(self) -> bool;

    /// Returns `true` if this number is
    /// [subnormal](https://en.wikipedia.org/wiki/Subnormal_number).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.is_subnormal>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.is_subnormal>
    fn _is_subnormal(self) -> bool;

    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal](https://en.wikipedia.org/wiki/Subnormal_number), or NaN.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.is_normal>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.is_normal>
    fn _is_normal(self) -> bool;

    /// Returns the floating point category of the number.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.classify>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.classify>
    fn _classify(self) -> std::num::FpCategory;

    /// Returns `true` if `self` has a positive sign.
    ///
    /// # References
    ///
    /// # \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.is_sign_positive>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.is_sign_positive>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.is_sign_positive>
    fn _is_sign_positive(self) -> bool;

    /// Returns `true` if `self` has a negative sign.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.is_sign_negative>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.is_sign_negative>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.is_sign_negative>
    fn _is_sign_negative(self) -> bool;

    /// Returns the least number greater than `self`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.next_up>
    #[must_use]
    fn _next_up(self) -> Self;

    /// Returns the greatest number less than `self`.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#method.next_down>
    #[must_use]
    fn _next_down(self) -> Self;

    // --------------------------
    // Floating point properties.
    // --------------------------

    /// Returns epsilon, a small positive value.
    ///
    /// For types like [`f64`], this is the
    /// [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon).
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#associatedconstant.EPSILON>
    /// * \[2\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#method.epsilon>
    /// * \[3\] <https://docs.rs/nalgebra/latest/nalgebra/trait.ComplexField.html#tymethod.epsilon>
    /// * \[4\] <https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html#tymethod.epsilon_impl>
    fn _epsilon() -> Self;

    /// The size of this type in bits.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#associatedconstant.BITS>
    /// * \[2\] <https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html#tymethod.nbits_impl>
    fn _bits() -> usize;

    /// Smallest positive normal value.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#associatedconstant.MIN_POSITIVE>
    /// * \[2\] <https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html#tymethod.min_positive_impl>
    /// * \[3\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.min_positive_value>
    fn _min_positive() -> Self;

    /// Largest positive finite value.
    ///
    /// # References
    ///
    /// * \[1\] <https://doc.rust-lang.org/std/primitive.f64.html#associatedconstant.MAX>
    /// * \[2\] <https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html#tymethod.max_positive_impl>
    /// * \[3\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.max_value>
    fn _max_positive() -> Self;

    /// Square root of the smallest positive normal value.
    ///
    /// # References
    ///
    /// * \[1\] <https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html#tymethod.sqrt_min_positive_impl>
    #[must_use]
    #[inline]
    fn _sqrt_min_positive() -> Self {
        Self::_min_positive()._sqrt()
    }

    /// Square root of the largest positive finite value.
    ///
    /// # References
    ///
    /// * \[1\] <https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html#tymethod.sqrt_max_positive_impl>
    #[must_use]
    #[inline]
    fn _sqrt_max_positive() -> Self {
        Self::_max_positive()._sqrt()
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
    /// # References
    ///
    /// # \[1\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.min_value>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.min_value>
    fn _min_value() -> Option<Self>;

    /// The largest finite positive value representable using this type.
    ///
    /// # References
    ///
    /// # \[1\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.max_value>
    /// * \[2\] <https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html#tymethod.max_value>
    fn _max_value() -> Option<Self>;

    /// Returns the `NaN` value.
    ///
    /// # References
    ///
    /// * \[1\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.nan>
    fn _nan() -> Self;

    /// Returns the infinite value.
    ///
    /// # References
    ///
    /// * \[1\] <https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#tymethod.infinity>
    fn _infinity() -> Self;

    // ------------------
    // Interface methods.
    // ------------------

    /// Express this real field type as a slice of `f64`.
    ///
    /// This method is primarily used for deriving [`approx::UlpsEq`] for real valued types that
    /// contain multiple components. Primarily, this can be used for something like a
    /// [dual number](https://docs.rs/numdiff/latest/numdiff/struct.Dual.html) that is substituted
    /// in place of a single real number for forward mode automatic differentiation.
    fn _as_slice(&self) -> &[f64];
}
