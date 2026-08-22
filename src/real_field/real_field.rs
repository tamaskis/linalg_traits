use crate::real_field::real_field_base::RealFieldBase;
use crate::real_field::real_field_faer::RealFieldFaer;
use crate::real_field::real_field_nalgebra::RealFieldNalgebra;
use crate::real_field::real_field_ndarray::RealFieldNdarray;

// TODO: define implementation sequence

/// Trait defining a real number.
///
/// # Overview
///
/// This trait defines the core functionality expected from any real number type, with the following
/// operators and methods implemented:
///
/// * All operations provided by [`crate::real_field_operations::RealFieldOperations`].
/// * All methods provided by [`crate::real_field::RealFieldBase`].
/// * All methods provided by [`num_traits::Zero`].
/// * All methods provided by [`num_traits::One`].
///
/// This set of operations and methods is a (99%) complete superset of all functionality provided by
/// [`f64`], [`num_traits::Float`] (and its supertraits), [`nalgebra::RealField`]
/// (and its supertraits), and [`faer_traits::RealField`] (and its supertraits). The methods that
/// are not included are ones that are complex number specific or more of a backend features of
/// [`nalgebra::RealField`] or [`faer_traits::RealField`].
///
/// # Background
///
/// This trait serves as the foundational interface for real number types that is cross-compatible
/// across different linear algebra and numerical computation crates. Consequently, it originated as
/// a trait that defined the superset of functionality provided by [`nalgebra::RealField`] and
/// [`faer_traits::RealField`]. However, real numbers are used well beyond these specific crates,
/// with many common operations not required by these crates (for example,
/// [`faer_traits::RealField`] doesn't require that a real number implement a `sin` method, which is
/// one of the most widely used mathematical functions). As a result, this trait defines the
/// superset (mostly; there are some methods missing) of functionality provided by
/// [`num_traits::Float`], [`nalgebra::RealField`], and [`faer_traits::RealField`].
///
/// # Interoperability with [`f64`]s.
///
/// We enforce that real types be interoperable with [`f64`]s. Some common differentiation methods,
/// notably forward-mode automatic differentation and complex-step differentiation, rely on
/// replacing real numbers with a custom type of number that has its own arithmetic (dual numbers
/// for forward-mode automatic differentiation, complex numbers for complex-step differentiation).
/// Forcing scalars to have this interoperability with [`f64`]s built-in helps enable downstream
/// crates to write functions in way that can be used with both plain [`f64`]s for most use cases,
/// and with custom types when the functions need to be differentiated.
///
/// Additionally, we chose to restrict this interoperability to be with [`f64`]s since
/// double-precision floating point numbers are the de facto standard for numerical computations.
///
/// This interoperability is mandated by [`crate::real_field_operations::F64Interop`] (a supertrait
/// of [`crate::real_field_operations::RealFieldOperations`]).
pub trait RealField: RealFieldBase + RealFieldFaer + RealFieldNalgebra + RealFieldNdarray {
    /// Construct an instance of this scalar from an [`f64`].
    ///
    /// # Arguments
    ///
    /// * `x` - An [`f64`].
    ///
    /// # Return
    ///
    /// An instance of this scalar type constructed from an [`f64`].
    #[must_use]
    #[inline]
    fn new(x: f64) -> Self {
        <Self as From<f64>>::from(x)
    }

    // ------------------------------------------------------------------------------------------
    // Ergonomic (non-underscore-prefixed) re-exports of `RealFieldBase`'s methods.
    //
    // When the `nalgebra` feature is disabled, these are provided as default methods delegating
    // to `RealFieldBase`'s underscore-prefixed methods of the same name. When the `nalgebra`
    // feature is enabled, the ones that collide with `nalgebra::RealField`/`simba` method names
    // are omitted here so that they are instead provided solely by the `nalgebra::RealField`
    // supertrait (via `RealFieldNalgebra`), avoiding ambiguous method resolution.
    // ------------------------------------------------------------------------------------------

    /// See [`RealFieldBase::_euler_gamma`].
    #[must_use]
    #[inline]
    fn euler_gamma() -> Self {
        Self::_euler_gamma()
    }

    /// See [`RealFieldBase::_frac_1_sqrt_2`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_2() -> Self {
        Self::_frac_1_sqrt_2()
    }

    /// See [`RealFieldBase::_frac_1_sqrt_2pi`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_2pi() -> Self {
        Self::_frac_1_sqrt_2pi()
    }

    /// See [`RealFieldBase::_frac_1_sqrt_3`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_3() -> Self {
        Self::_frac_1_sqrt_3()
    }

    /// See [`RealFieldBase::_frac_1_sqrt_5`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_5() -> Self {
        Self::_frac_1_sqrt_5()
    }

    /// See [`RealFieldBase::_frac_1_sqrt_pi`].
    #[must_use]
    #[inline]
    fn frac_1_sqrt_pi() -> Self {
        Self::_frac_1_sqrt_pi()
    }

    /// See [`RealFieldBase::_golden_ratio`].
    #[must_use]
    #[inline]
    fn golden_ratio() -> Self {
        Self::_golden_ratio()
    }

    /// See [`RealFieldBase::_log10_2`].
    #[must_use]
    #[inline]
    fn log10_2() -> Self {
        Self::_log10_2()
    }

    /// See [`RealFieldBase::_log2_10`].
    #[must_use]
    #[inline]
    fn log2_10() -> Self {
        Self::_log2_10()
    }

    /// See [`RealFieldBase::_sqrt_2`].
    #[must_use]
    #[inline]
    fn sqrt_2() -> Self {
        Self::_sqrt_2()
    }

    /// See [`RealFieldBase::_sqrt_3`].
    #[must_use]
    #[inline]
    fn sqrt_3() -> Self {
        Self::_sqrt_3()
    }

    /// See [`RealFieldBase::_sqrt_5`].
    #[must_use]
    #[inline]
    fn sqrt_5() -> Self {
        Self::_sqrt_5()
    }

    /// See [`RealFieldBase::_tau`].
    #[must_use]
    #[inline]
    fn tau() -> Self {
        Self::_tau()
    }

    /// See [`RealFieldBase::_e`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn e() -> Self {
        Self::_e()
    }

    /// See [`RealFieldBase::_frac_1_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_1_pi() -> Self {
        Self::_frac_1_pi()
    }

    /// See [`RealFieldBase::_frac_2_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_2_pi() -> Self {
        Self::_frac_2_pi()
    }

    /// See [`RealFieldBase::_frac_2_sqrt_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self::_frac_2_sqrt_pi()
    }

    /// See [`RealFieldBase::_frac_pi_2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_2() -> Self {
        Self::_frac_pi_2()
    }

    /// See [`RealFieldBase::_frac_pi_3`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_3() -> Self {
        Self::_frac_pi_3()
    }

    /// See [`RealFieldBase::_frac_pi_4`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_4() -> Self {
        Self::_frac_pi_4()
    }

    /// See [`RealFieldBase::_frac_pi_6`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_6() -> Self {
        Self::_frac_pi_6()
    }

    /// See [`RealFieldBase::_frac_pi_8`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn frac_pi_8() -> Self {
        Self::_frac_pi_8()
    }

    /// See [`RealFieldBase::_ln_10`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ln_10() -> Self {
        Self::_ln_10()
    }

    /// See [`RealFieldBase::_ln_2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ln_2() -> Self {
        Self::_ln_2()
    }

    /// See [`RealFieldBase::_log10_e`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log10_e() -> Self {
        Self::_log10_e()
    }

    /// See [`RealFieldBase::_log2_e`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log2_e() -> Self {
        Self::_log2_e()
    }

    /// See [`RealFieldBase::_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn pi() -> Self {
        Self::_pi()
    }

    /// See [`RealFieldBase::_two_pi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn two_pi() -> Self {
        Self::_two_pi()
    }

    /// See [`RealFieldBase::_abs`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn abs(self) -> Self {
        self._abs()
    }

    /// See [`RealFieldBase::_hypot`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn hypot(self, other: Self) -> Self {
        self._hypot(other)
    }

    /// See [`RealFieldBase::_scale`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn scale(self, factor: Self) -> Self {
        self._scale(factor)
    }

    /// See [`RealFieldBase::_unscale`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn unscale(self, factor: Self) -> Self {
        self._unscale(factor)
    }

    /// See [`RealFieldBase::_recip`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn recip(self) -> Self {
        self._recip()
    }

    /// See [`RealFieldBase::_mul_add`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self._mul_add(a, b)
    }

    /// See [`RealFieldBase::_sqrt`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn sqrt(self) -> Self {
        self._sqrt()
    }

    /// See [`RealFieldBase::_try_sqrt`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn try_sqrt(self) -> Option<Self> {
        self._try_sqrt()
    }

    /// See [`RealFieldBase::_cbrt`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn cbrt(self) -> Self {
        self._cbrt()
    }

    /// See [`RealFieldBase::_powi`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn powi(self, n: i32) -> Self {
        self._powi(n)
    }

    /// See [`RealFieldBase::_powf`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn powf(self, n: Self) -> Self {
        self._powf(n)
    }

    /// See [`RealFieldBase::_exp`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn exp(self) -> Self {
        self._exp()
    }

    /// See [`RealFieldBase::_exp2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn exp2(self) -> Self {
        self._exp2()
    }

    /// See [`RealFieldBase::_exp_m1`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn exp_m1(self) -> Self {
        self._exp_m1()
    }

    /// See [`RealFieldBase::_ln`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ln(self) -> Self {
        self._ln()
    }

    /// See [`RealFieldBase::_ln_1p`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ln_1p(self) -> Self {
        self._ln_1p()
    }

    /// See [`RealFieldBase::_log`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log(self, base: Self) -> Self {
        self._log(base)
    }

    /// See [`RealFieldBase::_log2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log2(self) -> Self {
        self._log2()
    }

    /// See [`RealFieldBase::_log10`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn log10(self) -> Self {
        self._log10()
    }

    /// See [`RealFieldBase::_to_degrees`].
    #[must_use]
    #[inline]
    fn to_degrees(self) -> Self {
        self._to_degrees()
    }

    /// See [`RealFieldBase::_to_radians`].
    #[must_use]
    #[inline]
    fn to_radians(self) -> Self {
        self._to_radians()
    }

    /// See [`RealFieldBase::_sin`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn sin(self) -> Self {
        self._sin()
    }

    /// See [`RealFieldBase::_cos`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn cos(self) -> Self {
        self._cos()
    }

    /// See [`RealFieldBase::_sin_cos`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        self._sin_cos()
    }

    /// See [`RealFieldBase::_tan`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn tan(self) -> Self {
        self._tan()
    }

    /// See [`RealFieldBase::_csc`].
    #[must_use]
    #[inline]
    fn csc(self) -> Self {
        self._csc()
    }

    /// See [`RealFieldBase::_sec`].
    #[must_use]
    #[inline]
    fn sec(self) -> Self {
        self._sec()
    }

    /// See [`RealFieldBase::_cot`].
    #[must_use]
    #[inline]
    fn cot(self) -> Self {
        self._cot()
    }

    /// See [`RealFieldBase::_asin`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn asin(self) -> Self {
        self._asin()
    }

    /// See [`RealFieldBase::_acos`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn acos(self) -> Self {
        self._acos()
    }

    /// See [`RealFieldBase::_atan`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn atan(self) -> Self {
        self._atan()
    }

    /// See [`RealFieldBase::_atan2`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn atan2(self, other: Self) -> Self {
        self._atan2(other)
    }

    /// See [`RealFieldBase::_acsc`].
    #[must_use]
    #[inline]
    fn acsc(self) -> Self {
        self._acsc()
    }

    /// See [`RealFieldBase::_asec`].
    #[must_use]
    #[inline]
    fn asec(self) -> Self {
        self._asec()
    }

    /// See [`RealFieldBase::_acot`].
    #[must_use]
    #[inline]
    fn acot(self) -> Self {
        self._acot()
    }

    /// See [`RealFieldBase::_sind`].
    #[must_use]
    #[inline]
    fn sind(self) -> Self {
        self._sind()
    }

    /// See [`RealFieldBase::_cosd`].
    #[must_use]
    #[inline]
    fn cosd(self) -> Self {
        self._cosd()
    }

    /// See [`RealFieldBase::_sind_cosd`].
    #[must_use]
    #[inline]
    fn sind_cosd(self) -> (Self, Self) {
        self._sind_cosd()
    }

    /// See [`RealFieldBase::_tand`].
    #[must_use]
    #[inline]
    fn tand(self) -> Self {
        self._tand()
    }

    /// See [`RealFieldBase::_cscd`].
    #[must_use]
    #[inline]
    fn cscd(self) -> Self {
        self._cscd()
    }

    /// See [`RealFieldBase::_secd`].
    #[must_use]
    #[inline]
    fn secd(self) -> Self {
        self._secd()
    }

    /// See [`RealFieldBase::_cotd`].
    #[must_use]
    #[inline]
    fn cotd(self) -> Self {
        self._cotd()
    }

    /// See [`RealFieldBase::_asind`].
    #[must_use]
    #[inline]
    fn asind(self) -> Self {
        self._asind()
    }

    /// See [`RealFieldBase::_acosd`].
    #[must_use]
    #[inline]
    fn acosd(self) -> Self {
        self._acosd()
    }

    /// See [`RealFieldBase::_atand`].
    #[must_use]
    #[inline]
    fn atand(self) -> Self {
        self._atand()
    }

    /// See [`RealFieldBase::_acscd`].
    #[must_use]
    #[inline]
    fn acscd(self) -> Self {
        self._acscd()
    }

    /// See [`RealFieldBase::_asecd`].
    #[must_use]
    #[inline]
    fn asecd(self) -> Self {
        self._asecd()
    }

    /// See [`RealFieldBase::_acotd`].
    #[must_use]
    #[inline]
    fn acotd(self) -> Self {
        self._acotd()
    }

    /// See [`RealFieldBase::_sinh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn sinh(self) -> Self {
        self._sinh()
    }

    /// See [`RealFieldBase::_cosh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn cosh(self) -> Self {
        self._cosh()
    }

    /// See [`RealFieldBase::_tanh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn tanh(self) -> Self {
        self._tanh()
    }

    /// See [`RealFieldBase::_csch`].
    #[must_use]
    #[inline]
    fn csch(self) -> Self {
        self._csch()
    }

    /// See [`RealFieldBase::_sech`].
    #[must_use]
    #[inline]
    fn sech(self) -> Self {
        self._sech()
    }

    /// See [`RealFieldBase::_coth`].
    #[must_use]
    #[inline]
    fn coth(self) -> Self {
        self._coth()
    }

    /// See [`RealFieldBase::_asinh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn asinh(self) -> Self {
        self._asinh()
    }

    /// See [`RealFieldBase::_acosh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn acosh(self) -> Self {
        self._acosh()
    }

    /// See [`RealFieldBase::_atanh`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn atanh(self) -> Self {
        self._atanh()
    }

    /// See [`RealFieldBase::_acsch`].
    #[must_use]
    #[inline]
    fn acsch(self) -> Self {
        self._acsch()
    }

    /// See [`RealFieldBase::_asech`].
    #[must_use]
    #[inline]
    fn asech(self) -> Self {
        self._asech()
    }

    /// See [`RealFieldBase::_acoth`].
    #[must_use]
    #[inline]
    fn acoth(self) -> Self {
        self._acoth()
    }

    /// See [`RealFieldBase::_floor`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn floor(self) -> Self {
        self._floor()
    }

    /// See [`RealFieldBase::_ceil`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn ceil(self) -> Self {
        self._ceil()
    }

    /// See [`RealFieldBase::_round`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn round(self) -> Self {
        self._round()
    }

    /// See [`RealFieldBase::_trunc`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn trunc(self) -> Self {
        self._trunc()
    }

    /// See [`RealFieldBase::_fract`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn fract(self) -> Self {
        self._fract()
    }

    /// See [`RealFieldBase::_copysign`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        self._copysign(sign)
    }

    /// See [`RealFieldBase::_min`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn min(self, other: Self) -> Self {
        self._min(other)
    }

    /// See [`RealFieldBase::_max`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn max(self, other: Self) -> Self {
        self._max(other)
    }

    /// See [`RealFieldBase::_clamp`].
    #[cfg(not(feature = "nalgebra"))]
    #[must_use]
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        self._clamp(min, max)
    }

    /// See [`RealFieldBase::_is_nan`].
    #[inline]
    fn is_nan(self) -> bool {
        self._is_nan()
    }

    /// See [`RealFieldBase::_is_infinite`].
    #[inline]
    fn is_infinite(self) -> bool {
        self._is_infinite()
    }

    /// See [`RealFieldBase::_is_finite`].
    #[inline]
    fn is_finite(self) -> bool {
        self._is_finite()
    }

    /// See [`RealFieldBase::_is_subnormal`].
    #[inline]
    fn is_subnormal(self) -> bool {
        self._is_subnormal()
    }

    /// See [`RealFieldBase::_is_normal`].
    #[inline]
    fn is_normal(self) -> bool {
        self._is_normal()
    }

    /// See [`RealFieldBase::_classify`].
    #[inline]
    fn classify(self) -> std::num::FpCategory {
        self._classify()
    }

    /// See [`RealFieldBase::_is_sign_positive`].
    #[inline]
    fn is_sign_positive(self) -> bool {
        self._is_sign_positive()
    }

    /// See [`RealFieldBase::_is_sign_negative`].
    #[inline]
    fn is_sign_negative(self) -> bool {
        self._is_sign_negative()
    }

    /// See [`RealFieldBase::_next_up`].
    #[must_use]
    #[inline]
    fn next_up(self) -> Self {
        self._next_up()
    }

    /// See [`RealFieldBase::_next_down`].
    #[must_use]
    #[inline]
    fn next_down(self) -> Self {
        self._next_down()
    }

    /// See [`RealFieldBase::_epsilon`].
    #[must_use]
    #[inline]
    fn epsilon() -> Self {
        Self::_epsilon()
    }

    /// See [`RealFieldBase::_bits`].
    #[must_use]
    #[inline]
    fn bits() -> usize {
        Self::_bits()
    }

    /// See [`RealFieldBase::_min_positive`].
    #[must_use]
    #[inline]
    fn min_positive() -> Self {
        Self::_min_positive()
    }

    /// See [`RealFieldBase::_max_positive`].
    #[must_use]
    #[inline]
    fn max_positive() -> Self {
        Self::_max_positive()
    }

    /// See [`RealFieldBase::_sqrt_min_positive`].
    #[must_use]
    #[inline]
    fn sqrt_min_positive() -> Self {
        Self::_sqrt_min_positive()
    }

    /// See [`RealFieldBase::_sqrt_max_positive`].
    #[must_use]
    #[inline]
    fn sqrt_max_positive() -> Self {
        Self::_sqrt_max_positive()
    }

    /// See [`RealFieldBase::_min_value`].
    #[cfg(not(feature = "nalgebra"))]
    #[inline]
    fn min_value() -> Option<Self> {
        Self::_min_value()
    }

    /// See [`RealFieldBase::_max_value`].
    #[cfg(not(feature = "nalgebra"))]
    #[inline]
    fn max_value() -> Option<Self> {
        Self::_max_value()
    }

    /// See [`RealFieldBase::_nan`].
    #[must_use]
    #[inline]
    fn nan() -> Self {
        Self::_nan()
    }

    /// See [`RealFieldBase::_infinity`].
    #[must_use]
    #[inline]
    fn infinity() -> Self {
        Self::_infinity()
    }

    /// See [`RealFieldBase::_as_slice`].
    #[inline]
    fn as_slice(&self) -> &[f64] {
        self._as_slice()
    }
}

// Blanket implementation.
impl<T> RealField for T where T: RealFieldBase + RealFieldFaer + RealFieldNalgebra + RealFieldNdarray
{}
