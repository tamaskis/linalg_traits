use linalg_traits::RealField;

/// Verifies that every method defined on `RealField` (including those inherited from
/// `linalg_traits::RealFieldBase`, `num_traits::Zero`, and `num_traits::One`) can be called on a
/// generic `T: RealField` without needing to disambiguate between traits.
fn call_all_real_field_methods<T: RealField>(a: T, b: T, c: T) {
    // Base operations.
    let _ = T::default().neg();
    let _ = T::default().add(T::default());
    let _ = T::default().sub(T::default());
    let _ = T::default().mul(T::default());
    let _ = T::default().div(T::default());
    let _ = T::default().rem(T::default());
    let _ = T::default().eq(&T::default());

    // Zero.
    let mut z = a;
    let _ = T::zero();
    let _ = a.is_zero();
    z.set_zero();

    // One.
    let mut o = a;
    let _ = T::one();
    let _ = a.is_one();
    o.set_one();

    // Constants.
    let _ = T::e();
    let _ = T::euler_gamma();
    let _ = T::frac_1_pi();
    let _ = T::frac_1_sqrt_2();
    let _ = T::frac_1_sqrt_2pi();
    let _ = T::frac_1_sqrt_3();
    let _ = T::frac_1_sqrt_5();
    let _ = T::frac_1_sqrt_pi();
    let _ = T::frac_2_pi();
    let _ = T::frac_2_sqrt_pi();
    let _ = T::frac_pi_2();
    let _ = T::frac_pi_3();
    let _ = T::frac_pi_4();
    let _ = T::frac_pi_6();
    let _ = T::frac_pi_8();
    let _ = T::golden_ratio();
    let _ = T::ln_10();
    let _ = T::ln_2();
    let _ = T::log10_2();
    let _ = T::log10_e();
    let _ = T::log2_10();
    let _ = T::log2_e();
    let _ = T::pi();
    let _ = T::sqrt_2();
    let _ = T::sqrt_3();
    let _ = T::sqrt_5();
    let _ = T::tau();
    let _ = T::two_pi();

    // Magnitude.
    let _ = a.abs();
    let _ = a.hypot(b);

    // Scaling / arithmetic.
    let _ = a.scale(b);
    let _ = a.unscale(b);
    let _ = a.recip();
    let _ = a.mul_add(b, c);

    // Roots.
    let _ = a.sqrt();
    let _ = a.try_sqrt();
    let _ = a.cbrt();

    // Powers.
    let _ = a.powi(2);
    let _ = a.powf(b);

    // Exponential / logarithmic.
    let _ = a.exp();
    let _ = a.exp2();
    let _ = a.exp_m1();
    let _ = a.ln();
    let _ = a.ln_1p();
    let _ = a.log(b);
    let _ = a.log2();
    let _ = a.log10();

    // Angle conversions.
    let _ = a.to_degrees();
    let _ = a.to_radians();

    // Trigonometric functions.
    let _ = a.sin();
    let _ = a.cos();
    let _ = a.sin_cos();
    let _ = a.tan();
    let _ = a.csc();
    let _ = a.sec();
    let _ = a.cot();
    let _ = a.asin();
    let _ = a.acos();
    let _ = a.atan();
    let _ = a.atan2(b);
    let _ = a.acsc();
    let _ = a.asec();
    let _ = a.acot();
    let _ = a.sind();
    let _ = a.cosd();
    let _ = a.sind_cosd();
    let _ = a.tand();
    let _ = a.cscd();
    let _ = a.secd();
    let _ = a.cotd();
    let _ = a.asind();
    let _ = a.acosd();
    let _ = a.atand();
    let _ = a.atan2d(b);
    let _ = a.acscd();
    let _ = a.asecd();
    let _ = a.acotd();

    // Hyperbolic functions.
    let _ = a.sinh();
    let _ = a.cosh();
    let _ = a.sinh_cosh();
    let _ = a.tanh();
    let _ = a.csch();
    let _ = a.sech();
    let _ = a.coth();
    let _ = a.asinh();
    let _ = a.acosh();
    let _ = a.atanh();
    let _ = a.acsch();
    let _ = a.asech();
    let _ = a.acoth();

    // Rounding.
    let _ = a.floor();
    let _ = a.ceil();
    let _ = a.round();
    let _ = a.trunc();
    let _ = a.fract();

    // Magnitude-specific operations.
    let _ = a.copysign(b);
    let _ = a.min(b);
    let _ = a.max(b);
    let _ = a.clamp(b, c);

    // Checking floating point properties.
    let _ = a.is_nan();
    let _ = a.is_infinite();
    let _ = a.is_finite();
    let _ = a.is_subnormal();
    let _ = a.is_normal();
    let _ = a.classify();
    let _ = a.is_sign_positive();
    let _ = a.is_sign_negative();
    let _ = a.next_up();
    let _ = a.next_down();

    // Floating point properties.
    let _ = T::epsilon();
    let _ = T::bits();
    let _ = T::min_positive();
    let _ = T::max_positive();
    let _ = T::sqrt_min_positive();
    let _ = T::sqrt_max_positive();
    let _ = T::min_value();
    let _ = T::max_value();
    let _ = T::nan();
    let _ = T::infinity();

    // Interface methods.
    let _ = a.as_slice();
}

#[test]
fn test_real_field_methods_no_disambiguation() {
    call_all_real_field_methods::<f64>(1.0, 2.0, 3.0);
}
