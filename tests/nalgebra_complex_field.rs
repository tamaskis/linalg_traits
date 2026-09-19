#![cfg(feature = "nalgebra")]

use linalg_traits::real_field::RealFieldBase;
use linalg_traits_example::my_real::MyReal;

#[test]
fn test_nalgebra_complex_field_from_real_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(<MyReal as nalgebra::ComplexField>::from_real(a), a);
}

#[test]
fn test_nalgebra_complex_field_real_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(<MyReal as nalgebra::ComplexField>::real(a), a);
}

#[test]
fn test_nalgebra_complex_field_imaginary_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let zero = <MyReal as RealFieldBase>::_zero();
    assert_eq!(<MyReal as nalgebra::ComplexField>::imaginary(a), zero);
}

#[test]
fn test_nalgebra_complex_field_modulus_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::modulus(a),
        <MyReal as RealFieldBase>::_abs(&a)
    );
}

#[test]
fn test_nalgebra_complex_field_modulus_squared_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::modulus_squared(a),
        <MyReal as RealFieldBase>::_mul(a, a)
    );
}

#[test]
fn test_nalgebra_complex_field_argument_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let zero = <MyReal as RealFieldBase>::_zero();
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::argument(a),
        <MyReal as RealFieldBase>::_atan2(a, zero)
    );
}

#[test]
fn test_nalgebra_complex_field_norm1_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::norm1(a),
        <MyReal as RealFieldBase>::_abs(&a)
    );
}

#[test]
fn test_nalgebra_complex_field_scale_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::scale(a, b),
        <MyReal as RealFieldBase>::_scale(a, b)
    );
}

#[test]
fn test_nalgebra_complex_field_unscale_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::unscale(a, b),
        <MyReal as RealFieldBase>::_unscale(a, b)
    );
}

#[test]
fn test_nalgebra_complex_field_floor_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::floor(a),
        <MyReal as RealFieldBase>::_floor(a)
    );
}

#[test]
fn test_nalgebra_complex_field_ceil_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::ceil(a),
        <MyReal as RealFieldBase>::_ceil(a)
    );
}

#[test]
fn test_nalgebra_complex_field_round_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::round(a),
        <MyReal as RealFieldBase>::_round(a)
    );
}

#[test]
fn test_nalgebra_complex_field_trunc_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::trunc(a),
        <MyReal as RealFieldBase>::_trunc(a)
    );
}

#[test]
fn test_nalgebra_complex_field_fract_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::fract(a),
        <MyReal as RealFieldBase>::_fract(a)
    );
}

#[test]
fn test_nalgebra_complex_field_mul_add_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    let c = MyReal::new(3.0);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::mul_add(a, b, c),
        <MyReal as RealFieldBase>::_mul_add(a, b, c)
    );
}

#[test]
fn test_nalgebra_complex_field_abs_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::abs(a),
        <MyReal as RealFieldBase>::_abs(&a)
    );
}

#[test]
fn test_nalgebra_complex_field_hypot_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::hypot(a, b),
        <MyReal as RealFieldBase>::_hypot(a, b)
    );
}

#[test]
fn test_nalgebra_complex_field_recip_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::recip(a),
        <MyReal as RealFieldBase>::_recip(a)
    );
}

#[test]
fn test_nalgebra_complex_field_conjugate_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(<MyReal as nalgebra::ComplexField>::conjugate(a), a);
}

#[test]
fn test_nalgebra_complex_field_sin_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::sin(a),
        <MyReal as RealFieldBase>::_sin(a)
    );
}

#[test]
fn test_nalgebra_complex_field_cos_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::cos(a),
        <MyReal as RealFieldBase>::_cos(a)
    );
}

#[test]
fn test_nalgebra_complex_field_sin_cos_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::sin_cos(a),
        <MyReal as RealFieldBase>::_sin_cos(a)
    );
}

#[test]
fn test_nalgebra_complex_field_tan_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::tan(a),
        <MyReal as RealFieldBase>::_tan(a)
    );
}

#[test]
fn test_nalgebra_complex_field_asin_matches_real_field_base() {
    let a = MyReal::new(0.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::asin(a),
        <MyReal as RealFieldBase>::_asin(a)
    );
}

#[test]
fn test_nalgebra_complex_field_acos_matches_real_field_base() {
    let a = MyReal::new(0.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::acos(a),
        <MyReal as RealFieldBase>::_acos(a)
    );
}

#[test]
fn test_nalgebra_complex_field_atan_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::atan(a),
        <MyReal as RealFieldBase>::_atan(a)
    );
}

#[test]
fn test_nalgebra_complex_field_sinh_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::sinh(a),
        <MyReal as RealFieldBase>::_sinh(a)
    );
}

#[test]
fn test_nalgebra_complex_field_cosh_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::cosh(a),
        <MyReal as RealFieldBase>::_cosh(a)
    );
}

#[test]
fn test_nalgebra_complex_field_tanh_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::tanh(a),
        <MyReal as RealFieldBase>::_tanh(a)
    );
}

#[test]
fn test_nalgebra_complex_field_asinh_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::asinh(a),
        <MyReal as RealFieldBase>::_asinh(a)
    );
}

#[test]
fn test_nalgebra_complex_field_acosh_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::acosh(a),
        <MyReal as RealFieldBase>::_acosh(a)
    );
}

#[test]
fn test_nalgebra_complex_field_atanh_matches_real_field_base() {
    let a = MyReal::new(0.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::atanh(a),
        <MyReal as RealFieldBase>::_atanh(a)
    );
}

#[test]
fn test_nalgebra_complex_field_log_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(10.0);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::log(a, b),
        <MyReal as RealFieldBase>::_log(a, b)
    );
}

#[test]
fn test_nalgebra_complex_field_log2_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::log2(a),
        <MyReal as RealFieldBase>::_log2(a)
    );
}

#[test]
fn test_nalgebra_complex_field_log10_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::log10(a),
        <MyReal as RealFieldBase>::_log10(a)
    );
}

#[test]
fn test_nalgebra_complex_field_ln_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::ln(a),
        <MyReal as RealFieldBase>::_ln(a)
    );
}

#[test]
fn test_nalgebra_complex_field_ln_1p_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::ln_1p(a),
        <MyReal as RealFieldBase>::_ln_1p(a)
    );
}

#[test]
fn test_nalgebra_complex_field_sqrt_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::sqrt(a),
        <MyReal as RealFieldBase>::_sqrt(a)
    );
}

#[test]
fn test_nalgebra_complex_field_exp_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::exp(a),
        <MyReal as RealFieldBase>::_exp(a)
    );
}

#[test]
fn test_nalgebra_complex_field_exp2_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::exp2(a),
        <MyReal as RealFieldBase>::_exp2(a)
    );
}

#[test]
fn test_nalgebra_complex_field_exp_m1_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::exp_m1(a),
        <MyReal as RealFieldBase>::_exp_m1(a)
    );
}

#[test]
fn test_nalgebra_complex_field_powi_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::powi(a, 3),
        <MyReal as RealFieldBase>::_powi(a, 3)
    );
}

#[test]
fn test_nalgebra_complex_field_powf_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(1.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::powf(a, b),
        <MyReal as RealFieldBase>::_powf(a, b)
    );
}

#[test]
fn test_nalgebra_complex_field_powc_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(1.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::powc(a, b),
        <MyReal as RealFieldBase>::_powf(a, b)
    );
}

#[test]
fn test_nalgebra_complex_field_cbrt_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::cbrt(a),
        <MyReal as RealFieldBase>::_cbrt(a)
    );
}

#[test]
fn test_nalgebra_complex_field_is_finite_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::is_finite(&a),
        <MyReal as RealFieldBase>::_is_finite(&a)
    );
}

#[test]
fn test_nalgebra_complex_field_try_sqrt_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::ComplexField>::try_sqrt(a),
        <MyReal as RealFieldBase>::_try_sqrt(a)
    );
}

#[test]
fn test_nalgebra_real_field_copysign_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    assert_eq!(
        <MyReal as nalgebra::RealField>::copysign(a, b),
        <MyReal as RealFieldBase>::_copysign(a, b)
    );
}

#[test]
fn test_nalgebra_real_field_max_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    assert_eq!(
        <MyReal as nalgebra::RealField>::max(a, b),
        <MyReal as RealFieldBase>::_max(a, b)
    );
}

#[test]
fn test_nalgebra_real_field_min_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    assert_eq!(
        <MyReal as nalgebra::RealField>::min(a, b),
        <MyReal as RealFieldBase>::_min(a, b)
    );
}

#[test]
fn test_nalgebra_real_field_clamp_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    assert_eq!(
        <MyReal as nalgebra::RealField>::clamp(a, b, MyReal::new(3.0)),
        <MyReal as RealFieldBase>::_clamp(a, b, MyReal::new(3.0))
    );
}

#[test]
fn test_nalgebra_real_field_pi_matches_real_field_base() {
    assert_eq!(
        <MyReal as nalgebra::RealField>::pi(),
        <MyReal as RealFieldBase>::_pi()
    );
}

#[test]
fn test_nalgebra_real_field_e_matches_real_field_base() {
    assert_eq!(
        <MyReal as nalgebra::RealField>::e(),
        <MyReal as RealFieldBase>::_e()
    );
}

#[test]
fn test_nalgebra_real_field_is_sign_positive_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as nalgebra::RealField>::is_sign_positive(&a),
        <MyReal as RealFieldBase>::_is_sign_positive(&a)
    );
}
