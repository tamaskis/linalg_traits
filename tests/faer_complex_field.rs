#![cfg(feature = "faer")]

use linalg_traits::real_field::RealFieldBase;
use linalg_traits_example::my_real::MyReal;

#[test]
#[allow(clippy::assertions_on_constants)]
fn test_faer_complex_field_is_real() {
    assert!(<MyReal as faer_traits::ComplexField>::IS_REAL);
}

#[test]
fn test_faer_complex_field_simd_capabilities_is_copy() {
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::SIMD_CAPABILITIES,
        faer_traits::SimdCapabilities::Copy
    );
}

#[test]
fn test_faer_complex_field_zero_impl_matches_real_field_base() {
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::zero_impl(),
        <MyReal as RealFieldBase>::_zero()
    );
}

#[test]
fn test_faer_complex_field_one_impl_matches_real_field_base() {
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::one_impl(),
        <MyReal as RealFieldBase>::_one()
    );
}

#[test]
fn test_faer_complex_field_nan_impl_is_nan() {
    assert!(
        <MyReal as faer_traits::ComplexField>::nan_impl()
            ._to_f64()
            .is_nan()
    );
    assert!(<MyReal as RealFieldBase>::_nan()._to_f64().is_nan());
}

#[test]
fn test_faer_complex_field_infinity_impl_matches_real_field_base() {
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::infinity_impl(),
        <MyReal as RealFieldBase>::_infinity()
    );
}

#[test]
fn test_faer_complex_field_from_real_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(<MyReal as faer_traits::ComplexField>::from_real_impl(&a), a);
}

#[test]
fn test_faer_complex_field_from_f64_impl_matches_real_field_base() {
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::from_f64_impl(3.5),
        <MyReal as From<f64>>::from(3.5)
    );
}

#[test]
fn test_faer_complex_field_real_part_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(<MyReal as faer_traits::ComplexField>::real_part_impl(&a), a);
}

#[test]
fn test_faer_complex_field_imag_part_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::imag_part_impl(&a),
        <MyReal as RealFieldBase>::_zero()
    );
}

#[test]
fn test_faer_complex_field_copy_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(<MyReal as faer_traits::ComplexField>::copy_impl(&a), a);
}

#[test]
fn test_faer_complex_field_conj_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(<MyReal as faer_traits::ComplexField>::conj_impl(&a), a);
}

#[test]
fn test_faer_complex_field_recip_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::recip_impl(&a),
        <MyReal as RealFieldBase>::_recip(a)
    );
}

#[test]
fn test_faer_complex_field_sqrt_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::sqrt_impl(&a),
        <MyReal as RealFieldBase>::_sqrt(a)
    );
}

#[test]
fn test_faer_complex_field_abs_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::abs_impl(&a),
        <MyReal as RealFieldBase>::_abs(&a)
    );
}

#[test]
fn test_faer_complex_field_abs1_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::abs1_impl(&a),
        <MyReal as RealFieldBase>::_abs(&a)
    );
}

#[test]
fn test_faer_complex_field_abs2_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::abs2_impl(&a),
        <MyReal as RealFieldBase>::_mul(a, a)
    );
}

#[test]
fn test_faer_complex_field_mul_real_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::mul_real_impl(&a, &b),
        <MyReal as RealFieldBase>::_mul(a, b)
    );
}

#[test]
fn test_faer_complex_field_mul_pow2_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    let b = MyReal::new(-1.25);
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::mul_pow2_impl(&a, &b),
        <MyReal as RealFieldBase>::_mul(a, b)
    );
}

#[test]
fn test_faer_complex_field_is_finite_impl_matches_real_field_base() {
    let a = MyReal::new(2.5);
    assert_eq!(
        <MyReal as faer_traits::ComplexField>::is_finite_impl(&a),
        <MyReal as RealFieldBase>::_is_finite(&a)
    );
}
