use crate::real_field::real_field::RealField;

/// Verifies that every method defined on [`crate::RealField`] (including those inherited from
/// [`crate::real_field::RealFieldBase`], [`num_traits::Zero`], and [`num_traits::One`]) can be
/// called on a generic `T: RealField` without needing to disambiguate between traits.
///
/// # Generic Arguments
///
/// * `T` - Type implementing [`crate::RealField`] to test.
///
/// # Arguments
///
/// * `a` - First value used to exercise the method surface.
/// * `b` - Second value used to exercise the method surface.
/// * `c` - Third value used to exercise the method surface.
///
/// # Panics
///
/// Panics if any method call is ambiguous or otherwise invalid for the generic type.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
pub fn assert_no_trait_disambiguation_needed<T: RealField>(a: T, b: T, c: T) {
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

/// Asserts that all supertraits defined by [`crate::real_field::Base`] produce the expected
/// results.
///
/// # Generic Arguments
///
/// * `T` - Type implementing [`crate::real_field::Base`] to test.
///
/// # Arguments
///
/// * `value` - The value to test.
/// * `default_value` - The expected default value for the type.
/// * `debug_str` - The expected debug string representation of `value`.
/// * `display_str` - The expected display string representation of `value`.
///
/// # Panics
///
/// If any trait's behavior does not match its expected behavior.
#[allow(clippy::clone_on_copy)]
pub fn assert_base<T: RealField>(value: T, default_value: T, debug_str: &str, display_str: &str) {
    // Check that debug output matches the expected string.
    assert_eq!(format!("{value:?}"), debug_str);

    // Check that display output matches the expected string.
    assert_eq!(format!("{value}"), display_str);

    // Check that the default value matches the expected default value.
    assert_eq!(T::default(), default_value);

    // Check copy.
    let mut copy = value;
    assert_eq!(copy, value);
    copy += 1.0;
    assert_eq!(copy, value + 1.0);

    // Check clone.
    let mut clone = value.clone();
    assert_eq!(clone, value);
    clone += 1.0;
    assert_eq!(clone, value + 1.0);
}

/// Asserts that all arithmetic operation forms guaranteed by
/// [`crate::real_field::RealFieldOperations`] produce the expected results.
///
/// # Generic Arguments
///
/// * `T` - Type implementing [`crate::real_field::RealFieldOperations`] to test.
///
/// # Arguments
///
/// * `lhs` - Left-hand side operand.
/// * `rhs` - Right-hand side operand.
/// * `expected_neg` - Expected result of `-lhs`.
/// * `expected_add` - Expected result of `lhs + rhs`.
/// * `expected_sub` - Expected result of `lhs - rhs`.
/// * `expected_mul` - Expected result of `lhs * rhs`.
/// * `expected_div` - Expected result of `lhs / rhs`.
/// * `expected_rem` - Expected result of `lhs % rhs`.
/// * `expected_eq` - Expected result of `lhs == rhs`.
/// * `expected_lt` - Expected result of `lhs < rhs`.
/// * `expected_le` - Expected result of `lhs <= rhs`.
/// * `expected_gt` - Expected result of `lhs > rhs`.
/// * `expected_ge` - Expected result of `lhs >= rhs`.
///
/// # Panics
///
/// Panics if any of the arithmetic operations on `lhs`/`rhs` don't match their expected result.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::fn_params_excessive_bools,
    clippy::op_ref
)]
pub fn assert_real_field_operations<T: RealField>(
    lhs: T,
    rhs: T,
    expected_neg: T,
    expected_add: T,
    expected_sub: T,
    expected_mul: T,
    expected_div: T,
    expected_rem: T,
    expected_eq: bool,
    expected_lt: bool,
    expected_le: bool,
    expected_gt: bool,
    expected_ge: bool,
) {
    assert_eq!(lhs == rhs, expected_eq);
    assert_eq!(lhs != rhs, !expected_eq);
    assert_eq!(lhs < rhs, expected_lt);
    assert_eq!(lhs <= rhs, expected_le);
    assert_eq!(lhs > rhs, expected_gt);
    assert_eq!(lhs >= rhs, expected_ge);

    // Negation: `-T`.
    assert_eq!(-lhs, expected_neg);

    // Addition: `T + T`, `T + &T`, `&T + T`, and `&T + &T`.
    assert_eq!(lhs + rhs, expected_add);
    assert_eq!(lhs + &rhs, expected_add);
    assert_eq!(&lhs + rhs, expected_add);
    assert_eq!(&lhs + &rhs, expected_add);

    // Subtraction: `T - T`, `T - &T`, `&T - T`, and `&T - &T`.
    assert_eq!(lhs - rhs, expected_sub);
    assert_eq!(lhs - &rhs, expected_sub);
    assert_eq!(&lhs - rhs, expected_sub);
    assert_eq!(&lhs - &rhs, expected_sub);

    // Multiplication: `T * T`, `T * &T`, `&T * T`, and `&T * &T`.
    assert_eq!(lhs * rhs, expected_mul);
    assert_eq!(lhs * &rhs, expected_mul);
    assert_eq!(&lhs * rhs, expected_mul);
    assert_eq!(&lhs * &rhs, expected_mul);

    // Division: `T / T`, `T / &T`, `&T / T`, and `&T / &T`.
    assert_eq!(lhs / rhs, expected_div);
    assert_eq!(lhs / &rhs, expected_div);
    assert_eq!(&lhs / rhs, expected_div);
    assert_eq!(&lhs / &rhs, expected_div);

    // Remainder: `T % T`, `T % &T`, `&T % T`, and `&T % &T`.
    assert_eq!(lhs % rhs, expected_rem);
    assert_eq!(lhs % &rhs, expected_rem);
    assert_eq!(&lhs % rhs, expected_rem);
    assert_eq!(&lhs % &rhs, expected_rem);

    // Add-assign: `T += T`.
    let mut assigned_lhs = lhs;
    assigned_lhs += rhs;
    assert_eq!(assigned_lhs, expected_add);

    // Add-assign: `T += &T`.
    assigned_lhs = lhs;
    assigned_lhs += &rhs;
    assert_eq!(assigned_lhs, expected_add);

    // Subtract-assign: `T -= T`.
    assigned_lhs = lhs;
    assigned_lhs -= rhs;
    assert_eq!(assigned_lhs, expected_sub);

    // Subtract-assign: `T -= &T`.
    assigned_lhs = lhs;
    assigned_lhs -= &rhs;
    assert_eq!(assigned_lhs, expected_sub);

    // Multiply-assign: `T *= T`.
    assigned_lhs = lhs;
    assigned_lhs *= rhs;
    assert_eq!(assigned_lhs, expected_mul);

    // Multiply-assign: `T *= &T`.
    assigned_lhs = lhs;
    assigned_lhs *= &rhs;
    assert_eq!(assigned_lhs, expected_mul);

    // Divide-assign: `T /= T`.
    assigned_lhs = lhs;
    assigned_lhs /= rhs;
    assert_eq!(assigned_lhs, expected_div);

    // Divide-assign: `T /= &T`.
    assigned_lhs = lhs;
    assigned_lhs /= &rhs;
    assert_eq!(assigned_lhs, expected_div);

    // Remainder-assign: `T %= T`.
    let mut assigned_lhs = lhs;
    assigned_lhs %= rhs;
    assert_eq!(assigned_lhs, expected_rem);

    // Remainder-assign: `T %= &T`.
    assigned_lhs = lhs;
    assigned_lhs %= &rhs;
    assert_eq!(assigned_lhs, expected_rem);
}

/// Asserts that all required to/from conversions between [`f64`] and `T`, as defined by
/// [`crate::real_field::F64Interop`], are correctly implemented.
///
/// Definitions of operators are not checked by this function, and are instead checked by one of
/// the following functions:
///
/// * [`crate::real_field::assert_f64_rhs_ops`]
/// * [`crate::real_field::assert_f64_lhs_ops`]
///
/// # Arguments
///
/// * `value_t` - A value of type `T`.
/// * `value_f64` - A value of type `f64`.
/// * `expected_t_into_f64` - The expected result of converting `value_t` into `f64`.
/// * `expected_f64_into_t` - The expected result of converting `value_f64` into `T`.
///
/// # Panics
///
/// Panics if any of the assertions for [`From`] or [`Into`] conversions fail.
pub fn assert_f64_conversions<T: RealField>(
    value_t: T,
    value_f64: f64,
    expected_t_into_f64: f64,
    expected_f64_into_t: T,
) {
    // Check `From`.
    assert_eq!(T::from(value_f64), expected_f64_into_t);
    let actual_f64_from_t: f64 = value_t.into();
    assert_eq!(actual_f64_from_t.to_bits(), expected_t_into_f64.to_bits());

    // Check `Into`.
    let actual_f64_from_into: f64 = value_t.into();
    assert_eq!(
        actual_f64_from_into.to_bits(),
        expected_t_into_f64.to_bits()
    );
    let actual_t_from_into: T = value_f64.into();
    assert_eq!(actual_t_from_into, expected_f64_into_t);
}

/// Asserts that all forms of a type's `f64` interoperability operations (with `f64` as the
/// left-hand side), as defined by [`crate::real_field::F64LhsOps`], produce the expected results.
///
/// # Generic Arguments
///
/// * `T` - Type implementing [`crate::real_field::F64LhsOps`] to test.
///
/// # Arguments
///
/// * `lhs` - Left-hand side operand (`f64`).
/// * `rhs` - Right-hand side operand (`T`).
/// * `expected_eq` - Expected result of `lhs == rhs`.
/// * `expected_lt` - Expected result of `lhs < rhs`.
/// * `expected_le` - Expected result of `lhs <= rhs`.
/// * `expected_gt` - Expected result of `lhs > rhs`.
/// * `expected_ge` - Expected result of `lhs >= rhs`.
/// * `expected_add` - Expected result of `lhs + rhs`.
/// * `expected_sub` - Expected result of `lhs - rhs`.
/// * `expected_mul` - Expected result of `lhs * rhs`.
/// * `expected_div` - Expected result of `lhs / rhs`.
/// * `expected_rem` - Expected result of `lhs % rhs`.
///
/// # Panics
///
/// Panics if any of the `f64` interoperability operations on `lhs`/`rhs` don't match their
/// expected result.
#[allow(
    clippy::op_ref,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools
)]
pub fn assert_f64_lhs_ops<T: RealField>(
    lhs: f64,
    rhs: T,
    expected_eq: bool,
    expected_lt: bool,
    expected_le: bool,
    expected_gt: bool,
    expected_ge: bool,
    expected_add: T,
    expected_sub: T,
    expected_mul: T,
    expected_div: T,
    expected_rem: T,
) {
    // Equality: `f64 == T`.
    assert_eq!(lhs == rhs, expected_eq);
    assert_eq!(lhs != rhs, !expected_eq);

    // Ordering: `f64 < T`, `f64 <= T`, `f64 > T`, and `f64 >= T`.
    assert_eq!(lhs < rhs, expected_lt);
    assert_eq!(lhs <= rhs, expected_le);
    assert_eq!(lhs > rhs, expected_gt);
    assert_eq!(lhs >= rhs, expected_ge);

    // Addition: `f64 + T`, `f64 + &T`, `&f64 + T`, and `&f64 + &T`.
    assert_eq!((lhs + rhs), expected_add);
    assert_eq!((lhs + &rhs), expected_add);
    assert_eq!((&lhs + rhs), expected_add);
    assert_eq!((&lhs + &rhs), expected_add);

    // Subtraction: `f64 - T`, `f64 - &T`, `&f64 - T`, and `&f64 - &T`.
    assert_eq!((lhs - rhs), expected_sub);
    assert_eq!((lhs - &rhs), expected_sub);
    assert_eq!((&lhs - rhs), expected_sub);
    assert_eq!((&lhs - &rhs), expected_sub);

    // Multiplication: `f64 * T`, `f64 * &T`, `&f64 * T`, and `&f64 * &T`.
    assert_eq!((lhs * rhs), expected_mul);
    assert_eq!((lhs * &rhs), expected_mul);
    assert_eq!((&lhs * rhs), expected_mul);
    assert_eq!((&lhs * &rhs), expected_mul);

    // Division: `f64 / T`, `f64 / &T`, `&f64 / T`, and `&f64 / &T`.
    assert_eq!((lhs / rhs), expected_div);
    assert_eq!((lhs / &rhs), expected_div);
    assert_eq!((&lhs / rhs), expected_div);
    assert_eq!((&lhs / &rhs), expected_div);

    // Remainder: `f64 % T`, `f64 % &T`, `&f64 % T`, and `&f64 % &T`.
    assert_eq!((lhs % rhs), expected_rem);
    assert_eq!((lhs % &rhs), expected_rem);
    assert_eq!((&lhs % rhs), expected_rem);
    assert_eq!((&lhs % &rhs), expected_rem);
}

/// Asserts that all forms of a type's `f64` interoperability operations (with `f64` as the
/// right-hand side), as defined by [`crate::real_field::F64RhsOps`], produce the expected results.
///
/// # Generic Arguments
///
/// * `T` - Type implementing [`crate::real_field::F64RhsOps`] to test.
///
/// # Arguments
///
/// * `lhs` - Left-hand side operand (`T`).
/// * `rhs` - Right-hand side operand (`f64`).
/// * `expected_eq` - Expected result of `lhs == rhs`.
/// * `expected_lt` - Expected result of `lhs < rhs`.
/// * `expected_le` - Expected result of `lhs <= rhs`.
/// * `expected_gt` - Expected result of `lhs > rhs`.
/// * `expected_ge` - Expected result of `lhs >= rhs`.
/// * `expected_add` - Expected result of `lhs + rhs`.
/// * `expected_sub` - Expected result of `lhs - rhs`.
/// * `expected_mul` - Expected result of `lhs * rhs`.
/// * `expected_div` - Expected result of `lhs / rhs`.
/// * `expected_rem` - Expected result of `lhs % rhs`.
///
/// # Panics
///
/// Panics if any of the `f64` interoperability operations on `lhs`/`rhs` don't match their
/// expected result.
#[allow(
    clippy::op_ref,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools
)]
pub fn assert_f64_rhs_ops<T: RealField>(
    lhs: T,
    rhs: f64,
    expected_eq: bool,
    expected_lt: bool,
    expected_le: bool,
    expected_gt: bool,
    expected_ge: bool,
    expected_add: T,
    expected_sub: T,
    expected_mul: T,
    expected_div: T,
    expected_rem: T,
) {
    // Equality: `T == f64`.
    assert_eq!(lhs == rhs, expected_eq);
    assert_eq!(lhs != rhs, !expected_eq);

    // Ordering: `T < f64`, `T <= f64`, `T > f64`, and `T >= f64`.
    assert_eq!(lhs < rhs, expected_lt);
    assert_eq!(lhs <= rhs, expected_le);
    assert_eq!(lhs > rhs, expected_gt);
    assert_eq!(lhs >= rhs, expected_ge);

    // Addition: `T + f64`, `T + &f64`, `&T + f64`, and `&T + &f64`.
    assert_eq!((lhs + rhs), expected_add);
    assert_eq!((lhs + &rhs), expected_add);
    assert_eq!((&lhs + rhs), expected_add);
    assert_eq!((&lhs + &rhs), expected_add);

    // Subtraction: `T - f64`, `T - &f64`, `&T - f64`, and `&T - &f64`.
    assert_eq!((lhs - rhs), expected_sub);
    assert_eq!((lhs - &rhs), expected_sub);
    assert_eq!((&lhs - rhs), expected_sub);
    assert_eq!((&lhs - &rhs), expected_sub);

    // Multiplication: `T * f64`, `T * &f64`, `&T * f64`, and `&T * &f64`.
    assert_eq!((lhs * rhs), expected_mul);
    assert_eq!((lhs * &rhs), expected_mul);
    assert_eq!((&lhs * rhs), expected_mul);
    assert_eq!((&lhs * &rhs), expected_mul);

    // Division: `T / f64`, `T / &f64`, `&T / f64`, and `&T / &f64`.
    assert_eq!((lhs / rhs), expected_div);
    assert_eq!((lhs / &rhs), expected_div);
    assert_eq!((&lhs / rhs), expected_div);
    assert_eq!((&lhs / &rhs), expected_div);

    // Remainder: `T % f64`, `T % &f64`, `&T % f64`, and `&T % &f64`.
    assert_eq!((lhs % rhs), expected_rem);
    assert_eq!((lhs % &rhs), expected_rem);
    assert_eq!((&lhs % rhs), expected_rem);
    assert_eq!((&lhs % &rhs), expected_rem);

    // Add-assign: `T += f64`.
    let mut assigned_lhs = lhs;
    assigned_lhs += rhs;
    assert_eq!(assigned_lhs, expected_add);

    // Add-assign: `T += &f64`.
    assigned_lhs = lhs;
    assigned_lhs += &rhs;
    assert_eq!(assigned_lhs, expected_add);

    // Subtract-assign: `T -= f64`.
    assigned_lhs = lhs;
    assigned_lhs -= rhs;
    assert_eq!(assigned_lhs, expected_sub);

    // Subtract-assign: `T -= &f64`.
    assigned_lhs = lhs;
    assigned_lhs -= &rhs;
    assert_eq!(assigned_lhs, expected_sub);

    // Multiply-assign: `T *= f64`.
    assigned_lhs = lhs;
    assigned_lhs *= rhs;
    assert_eq!(assigned_lhs, expected_mul);

    // Multiply-assign: `T *= &f64`.
    assigned_lhs = lhs;
    assigned_lhs *= &rhs;
    assert_eq!(assigned_lhs, expected_mul);

    // Divide-assign: `T /= f64`.
    assigned_lhs = lhs;
    assigned_lhs /= rhs;
    assert_eq!(assigned_lhs, expected_div);

    // Divide-assign: `T /= &f64`.
    assigned_lhs = lhs;
    assigned_lhs /= &rhs;
    assert_eq!(assigned_lhs, expected_div);

    // Remainder-assign: `T %= f64`.
    assigned_lhs = lhs;
    assigned_lhs %= rhs;
    assert_eq!(assigned_lhs, expected_rem);

    // Remainder-assign: `T %= &f64`.
    assigned_lhs = lhs;
    assigned_lhs %= &rhs;
    assert_eq!(assigned_lhs, expected_rem);
}

#[cfg(test)]
mod tests {
    use super::{assert_f64_lhs_ops, assert_f64_rhs_ops};

    #[test]
    fn test_f64_lhs_comparisons() {
        assert_f64_lhs_ops(
            1.0, 2.0, false, true, true, false, false, 3.0, -1.0, 2.0, 0.5, 1.0,
        );
        assert_f64_lhs_ops(
            2.0, 2.0, true, false, true, false, true, 4.0, 0.0, 4.0, 1.0, 0.0,
        );
        assert_f64_lhs_ops(
            3.0, 2.0, false, false, false, true, true, 5.0, 1.0, 6.0, 1.5, 1.0,
        );
    }

    #[test]
    fn test_f64_rhs_comparisons() {
        assert_f64_rhs_ops(
            1.0, 2.0, false, true, true, false, false, 3.0, -1.0, 2.0, 0.5, 1.0,
        );
        assert_f64_rhs_ops(
            2.0, 2.0, true, false, true, false, true, 4.0, 0.0, 4.0, 1.0, 0.0,
        );
        assert_f64_rhs_ops(
            3.0, 2.0, false, false, false, true, true, 5.0, 1.0, 6.0, 1.5, 1.0,
        );
    }
}
