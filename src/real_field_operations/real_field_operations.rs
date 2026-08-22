use crate::real_field_operations::f64_interop::f64_interop::F64Interop;
use crate::real_field_operations::real_field_operations_ref_ops::RealFieldOperationsRefOps;
use crate::real_field_operations::real_field_operations_rem::RealFieldOperationsRem;
use std::fmt::Debug;

/// Trait defining all arithmetic operations that need to be supported by real fields.
///
/// This trait extends far beyond what is required by popular linear algebra crates (e.g.
/// [`nalgebra`], [`faer`]) and is instead geared towards making generic real numbers as ergonomic
/// as possible (including being highly interoperable with [`f64`], which is the de facto standard
/// floating point representation). This is especially useful for fully-generic implementations that
/// are required for forward-mode automatic differentiation (see the
/// [`numdiff` crate documentation](https://docs.rs/numdiff/latest/numdiff/)).
///
/// # Supported Operators
///
/// These operators are supported by the [`RealFieldOperations`] trait and can be substituted in for
/// "`op`" in all the definitions below:
///
/// `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`
///
/// # Supported Operations
///
/// ## Unary negation
/// * `-T`
/// * `-&T`
///
/// ## Binary operations between two `T`'s
///
/// * `T op T`
/// * `T op &T`
/// * `&T op T`
/// * `&T op &T`
///
/// ## Binary-assignment operations between two `T`'s
///
/// * `T op= T`
/// * `T op= &T`
///
/// ## Binary operations between `T` and `f64` with `T` on the left-hand side
///
/// * `T op f64`
/// * `T op &f64`
/// * `&T op f64`
/// * `&T op &f64`
///
/// ## Binary-assignment operations between `T` and `f64` with `T` on the left-hand side
///
/// * `T op= f64`
/// * `T op= &f64`
///
/// ## Binary operations between `f64` and `T` with `f64` on the left-hand side
///
/// * `f64 op T`
/// * `f64 op &T`
/// * `&f64 op T`
/// * `&f64 op &T`
///
/// ## Binary-assignment operations between `T` and `f64` with `f64` on the left-hand side
///
/// **These operations are NOT implemented.** This is because the primary use case of
/// [`crate::RealField`] (for which [`RealFieldOperations`] is a supertrait) is for use in a generic
/// autodiff backend (e.g. [`numdiff` crate documentation](https://docs.rs/numdiff/latest/numdiff/))
/// where the types that will be substituted in for `T` are typically more complex types (e.g.
/// [`numdiff::Dual`](https://docs.rs/numdiff/latest/numdiff/struct.Dual.html),
/// [`numdiff::HyperDual`](https://docs.rs/numdiff/latest/numdiff/struct.HyperDual.html)). It is
/// perfectly safe "up-casting" from an `f64` to a one of these types (no information is lost), but
/// the reverse is not true (i.e., information will be lost). This is why binary-assignment
/// operations with an `f64` on the left-hand side are not implemented, because doing something like
/// `1.0 += Dual::new(2.0, 3.0)` would lose information.
///
/// # How To Implement
///
/// To implement this trait for a type `T`:
///
/// 1. Implement [`crate::real_field_operations::BaseOperations`].
/// 2. Call [`crate::impl_real_field_operations!`] on the type `T`.
///
/// This trait definition is quite complex, and is further complicated by having to satisfy multiple
/// optional third party dependencies (e.g. see the [`RealFieldOperationsRefOps`] docs as an example
/// for where things get complicated). If someone really wanted to, they _could_ trace through the
/// extensive set of trait requirements and manually implement all of the required traits manually,
/// but implementing the minimal [`crate::real_field_operations::BaseOperations`] trait and then
/// letting [`crate::impl_real_field_operations!`] take care of the rest is far easier.
pub trait RealFieldOperations:
    RealFieldOperationsRefOps + RealFieldOperationsRem + PartialEq + PartialOrd + F64Interop
{
}

// Blanket implementation.
impl<T> RealFieldOperations for T where
    T: RealFieldOperationsRefOps + RealFieldOperationsRem + PartialEq + PartialOrd + F64Interop
{
}

/// Asserts that all arithmetic operation forms guaranteed by [`RealFieldOperations`] produce the
/// expected results.
///
/// # Generic Arguments
///
/// * `T` - Type implementing [`RealFieldOperations`] to test.
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
///
/// # Warning
///
/// This function assumes that `T` also implements `Copy` and `Debug`. If `T` does not implement
/// these traits, this function will fail to compile.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::fn_params_excessive_bools,
    clippy::op_ref
)]
pub fn assert_real_field_operations<T: RealFieldOperations + Copy + Debug>(
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
