use crate::verify_trait_implemented;
use imply_hack::Imply;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Rem, Sub};

const _: bool = verify_trait_implemented!(f64: F64LhsOpsBase<f64>);

/// Operations between `f64` (LHS) and `T` (RHS).
///
/// See [`F64LhsOps`] for the full set of documentation on what is actually implemented here.
///
/// This is the "base" trait that is implemented for [`f64`]. [`F64LhsOps`] then uses a clever hack
/// to effectively include this as a trait bound on `T`, even though it's implemented for `f64` (the
/// less user-friendly alternative would be to always have to have a `where` clause constraining
/// `f64: F64LhsOpsBase<T>`).
pub trait F64LhsOpsBase<T>:
    // Binary operations with an owned right-hand side: `f64 op T`.
    Add<T, Output = T>
    + Sub<T, Output = T>
    + Mul<T, Output = T>
    + Div<T, Output = T>
    + Rem<T, Output = T>
    // Binary operations with an owned left-hand side and borrowed right-hand side: `f64 op &T`.
    + for<'a> Add<&'a T, Output = T>
    + for<'a> Sub<&'a T, Output = T>
    + for<'a> Mul<&'a T, Output = T>
    + for<'a> Div<&'a T, Output = T>
    + for<'a> Rem<&'a T, Output = T>
    // Establishes the corresponding `&f64 op T` and `&f64 op &T` forms.
    + for<'a> Imply<
        &'a Self,
        Is: Add<T, Output = T>
                + Sub<T, Output = T>
                + Mul<T, Output = T>
                + Div<T, Output = T>
                + Rem<T, Output = T>
                + for<'b> Add<&'b T, Output = T>
                + for<'b> Sub<&'b T, Output = T>
                + for<'b> Mul<&'b T, Output = T>
                + for<'b> Div<&'b T, Output = T>
                + for<'b> Rem<&'b T, Output = T>,
    >
{
}

// Blanket implementation.
impl<T> F64LhsOpsBase<T> for f64
where
    // Binary operations with an owned right-hand side: `f64 op T`.
    f64: Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Rem<T, Output = T>
        // Binary operations with an owned left-hand side and borrowed right-hand side: `f64 op &T`.
        + for<'a> Add<&'a T, Output = T>
        + for<'a> Sub<&'a T, Output = T>
        + for<'a> Mul<&'a T, Output = T>
        + for<'a> Div<&'a T, Output = T>
        + for<'a> Rem<&'a T, Output = T>,
    // Implements the corresponding `&f64 op T` and `&f64 op &T` forms.
    for<'a> f64: Imply<
            &'a f64,
            Is: Add<T, Output = T>
                    + Sub<T, Output = T>
                    + Mul<T, Output = T>
                    + Div<T, Output = T>
                    + Rem<T, Output = T>
                    + for<'b> Add<&'b T, Output = T>
                    + for<'b> Sub<&'b T, Output = T>
                    + for<'b> Mul<&'b T, Output = T>
                    + for<'b> Div<&'b T, Output = T>
                    + for<'b> Rem<&'b T, Output = T>,
        >,
{
}

/// Operations between `f64` (LHS) and `T` (RHS).
///
/// # Required Operations
///
/// ## Addition
///
/// * `f64 + T`
/// * `f64 + &T`
/// * `&f64 + T`
/// * `&f64 + &T`
///
/// ## Subtraction
///
/// * `f64 - T`
/// * `f64 - &T`
/// * `&f64 - T`
/// * `&f64 - &T`
///
/// ## Multiplication
///
/// * `f64 * T`
/// * `f64 * &T`
/// * `&f64 * T`
/// * `&f64 * &T`
///
/// ## Division
///
/// * `f64 / T`
/// * `f64 / &T`
/// * `&f64 / T`
/// * `&f64 / &T`
///
/// ## Remainder
///
/// * `f64 % T`
/// * `f64 % &T`
/// * `&f64 % T`
/// * `&f64 % &T`
pub trait F64LhsOps: Sized + Imply<f64, Is: F64LhsOpsBase<Self>> {}

// Blanket implementation.
impl<T> F64LhsOps for T where f64: F64LhsOpsBase<T> {}

/// Asserts that all forms of a type's `f64` interoperability operations (with `f64` as the
/// left-hand side) produce the expected results.
///
/// # Generic Arguments
///
/// * `T` - Type implementing [`F64LhsOps`] to test.
///
/// # Arguments
///
/// * `lhs` - Left-hand side operand (`f64`).
/// * `rhs` - Right-hand side operand (`T`).
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
///
/// # Warning
///
/// This function assumes that `T` also implements `Copy`, `Debug`, and `PartialEq`. These traits
/// are not required by [`F64LhsOps`], but they are generally already implemented for types that
/// implement this trait. If `T` does not implement these traits, this function will fail to
/// compile.
#[allow(clippy::op_ref, clippy::too_many_arguments)]
pub fn assert_f64_lhs_ops<T: F64LhsOps + Copy + Debug + PartialEq>(
    lhs: f64,
    rhs: T,
    expected_add: T,
    expected_sub: T,
    expected_mul: T,
    expected_div: T,
    expected_rem: T,
) {
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
