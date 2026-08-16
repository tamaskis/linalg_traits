use crate::verify_trait_implemented;
use imply_hack::Imply;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

const _: bool = verify_trait_implemented!(f64: F64RhsOps);

/// Operations between `T` (LHS) and `f64` (RHS).
///
/// # Required Operations
///
/// ## Addition
///
/// * `T + f64`
/// * `T + &f64`
/// * `&T + f64`
/// * `&T + &f64`
///
/// ## Addition-Assignment
///
/// * `T += f64`
/// * `T += &f64`
///
/// ## Subtraction
///
/// * `T - f64`
/// * `T - &f64`
/// * `&T - f64`
/// * `&T - &f64`
///
/// ## Subtraction-Assignment
///
/// * `T -= f64`
/// * `T -= &f64`
///
/// ## Multiplication
///
/// * `T * f64`
/// * `T * &f64`
/// * `&T * f64`
/// * `&T * &f64`
///
/// ## Multiplication-Assignment
///
/// * `T *= f64`
/// * `T *= &f64`
///
/// ## Division
///
/// * `T / f64`
/// * `T / &f64`
/// * `&T / f64`
/// * `&T / &f64`
///
/// ## Division-Assignment
///
/// * `T /= f64`
/// * `T /= &f64`
///
/// ## Remainder
///
/// * `T % f64`
/// * `T % &f64`
/// * `&T % f64`
/// * `&T % &f64`
///
/// ## Remainder-Assignment
///
/// * `T %= f64`
/// * `T %= &f64`
pub trait F64RhsOps:
    // Binary operations with an owned right-hand side: `T op f64`.
    Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + Rem<f64, Output = Self>
    // Assignment operations on an owned value: `T op= f64`.
    + AddAssign<f64>
    + SubAssign<f64>
    + MulAssign<f64>
    + DivAssign<f64>
    + RemAssign<f64>
    // Binary operations with an owned left-hand side and borrowed right-hand side: `T op &f64`.
    + for<'a> Add<&'a f64, Output = Self>
    + for<'a> Sub<&'a f64, Output = Self>
    + for<'a> Mul<&'a f64, Output = Self>
    + for<'a> Div<&'a f64, Output = Self>
    + for<'a> Rem<&'a f64, Output = Self>
    // Assignment operations involving a borrowed value: `T op= &f64`.
    + for<'a> AddAssign<&'a f64>
    + for<'a> SubAssign<&'a f64>
    + for<'a> MulAssign<&'a f64>
    + for<'a> DivAssign<&'a f64>
    + for<'a> RemAssign<&'a f64>
    // Establishes the corresponding `&T op f64` and `&T op &f64` forms.
    + for<'a> Imply<
        &'a Self,
        Is: Add<f64, Output = Self>
            + Sub<f64, Output = Self>
            + Mul<f64, Output = Self>
            + Div<f64, Output = Self>
            + Rem<f64, Output = Self>
            + Add<&'a f64, Output = Self>
            + Sub<&'a f64, Output = Self>
            + Mul<&'a f64, Output = Self>
            + Div<&'a f64, Output = Self>
            + Rem<&'a f64, Output = Self>,
    >
{
}

// Blanket implementation.
impl<T> F64RhsOps for T
where
    T: Add<f64, Output = T>
        + Sub<f64, Output = T>
        + Mul<f64, Output = T>
        + Div<f64, Output = T>
        + Rem<f64, Output = T>
        // Assignment operations on an owned value: `T op= f64`.
        + AddAssign<f64>
        + SubAssign<f64>
        + MulAssign<f64>
        + DivAssign<f64>
        + RemAssign<f64>,
    // Binary operations with an owned left-hand side and borrowed right-hand side: `T op &f64`.
    for<'a> T: Add<&'a f64, Output = T>
        + Sub<&'a f64, Output = T>
        + Mul<&'a f64, Output = T>
        + Div<&'a f64, Output = T>
        + Rem<&'a f64, Output = T>
        // Assignment operations involving a borrowed value: `T op= &f64`.
        + AddAssign<&'a f64>
        + SubAssign<&'a f64>
        + MulAssign<&'a f64>
        + DivAssign<&'a f64>
        + RemAssign<&'a f64>,
    // Implements the corresponding `&T op f64` and `&T op &f64` forms.
    for<'a> T: Imply<
            &'a T,
            Is: Add<f64, Output = T>
                    + Sub<f64, Output = T>
                    + Mul<f64, Output = T>
                    + Div<f64, Output = T>
                    + Rem<f64, Output = T>
                    + Add<&'a f64, Output = T>
                    + Sub<&'a f64, Output = T>
                    + Mul<&'a f64, Output = T>
                    + Div<&'a f64, Output = T>
                    + Rem<&'a f64, Output = T>,
        >,
{
}

/// Asserts that all forms of a type's `f64` interoperability operations (with `f64` as the
/// right-hand side) produce the expected results.
///
/// # Generic Arguments
///
/// * `T` - Type implementing [`F64RhsOps`] to test.
///
/// # Arguments
///
/// * `lhs` - Left-hand side operand (`T`).
/// * `rhs` - Right-hand side operand (`f64`).
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
/// are not required by [`F64RhsOps`], but they are generally already implemented for types that
/// implement this trait. If `T` does not implement these traits, this function will fail to
/// compile.
#[allow(clippy::op_ref, clippy::too_many_arguments)]
pub fn assert_f64_rhs_ops<T: F64RhsOps + Copy + Debug + PartialEq>(
    lhs: T,
    rhs: f64,
    expected_add: T,
    expected_sub: T,
    expected_mul: T,
    expected_div: T,
    expected_rem: T,
) {
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
