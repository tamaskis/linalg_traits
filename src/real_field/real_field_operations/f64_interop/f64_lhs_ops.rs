use crate::verify_trait_implemented;
use imply_hack::Imply;
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
    // Equality comparison: `f64 == T`.
    PartialEq<T>
    // Ordering comparison: `f64 < T`.
    + PartialOrd<T>
    // Binary operations with an owned right-hand side: `f64 op T`.
    + Add<T, Output = T>
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
    f64: PartialEq<T>
        + PartialOrd<T>
        // Binary operations with an owned right-hand side: `f64 op T`.
        + Add<T, Output = T>
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
/// ## Equality
///
/// * `f64 == T`
/// * `f64 != T`
///
/// ## Ordering
///
/// * `f64 < T`
/// * `f64 <= T`
/// * `f64 > T`
/// * `f64 >= T`
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
