use crate::verify_trait_implemented;
use imply_hack::Imply;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

const _: bool = verify_trait_implemented!(f64: F64RhsOps);

/// Operations between `T` (LHS) and `f64` (RHS).
///
/// # Required Operations
///
/// ## Equality
///
/// * `T == f64`
/// * `T != f64`
///
/// ## Ordering
///
/// * `T < f64`
/// * `T <= f64`
/// * `T > f64`
/// * `T >= f64`
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
    // Equality comparison: `T == f64`.
    PartialEq<f64>
    // Ordering comparison: `T < f64`.
    + PartialOrd<f64>
    // Binary operations with an owned right-hand side: `T op f64`.
    + Add<f64, Output = Self>
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
    T: PartialEq<f64>
        + PartialOrd<f64>
        + Add<f64, Output = T>
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
