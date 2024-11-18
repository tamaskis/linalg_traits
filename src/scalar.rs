//! Definition and implementions of the [`Scalar`] trait.

use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Scalar type.
///
/// # Interoperability with [`f64`]s.
///
/// We enforce that scalar types be interoperable with [`f64`]s. Some common differentiation
/// methods, notably forward-mode automatic differentation and complex-step differentiation, rely
/// on replacing real numbers with a custom type of number that has its own arithmetic (dual numbers
/// for forward-mode automatic differentiation, complex numbers for complex-step differentiation).
/// Forcing scalars to have this interoperability with [`f64`]s built-in helps enable downstream
/// crates to write functions in way that can be used with both plain [`f64`]s for most use cases,
/// and with custom types when the functions need to be differentiated.
///
/// Additionally, we chose to restrict this interoperability to be with [`f64`]s since
/// double-precision floating point numbers are the de facto standard for numerical computations.
///
/// # Note
///
/// [`nalgebra::Complex`] does not satisfy the the [`Scalar`] trait because it does not implement
/// the [`PartialOrd`] trait.
pub trait Scalar:
    // Arithmetic operators with itself.
    Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    // Arithmetic-assignment operators with itself.
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    // Arithmetic operators with f64.
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    // Arithmetic-assignment operators with f64.
    + AddAssign<f64>
    + SubAssign<f64>
    + MulAssign<f64>
    + DivAssign<f64>
    // Needed for construction.
    + Zero
    // Other "standard" traits for primitive numeric types.
    + Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Debug
    // Type must be defined at compile time.
    + 'static
{
}

impl<T> Scalar for T where
    T: Add<Self, Output = Self>
        + Sub<Self, Output = Self>
        + Mul<Self, Output = Self>
        + Div<Self, Output = Self>
        + AddAssign<Self>
        + SubAssign<Self>
        + MulAssign<Self>
        + DivAssign<Self>
        + Add<f64, Output = Self>
        + Sub<f64, Output = Self>
        + Mul<f64, Output = Self>
        + Div<f64, Output = Self>
        + AddAssign<f64>
        + SubAssign<f64>
        + MulAssign<f64>
        + DivAssign<f64>
        + Zero
        + Copy
        + Clone
        + PartialEq
        + PartialOrd
        + Debug
        + 'static
{
}
