use num_traits::Float;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

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
    // Float trait from num-traits, providing:
    //  --> Copy
    //  --> PartialEq
    //  --> PartialOrd
    //  --> Add<Self, Output = Self>
    //  --> Sub<Self, Output = Self>
    //  --> Mul<Self, Output = Self>
    //  --> Div<Self, Output = Self>
    //  --> Rem<Self, Output = Self>
    //  --> Neg<Output = Self>
    //  --> Zero
    //  --> One
    //  --> Standard mathematical methods (i.e. most methods implement for f64s).
    Float
    // Arithmetic-assignment operators with itself.
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + RemAssign<Self>
    // Arithmetic operators with f64.
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + Rem<f64, Output = Self>
    // Arithmetic-assignment operators with f64.
    + AddAssign<f64>
    + SubAssign<f64>
    + MulAssign<f64>
    + DivAssign<f64>
    // Debug printing.
    + Debug
    // Type must be defined at compile time.
    + 'static
{
}

impl<T> Scalar for T where
    T: Float
        + AddAssign<Self>
        + SubAssign<Self>
        + MulAssign<Self>
        + DivAssign<Self>
        + RemAssign<Self>
        + Add<f64, Output = Self>
        + Sub<f64, Output = Self>
        + Mul<f64, Output = Self>
        + Div<f64, Output = Self>
        + Rem<f64, Output = Self>
        + AddAssign<f64>
        + SubAssign<f64>
        + MulAssign<f64>
        + DivAssign<f64>
        + RemAssign<f64>
        + Debug
        + 'static
{
}
