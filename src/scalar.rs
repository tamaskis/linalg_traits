use num_traits::Float;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

/// Trait defining a generic scalar type.
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
pub trait ScalarBase:
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
    + RemAssign<f64>
    // Conversion to/from f64.
    + From<f64>
    + Into<f64>
    // Debug printing.
    + Debug
    // Type must be defined at compile time.
    + 'static
{
    /// Construct an instance of this scalar from an [`f64`].
    /// 
    /// # Arguments
    /// 
    /// * `x` - An [`f64`].
    /// 
    /// # Return
    /// 
    /// An instance of this scalar type constructed from an [`f64`].
    #[must_use]
    fn new(x: f64) -> Self {
        <Self as From<f64>>::from(x)
    }
}

impl<T> ScalarBase for T where
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
        + From<f64>
        + Into<f64>
        + Debug
        + 'static
{
}

/// Additional requirements when the `ndarray` feature is enabled.
#[cfg(feature = "ndarray")]
pub trait NdarrayScalar: ndarray::ScalarOperand + ndarray::LinalgScalar {}

#[cfg(feature = "ndarray")]
impl<T> NdarrayScalar for T where T: ndarray::ScalarOperand + ndarray::LinalgScalar {}

/// No additional requirements when `ndarray` is disabled.
#[cfg(not(feature = "ndarray"))]
pub trait NdarrayScalar {}

#[cfg(not(feature = "ndarray"))]
impl<T> NdarrayScalar for T {}

/// Additional requirements when the `faer` feature is enabled.
#[cfg(feature = "faer")]
pub trait FaerScalar: faer_traits::RealField {}

#[cfg(feature = "faer")]
impl<T> FaerScalar for T where T: faer_traits::RealField {}

/// No additional requirements when `faer` is disabled.
#[cfg(not(feature = "faer"))]
pub trait FaerScalar {}

#[cfg(not(feature = "faer"))]
impl<T> FaerScalar for T {}

/// Trait defining a generic scalar type.
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
/// the [`PartialOrd`] trait, which is required by the [`Float`] trait that is in turn required by
/// the [`ScalarBase`] trait that is a supertrait of [`Scalar`].
pub trait Scalar: ScalarBase + NdarrayScalar + FaerScalar {
    /// Construct an instance of this scalar from an `f64`.
    #[must_use]
    fn new(x: f64) -> Self {
        <Self as ScalarBase>::new(x)
    }
}

impl<T> Scalar for T where T: ScalarBase + NdarrayScalar + FaerScalar {}
