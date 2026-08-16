#[cfg(not(feature = "faer"))]
use imply_hack::Imply;
#[cfg(not(feature = "faer"))]
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// See `RealFieldOperationsRefOps` below.
///
/// # License
///
/// Adapted from [`faer_traits`], licensed under the [MIT License](https://codeberg.org/sarah-quinones/faer/src/branch/main/LICENSE).
/// Copyright (c) 2026 sarah quiñones el kazdadi
///
/// A copy of the original license is included in this repository
/// [**here**](https://github.com/tamaskis/linalg_traits/THIRD_PARTY_LICENSES/faer/LICENSE).
#[cfg(not(feature = "faer"))]
pub trait RefOps:
    Sized
    // Unary operations on an owned value.
    + Neg<Output = Self>
    // Binary operations with an owned right-hand side: `T op T`.
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    // Assignment operations on an owned value: `op T` and `T op= T`.
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    // Binary operations with an owned left-hand side and borrowed right-hand side: `T op &T`.
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    // Assignment operations involving a borrowed value: `T op= &T`.
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
    // Establishes the corresponding `&T op T` and `&T op &T` forms.
    + for<'a> Imply<
		&'a Self,
		Is: Neg<Output = Self>
		        + Add<Output = Self>
		        + Sub<Output = Self>
		        + Mul<Output = Self>
		        + Div<Output = Self>
		        + Add<Self, Output = Self>
		        + Sub<Self, Output = Self>
		        + Mul<Self, Output = Self>
		        + Div<Self, Output = Self>,
	>
{
}
#[cfg(not(feature = "faer"))]
impl<T> RefOps for T
where
    T: Neg<Output = Self>
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + for<'a> Add<&'a T, Output = Self>
        + for<'a> Sub<&'a T, Output = Self>
        + for<'a> Mul<&'a T, Output = Self>
        + for<'a> Div<&'a T, Output = Self>
        + for<'a> AddAssign<&'a T>
        + for<'a> SubAssign<&'a T>
        + for<'a> MulAssign<&'a T>
        + for<'a> DivAssign<&'a T>,
    for<'a> T: Imply<
            &'a T,
            Is: Neg<Output = T>
                    + Add<Output = T>
                    + Sub<Output = T>
                    + Mul<Output = T>
                    + Div<Output = T>
                    + Add<T, Output = T>
                    + Sub<T, Output = T>
                    + Mul<T, Output = T>
                    + Div<T, Output = T>,
        >,
{
}

/// Subset of operations required by [`crate::real_field::RealFieldOperations`] that are defined by
/// [`faer_traits::RefOps`].
///
/// * When the `faer` feature is active, this trait is just a marker trait for any type that
///   implements [`faer_traits::RefOps`].
/// * When the `faer` feature is not active, this trait is a marker trait for any type that
///   implements the same set of operations as [`faer_traits::RefOps`].
///    * This is done by having an internal `linalg-traits` redefinition of the
///      [`faer_traits::RefOps`] trait.
///
/// # Why this is a separate trait
///
/// [`faer_traits::RefOps`] cannot be used directly because [`faer_traits`] is an optional
/// dependency of this crate. Making it a required dependency would impose [`faer_traits`]'s
/// dependency tree on users who do not use the [`faer`] backend. This local trait therefore
/// provides the same contract without making [`faer`] part of the core API.
///
/// The contract must remain effectively identical to [`faer_traits::RefOps`]. In particular, it
/// cannot add [`std::ops::Rem`], [`PartialEq`], [`PartialOrd`], or other supertrait bounds. Backend
/// traits such as [`faer_traits::RealField`] inherit [`faer_traits::RefOps`]; adding requirements
/// here would make generic implementations of this crate's real-field traits require bounds that
/// [`faer`]'s generic contracts do not provide. The compiler then cannot prove the associated `&T`
/// operator outputs while checking those implementations.
///
/// The additional operations needed by this crate are consequently kept in
/// [`crate::real_field::RealFieldOperationsRem`] and [`crate::real_field::RealFieldOperations`]
/// instead.
///
/// /// ## Deconflicting with [`faer_traits::RefOps`]
///
/// When the `faer` feature is enabled, this trait is defined as a plain alias for
/// [`faer_traits::RefOps`] rather than re-declaring the same `Imply`-derived `&T op T` bounds
/// itself. [`faer_traits::RealField`] (required by
/// [`RealFieldFaer`](crate::real_field::RealFieldFaer)) also carries [`faer_traits::RefOps`] as a
/// supertrait. If this trait independently re-derived the same `&T op T` bounds via its own
/// [`imply_hack::Imply`] bound instead of deferring to [`faer_traits::RefOps`], the compiler would
/// see two distinct supertrait-elaborated sources for the identical `&T op T` obligations and fail
/// with an ambiguity error when normalizing the ssociated `Output` types. Deferring entirely to
/// [`faer_traits::RefOps`] keeps a single source of truth for those bounds.
///
/// # License
///
/// Adapted from [`faer_traits`], licensed under the [MIT License](https://codeberg.org/sarah-quinones/faer/src/branch/main/LICENSE).
/// Copyright (c) 2026 sarah quiñones el kazdadi
///
/// A copy of the original license is included in this repository
/// [**here**](https://github.com/tamaskis/linalg_traits/THIRD_PARTY_LICENSES/faer/LICENSE).
#[cfg(feature = "faer")]
pub trait RealFieldOperationsRefOps: faer_traits::RefOps {}
#[allow(missing_docs)]
#[cfg(not(feature = "faer"))]
pub trait RealFieldOperationsRefOps: RefOps {}

// Blanket implementations.
#[cfg(feature = "faer")]
impl<T> RealFieldOperationsRefOps for T where T: faer_traits::RefOps {}
#[cfg(not(feature = "faer"))]
impl<T> RealFieldOperationsRefOps for T where T: RefOps {}
