/// Additional requirements to add on top of [`crate::real_field::RealFieldBase`] to make a type
/// compatible with [`faer`] when the `faer` feature is enabled.
///
/// When the `faer` feature is _NOT_ enabled, this trait is simply an empty marker trait.
///
/// # Blanket Implementations
///
/// * When the `faer` feature is disabled, this trait is automatically implemented for any type that
///   implements [`crate::real_field::RealFieldBase`].
/// * When the `faer` feature is enabled, this trait is automatically implemented for any type
///   that implements [`crate::real_field::RealFieldBase`] and [`faer_traits::RealField`](https://docs.rs/faer-traits/latest/faer_traits/trait.RealField.html).
///     * If a type already implements [`crate::real_field::RealFieldBase`], the
///       [`faer_traits::RealField`] trait can be implemented using [`crate::impl_faer_traits_real_field`].
#[cfg(feature = "faer")]
pub trait RealFieldFaer: faer_traits::RealField {}
#[allow(missing_docs)]
#[cfg(not(feature = "faer"))]
pub trait RealFieldFaer {}

// Blanket implementations.
#[cfg(feature = "faer")]
impl<T> RealFieldFaer for T where T: crate::real_field::RealFieldBase + faer_traits::RealField {}
#[allow(missing_docs)]
#[cfg(not(feature = "faer"))]
impl<T> RealFieldFaer for T where T: crate::real_field::RealFieldBase {}
