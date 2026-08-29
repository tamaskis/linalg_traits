/// Additional requirements to add on top of [`crate::real_field::RealFieldBase`] to make a type
/// compatible with [`nalgebra`] when the `nalgebra` feature is enabled.
///
/// When the `nalgebra` feature is _NOT_ enabled, this trait is simply an empty marker trait.
///
/// # Blanket Implementations
///
/// * When the `nalgebra` feature is disabled, this trait is automatically implemented for any type
///   that implements [`crate::real_field::RealFieldBase`].
/// * When the `nalgebra` feature is enabled, this trait is automatically implemented for any type
///   that implements [`crate::real_field::RealFieldBase`] and [`nalgebra::RealField`].
///     * If a type already implements [`crate::real_field::RealFieldBase`], the
///       [`nalgebra::RealField`](https://docs.rs/nalgebra/latest/nalgebra/trait.RealField.html)trait can be implemented using
///       [`crate::impl_nalgebra_real_field`].
#[cfg(feature = "nalgebra")]
pub trait RealFieldNalgebra: nalgebra::RealField {}
#[allow(missing_docs)]
#[cfg(not(feature = "nalgebra"))]
pub trait RealFieldNalgebra {}

// Blanket implementations.
#[cfg(feature = "nalgebra")]
impl<T> RealFieldNalgebra for T where T: crate::real_field::RealFieldBase + nalgebra::RealField {}
#[allow(missing_docs)]
#[cfg(not(feature = "nalgebra"))]
impl<T> RealFieldNalgebra for T where T: crate::real_field::RealFieldBase {}
