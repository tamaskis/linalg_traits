/// Additional requirements to add on top of [`crate::real_field::RealFieldBase`] to make a type
/// compatible with [`ndarray`] when the `ndarray` feature is enabled.
///
/// When the `ndarray` feature is _NOT_ enabled, this trait is simply an empty marker trait.
///
/// # Blanket Implementations
///
/// * When the `ndarray` feature is disabled, this trait is automatically implemented for any type
///   that implements [`crate::real_field::RealFieldBase`].
/// * When the `ndarray` feature is enabled, this trait is automatically implemented for any type
///   that implements [`crate::real_field::RealFieldBase`], [`ndarray::ScalarOperand`], and
///   [`ndarray::LinalgScalar`].
#[cfg(feature = "ndarray")]
pub trait RealFieldNdarray: ndarray::ScalarOperand + ndarray::LinalgScalar {}
#[allow(missing_docs)]
#[cfg(not(feature = "ndarray"))]
pub trait RealFieldNdarray {}

// Blanket implementations.
#[cfg(feature = "ndarray")]
impl<T> RealFieldNdarray for T where
    T: crate::real_field::RealFieldBase + ndarray::ScalarOperand + ndarray::LinalgScalar
{
}
#[allow(missing_docs)]
#[cfg(not(feature = "ndarray"))]
impl<T> RealFieldNdarray for T where T: crate::real_field::RealFieldBase {}
