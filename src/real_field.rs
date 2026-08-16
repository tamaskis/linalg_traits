//! Defines the [`crate::RealField`] trait.

// Module declarations.
pub(crate) mod base;
#[allow(clippy::module_inception)]
pub(crate) mod real_field;
pub(crate) mod real_field_base;
pub(crate) mod real_field_f64;
pub(crate) mod real_field_faer;
pub(crate) mod real_field_nalgebra;
pub(crate) mod real_field_ndarray;

// Re-exports.
pub use real_field_base::RealFieldBase;
pub use real_field_faer::RealFieldFaer;
pub use real_field_nalgebra::RealFieldNalgebra;
pub use real_field_ndarray::RealFieldNdarray;
