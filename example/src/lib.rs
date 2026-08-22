//! A fully custom (i.e. not backed by [`nalgebra`], [`ndarray`], or [`faer`]) scalar type used to
//! exercise `linalg-traits` in unit tests.
//!
//! This crate is a workspace member of `linalg-traits` that is never published; it exists solely
//! as a dev-dependency used in `linalg-traits`'s unit tests.

// Module declarations.
pub mod base_operations;
pub mod faer_impl;
pub mod my_scalar;
pub mod nalgebra_impl;
pub mod num_traits_impl;
pub mod real_field_base;
pub mod real_field_operations;

// Re-exports.
pub use my_scalar::MyScalar;
