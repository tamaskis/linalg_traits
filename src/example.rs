// TODO: put this into a subcrate?

// Module declarations.
pub(crate) mod base_operations;
#[cfg(feature = "faer")]
pub(crate) mod faer_impl;
pub(crate) mod my_scalar;
#[cfg(feature = "nalgebra")]
pub(crate) mod nalgebra_impl;
pub(crate) mod num_traits_impl;
pub(crate) mod real_field_base;
pub(crate) mod real_field_operations;
