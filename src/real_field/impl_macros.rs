// Module declarations.
#[cfg(feature = "faer")]
pub(crate) mod impl_faer_traits_real_field;
#[cfg(feature = "nalgebra")]
pub(crate) mod impl_nalgebra_real_field;
#[cfg(feature = "nalgebra")]
pub(crate) mod impl_num_traits_from_primitive;
#[cfg(any(feature = "faer", feature = "nalgebra", feature = "ndarray"))]
pub(crate) mod impl_num_traits_num;
#[cfg(feature = "nalgebra")]
pub(crate) mod impl_num_traits_signed;
pub(crate) mod impl_real_field;
pub(crate) mod impl_real_field_operations;
pub(crate) mod impl_to_from_f64;
