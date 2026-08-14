// Module declarations.
#[cfg(feature = "faer")]
pub(crate) mod faer_col;

#[cfg(feature = "nalgebra")]
pub(crate) mod nalgebra_dvector;

#[cfg(feature = "nalgebra")]
pub(crate) mod nalgebra_svector;

#[cfg(feature = "ndarray")]
pub(crate) mod ndarray_array1;

pub(crate) mod vec;
pub(crate) mod vector_trait;
