//! Defines the [`crate::RealField`] trait.

// Module declarations.
pub(crate) mod base;
pub(crate) mod impl_macros;
#[allow(clippy::module_inception)]
pub(crate) mod real_field;
pub(crate) mod real_field_base;
pub(crate) mod real_field_f64;
pub(crate) mod real_field_faer;
pub(crate) mod real_field_nalgebra;
pub(crate) mod real_field_ndarray;
pub(crate) mod real_field_operations;
pub(crate) mod testing;

// Re-exports.
pub use base::Base;
pub use real_field_base::RealFieldBase;
pub use real_field_faer::RealFieldFaer;
pub use real_field_nalgebra::RealFieldNalgebra;
pub use real_field_ndarray::RealFieldNdarray;
pub use real_field_operations::f64_interop::f64_interop::F64Interop;
pub use real_field_operations::f64_interop::f64_lhs_ops::{F64LhsOps, F64LhsOpsBase};
pub use real_field_operations::f64_interop::f64_rhs_ops::F64RhsOps;
pub use real_field_operations::real_field_operations::RealFieldOperations;
pub use real_field_operations::real_field_operations_ref_ops::RealFieldOperationsRefOps;
#[cfg(not(feature = "faer"))]
pub use real_field_operations::real_field_operations_ref_ops::RefOps;
pub use real_field_operations::real_field_operations_rem::RealFieldOperationsRem;
pub use testing::{
    assert_base, assert_f64_interop, assert_f64_lhs_ops, assert_f64_rhs_ops,
    assert_real_field_operations,
};
