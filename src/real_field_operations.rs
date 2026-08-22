//! Defines the [`crate::real_field_operations::RealFieldOperations`] trait, which encompasses
//! arithmetic operations for real fields and is required by the [`crate::RealField`] trait.

// Module declarations.
pub(crate) mod base_operations;
pub(crate) mod f64_interop;
pub(crate) mod impl_real_field_operations;
#[allow(clippy::module_inception)]
pub(crate) mod real_field_operations;
pub(crate) mod real_field_operations_ref_ops;
pub(crate) mod real_field_operations_rem;

// Re-exports.
pub use base_operations::BaseOperations;
pub use f64_interop::f64_interop::{F64Interop, assert_f64_interop};
pub use f64_interop::f64_lhs_ops::{F64LhsOps, F64LhsOpsBase, assert_f64_lhs_ops};
pub use f64_interop::f64_rhs_ops::{F64RhsOps, assert_f64_rhs_ops};
pub use real_field_operations::{RealFieldOperations, assert_real_field_operations};
pub use real_field_operations_ref_ops::RealFieldOperationsRefOps;
#[cfg(not(feature = "faer"))]
pub use real_field_operations_ref_ops::RefOps;
pub use real_field_operations_rem::RealFieldOperationsRem;
