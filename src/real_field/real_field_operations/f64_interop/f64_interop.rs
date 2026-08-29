use crate::real_field::real_field_operations::f64_interop::f64_lhs_ops::F64LhsOps;
use crate::real_field::real_field_operations::f64_interop::f64_rhs_ops::F64RhsOps;
use crate::verify_trait_implemented;
use imply_hack::Imply;

const _: bool = verify_trait_implemented!(f64: F64Interop);

/// Trait defining the interoperability of a generic type `T` with [`f64`].
pub trait F64Interop:
    Sized
    // Conversion between the scalar type and `f64`.
    + From<f64>
    + Into<f64>
    // Operations between `T` (LHS) and `f64` (RHS).
    + F64RhsOps
    // Operations between `f64` (LHS) and `T` (RHS).
    + F64LhsOps
    // Conversions to/from `T` for `f64`.
    + Imply<f64, Is: From<Self>>
    + Imply<f64, Is: Into<Self>>
{
}

// Blanket implementation.
impl<T> F64Interop for T
where
    T: From<f64> + Into<f64> + F64RhsOps + F64LhsOps,
    f64: From<T> + Into<T>,
{
}
