use crate::real_field_operations::f64_interop::f64_lhs_ops::F64LhsOps;
use crate::real_field_operations::f64_interop::f64_rhs_ops::F64RhsOps;
use crate::verify_trait_implemented;
use imply_hack::Imply;
use std::fmt::Debug;

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

/// Asserts that all required to/from conversions between [`f64`] and `T` are correctly implemented.
/// 
/// Definitions of operators are not checked by this function, and are instead checked by one of
/// the following functions:
/// 
/// * [`crate::f64_interop::assert_f64_rhs_ops`]
/// * [`crate::f64_interop::assert_f64_lhs_ops`]
/// 
/// # Arguments
/// 
/// * `value_t` - A value of type `T`.
/// * `value_f64` - A value of type `f64`.
/// * `expected_t_into_f64` - The expected result of converting `value_t` into `f64`.
/// * `expected_f64_into_t` - The expected result of converting `value_f64` into `T`.
/// 
/// # Panics
/// 
/// Panics if any of the assertions for [`From`] or [`Into`] conversions fail.
pub fn assert_f64_interop<T: F64Interop + Copy + Debug + PartialEq>(
    value_t: T,
    value_f64: f64,
    expected_t_into_f64: f64,
    expected_f64_into_t: T
) {
    // Check `From`.
    assert_eq!(T::from(value_f64), expected_f64_into_t);
    assert_eq!(f64::from(value_t), expected_t_into_f64);

    // Check `Into`.
    assert_eq!(value_t.into(), expected_t_into_f64);
    assert_eq!(value_f64.into(), expected_f64_into_t);
}