/// Implement everything needed for a type to satisfy [`crate::RealField`], given that the type
/// already implements [`crate::real_field::RealFieldBase`].
///
/// For this to work correctly, the calling crate must define its own `faer`, `nalgebra`, and
/// `ndarray` features (typically forwarding to `linalg-traits`'s features of the same names).
///
/// # Generic Arguments
///
/// * `$t` - The type for which the rest of [`crate::RealField`]'s requirements are being
///   implemented.
#[macro_export]
macro_rules! impl_real_field {
    ($t:ty) => {
        $crate::impl_to_from_f64!($t);

        // Implement real field operations.
        $crate::impl_real_field_operations!($t);

        // `faer_traits::RealField`, `nalgebra::RealField`, and `ndarray::LinalgScalar` all require
        // `num_traits::Num` (implemented exactly once here, regardless of how many of them are
        // enabled).
        #[cfg(any(feature = "faer", feature = "nalgebra", feature = "ndarray"))]
        $crate::impl_num_traits_num!($t);

        // `nalgebra::RealField` additionally requires `num_traits::Signed` and
        // `num_traits::FromPrimitive`.
        #[cfg(feature = "nalgebra")]
        $crate::impl_num_traits_signed!($t);
        #[cfg(feature = "nalgebra")]
        $crate::impl_num_traits_from_primitive!($t);

        // `ndarray::ScalarOperand` is a marker that must be implemented for custom scalar types.
        #[cfg(feature = "ndarray")]
        impl ndarray::ScalarOperand for $t {}

        // At this point, `ndarray::LinalgScalar` is satisfied by ndarray's blanket implementation.

        // Implement `faer_traits::RealField` (and its required supertraits that haven't already
        // been implemented).
        #[cfg(feature = "faer")]
        $crate::impl_faer_traits_real_field!($t);

        // Implement `nalgebra::RealField` (and its required supertraits that haven't already been
        // implemented).
        #[cfg(feature = "nalgebra")]
        $crate::impl_nalgebra_real_field!($t);
    };
}
