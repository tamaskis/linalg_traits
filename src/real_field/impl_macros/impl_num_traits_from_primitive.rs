/// Implement the [`num_traits::FromPrimitive`] trait for a type that has already implemented
/// [`crate::real_field::RealFieldBase`].
///
/// # Generic Arguments
///
/// * `$t` - The type for which the [`num_traits::FromPrimitive`] trait is being implemented.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_num_traits_from_primitive {
    ($t:ty) => {
        impl $crate::__private::num_traits::FromPrimitive for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn from_i64(n: i64) -> Option<Self> {
                Some(<$t as From<f64>>::from(n as f64))
            }

            #[inline]
            fn from_u64(n: u64) -> Option<Self> {
                Some(<$t as From<f64>>::from(n as f64))
            }
        }
    };
}
