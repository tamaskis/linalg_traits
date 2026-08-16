/// Implement conversions to and from `f64` for a type that has already implemented
/// [`crate::real_field::RealFieldBase`].
///
/// # Generic Arguments
///
/// * `$t` - The type for which the conversions are implemented.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_to_from_f64 {
    ($t:ty) => {
        impl From<f64> for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn from(value: f64) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_from_f64(value)
            }
        }

        impl From<$t> for f64
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn from(value: $t) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_to_f64(&value)
            }
        }
    };
}
