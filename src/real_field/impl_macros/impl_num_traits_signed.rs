/// Implement the [`num_traits::Signed`] trait for a type that has already implemented
/// [`num_traits::Num`] (can be done via [`crate::impl_num_traits_num!`]) and
/// [`crate::real_field::RealFieldBase`].
///
/// # Generic Arguments
///
/// * `$t` - The type for which the [`num_traits::Signed`] trait is being implemented.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_num_traits_signed {
    ($t:ty) => {
        impl $crate::__private::num_traits::Signed for $t
        where
            $t: $crate::__private::num_traits::Num + $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn abs(&self) -> Self {
                <$t as $crate::real_field::RealFieldBase>::_abs(self)
            }

            #[inline]
            fn abs_sub(&self, other: &Self) -> Self {
                if *self <= *other {
                    <$t as $crate::__private::num_traits::Zero>::zero()
                } else {
                    *self - *other
                }
            }

            #[inline]
            fn signum(&self) -> Self {
                if <$t as $crate::real_field::RealFieldBase>::_is_nan(*self) {
                    *self
                } else if <$t as $crate::real_field::RealFieldBase>::_is_sign_negative(self) {
                    -<$t as $crate::__private::num_traits::One>::one()
                } else if <$t as $crate::__private::num_traits::Zero>::is_zero(self) {
                    <$t as $crate::__private::num_traits::Zero>::zero()
                } else {
                    <$t as $crate::__private::num_traits::One>::one()
                }
            }

            #[inline]
            fn is_positive(&self) -> bool {
                !<$t as $crate::__private::num_traits::Zero>::is_zero(self)
                    && <$t as $crate::real_field::RealFieldBase>::_is_sign_positive(self)
            }

            #[inline]
            fn is_negative(&self) -> bool {
                <$t as $crate::real_field::RealFieldBase>::_is_sign_negative(self)
            }
        }
    };
}
