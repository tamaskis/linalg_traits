/// Implement the [`num_traits::Num`] trait (and its supertraits [`num_traits::Zero`] and
/// [`num_traits::One`]) for a type that has already implemented
/// [`crate::real_field::RealFieldBase`].
///
/// # Generic Arguments
///
/// * `$t` - The type for which the [`num_traits::Num`] trait is being implemented.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_num_traits_num {
    ($t:ty) => {
        impl $crate::__private::num_traits::Zero for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn zero() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_zero()
            }

            #[inline]
            fn is_zero(&self) -> bool {
                <$t as $crate::real_field::RealFieldBase>::_is_zero(self)
            }

            #[inline]
            fn set_zero(&mut self) {
                <$t as $crate::real_field::RealFieldBase>::_set_zero(self)
            }
        }

        impl $crate::__private::num_traits::One for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn one() -> Self {
                <$t as $crate::real_field::RealFieldBase>::_one()
            }
        }

        impl $crate::__private::num_traits::Num for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type FromStrRadixErr = core::num::ParseFloatError;

            #[inline]
            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                if radix == 10 {
                    str.parse::<f64>().map(<$t as From<f64>>::from)
                } else {
                    "invalid radix".parse::<f64>().map(<$t as From<f64>>::from)
                }
            }
        }
    };
}
