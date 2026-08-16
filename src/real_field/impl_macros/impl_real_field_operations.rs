/// This macro implements the following operations for a type `T` that implements
/// [`crate::real_field::RealFieldBase`]:
///
/// * `T op T`
/// * `T op &T`
/// * `&T op T`
/// * `&T op &T`
/// * `T op= T`
/// * `T op= &T`
/// * `T op f64`
/// * `T op &f64`
/// * `&T op f64`
/// * `&T op &f64`
/// * `T op= f64`
/// * `T op= &f64`
///
/// where `op` is each of `+`, `-`, `*`, `/`, `%`.
///
/// # Generic Arguments
///
/// * `$t` - The type for which to implement the operations listed above.
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_base_op {
    (
        $t:ty,
        $op_trait:ident,
        $op_method:ident,
        $base_op_method:ident,
        $assign_trait:ident,
        $assign_method:ident
    ) => {
        // T op T
        impl $op_trait<$t> for $t {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: $t) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(self, rhs)
            }
        }

        // T op &T
        impl $op_trait<&$t> for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &$t) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(self, *rhs)
            }
        }

        // &T op T
        impl $op_trait<$t> for &$t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: $t) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(*self, rhs)
            }
        }

        // &T op &T
        impl $op_trait<&$t> for &$t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &$t) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(*self, *rhs)
            }
        }

        // T op= T
        impl $assign_trait<$t> for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: $t) {
                *self = $crate::real_field::RealFieldBase::$base_op_method(*self, rhs);
            }
        }

        // T op= &T
        impl $assign_trait<&$t> for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: &$t) {
                *self = $crate::real_field::RealFieldBase::$base_op_method(*self, *rhs);
            }
        }

        // T op f64
        impl $op_trait<f64> for $t
        where
            $t: From<f64>,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: f64) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(self, <$t>::from(rhs))
            }
        }

        // T op &f64
        impl $op_trait<&f64> for $t
        where
            $t: From<f64>,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &f64) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(self, <$t>::from(*rhs))
            }
        }

        // &T op f64
        impl $op_trait<f64> for &$t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: f64) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(*self, <$t>::from(rhs))
            }
        }

        // &T op &f64
        impl $op_trait<&f64> for &$t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &f64) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(*self, <$t>::from(*rhs))
            }
        }

        // T op= f64
        impl $assign_trait<f64> for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: f64) {
                *self = $crate::real_field::RealFieldBase::$base_op_method(*self, <$t>::from(rhs));
            }
        }

        // T op= &f64
        impl $assign_trait<&f64> for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: &f64) {
                *self = $crate::real_field::RealFieldBase::$base_op_method(*self, <$t>::from(*rhs));
            }
        }
    };
}

/// This macro implements the following operations for a type `T` that implements
/// [`crate::real_field::RealFieldBase`]:
///
/// * `T op f64`
///
/// where `op` is each of `==`, `!=`, `<`, `<=`, `>`, `>=`.
///
/// # Generic Arguments
///
/// * `$t` - The type for which to implement the operations listed above.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_base_cmp_f64_rhs {
    ($t:ty) => {
        // T == f64
        impl PartialEq<f64> for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn eq(&self, other: &f64) -> bool {
                $crate::real_field::RealFieldBase::_eq(*self, <$t>::from(*other))
            }
        }

        // T partial_cmp f64
        impl PartialOrd<f64> for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
                $crate::real_field::RealFieldBase::_partial_cmp(*self, <$t>::from(*other))
            }
        }
    };
}

/// This macro implements the following operations for a type `T` that implements
/// [`crate::real_field::RealFieldBase`]:
///
/// * `f64 op T`
/// * `f64 op &T`
/// * `&f64 op T`
/// * `&f64 op &T`
///
/// where `op` is each of `+`, `-`, `*`, `/`, `%`.
///
/// # Generic Arguments
///
/// * `$t` - The type for which to implement the operations listed above.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_base_op_f64_lhs {
    (
        $t:ty,
        $op_trait:ident,
        $op_method:ident,
        $base_op_method:ident
    ) => {
        // f64 op T
        impl $op_trait<$t> for f64
        where
            $t: From<f64>,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: $t) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(<$t>::from(self), rhs)
            }
        }

        // f64 op &T
        impl $op_trait<&$t> for f64
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &$t) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(<$t>::from(self), *rhs)
            }
        }

        // &f64 op T
        impl $op_trait<$t> for &f64
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: $t) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(<$t>::from(*self), rhs)
            }
        }

        // &f64 op &T
        impl $op_trait<&$t> for &f64
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &$t) -> Self::Output {
                $crate::real_field::RealFieldBase::$base_op_method(<$t>::from(*self), *rhs)
            }
        }
    };
}

/// This macro implements the following operations for a type `T` that implements
/// [`crate::real_field::RealFieldBase`]:
///
/// * `f64 op T`
///
/// where `op` is each of `==`, `!=`, `<`, `<=`, `>`, `>=`.
///
/// # Generic Arguments
///
/// * `$t` - The type for which to implement the operations listed above.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_base_cmp_f64_lhs {
    ($t:ty) => {
        // f64 == T
        impl PartialEq<$t> for f64
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn eq(&self, other: &$t) -> bool {
                $crate::real_field::RealFieldBase::_eq(<$t>::from(*self), *other)
            }
        }

        // f64 partial_cmp T
        impl PartialOrd<$t> for f64
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                <$t>::from(*self).partial_cmp(other)
            }
        }
    };
}

/// This macro implements the following operations for a type `T` that implements
/// [`crate::real_field::RealFieldBase`]:
///
/// * `-T`
/// * `-&T`
/// * `T op T`
/// * `T op &T`
/// * `&T op T`
/// * `&T op &T`
/// * `T op= T`
/// * `T op= &T`
/// * `T op f64`
/// * `T op &f64`
/// * `&T op f64`
/// * `&T op &f64`
/// * `T op= f64`
/// * `T op= &f64`
/// * `f64 op T`
/// * `f64 op &T`
/// * `&f64 op T`
/// * `&f64 op &T`
///
/// where `op` is each of `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`.
///
/// # Generic Arguments
///
/// * `$t` - The type for which to implement the base operations.
///
/// # Example
///
/// ```rust,ignore
/// use linalg_traits::impl_real_field_operations;
///
/// // This is assuming `MyType` already implements
/// // `linalg_traits::real_field::RealFieldBase`
/// impl_real_field_operations!(MyType);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! impl_real_field_operations {
    ($t:ty) => {
        use std::ops::{
            Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
        };

        // ------------
        // Comparisons.
        // ------------

        // TODO: these things should be in dedicated macros

        // ==, !=
        impl PartialEq for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                $crate::real_field::RealFieldBase::_eq(*self, *other)
            }
        }

        // T == f64
        $crate::__impl_base_cmp_f64_rhs!($t);

        // <, <=, >, >=
        impl PartialOrd for $t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                $crate::real_field::RealFieldBase::_partial_cmp(*self, *other)
            }
        }

        // ---------------
        // Unary negation.
        // ---------------

        // -T
        impl Neg for $t {
            type Output = $t;

            #[inline]
            fn neg(self) -> Self::Output {
                $crate::real_field::RealFieldBase::_neg(self)
            }
        }

        // -&T
        impl Neg for &$t
        where
            $t: $crate::real_field::RealFieldBase,
        {
            type Output = $t;

            #[inline]
            fn neg(self) -> Self::Output {
                $crate::real_field::RealFieldBase::_neg(*self)
            }
        }

        // ----------------------------------------------------
        // Operations with `T` LHS and either `T` or `f64` RHS.
        // ----------------------------------------------------

        // Addition between `T`'s: T + T, T + &T, &T + T, &T + &T
        // Addition-assignemnt for `T`'s: T += T, T += &T
        // Addition between `T` and `f64`: T + f64, T + &f64, &T + f64, &T + &f64
        $crate::__impl_base_op!($t, Add, add, _add, AddAssign, add_assign);

        // Subtraction between `T`'s: T - T, T - &T, &T - T, &T - &T
        // Subtraction-assignemnt for `T`'s: T -= T, T -= &T
        // Subtraction between `T` and `f64`: T - f64, T - &f64, &T - f64, &T - &f64
        $crate::__impl_base_op!($t, Sub, sub, _sub, SubAssign, sub_assign);

        // Multiplication between `T`'s: T * T, T * &T, &T * T, &T * &T
        // Multiplication-assignemnt for `T`'s: T *= T, T *= &T
        // Multiplication between `T` and `f64`: T * f64, T * &f64, &T * f64, &T * &f64
        $crate::__impl_base_op!($t, Mul, mul, _mul, MulAssign, mul_assign);

        // Division between `T`'s: T / T, T / &T, &T / T, &T / &T
        // Division-assignemnt for `T`'s: T /= T, T /= &T
        // Division between `T` and `f64`: T / f64, T / &f64, &T / f64, &T / &f64
        $crate::__impl_base_op!($t, Div, div, _div, DivAssign, div_assign);

        // Remainder between `T`'s: T % T, T % &T, &T % T, &T % &T
        // Remainder-assignemnt for `T`'s: T %= T, T %= &T
        // Remainder between `T` and `f64`: T % f64, T % &f64, &T % f64, &T % &f64
        $crate::__impl_base_op!($t, Rem, rem, _rem, RemAssign, rem_assign);

        // --------------------------------------
        // Operations with `f64` LHS and `T` RHS.
        // --------------------------------------

        // Addition between `f64` and `T`: f64 + T, f64 + &T, &f64 + T, &f64 + &T
        $crate::__impl_base_op_f64_lhs!($t, Add, add, _add);

        // Subtraction between `f64` and `T`: f64 - T, f64 - &T, &f64 - T, &f64 - &T
        $crate::__impl_base_op_f64_lhs!($t, Sub, sub, _sub);

        // Multiplication between `f64` and `T`: f64 * T, f64 * &T, &f64 * T, &f64 * &T
        $crate::__impl_base_op_f64_lhs!($t, Mul, mul, _mul);

        // Division between `f64` and `T`: f64 / T, f64 / &T, &f64 / T, &f64 / &T
        $crate::__impl_base_op_f64_lhs!($t, Div, div, _div);

        // Remainder between `f64` and `T`: f64 % T, f64 % &T, &f64 % T, &f64 % &T
        $crate::__impl_base_op_f64_lhs!($t, Rem, rem, _rem);

        // f64 == T
        $crate::__impl_base_cmp_f64_lhs!($t);
    };
}
