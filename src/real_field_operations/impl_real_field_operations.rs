/// This macro implements the following operations for a type `T` that implements
/// [`crate::real_field_operations::BaseOperations`]:
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
///
/// where `op` is each of `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`.
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
        $assign_trait:ident,
        $assign_method:ident
    ) => {
        // T op T
        impl $op_trait<$t> for $t {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: $t) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(self, rhs)
            }
        }

        // T op &T
        impl $op_trait<&$t> for $t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &$t) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(self, *rhs)
            }
        }

        // &T op T
        impl $op_trait<$t> for &$t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: $t) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(*self, rhs)
            }
        }

        // &T op &T
        impl $op_trait<&$t> for &$t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &$t) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(*self, *rhs)
            }
        }

        // T op= T
        impl $assign_trait<$t> for $t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: $t) {
                *self = $crate::real_field_operations::BaseOperations::$op_method(*self, rhs);
            }
        }

        // T op= &T
        impl $assign_trait<&$t> for $t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: &$t) {
                *self = $crate::real_field_operations::BaseOperations::$op_method(*self, *rhs);
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
                $crate::real_field_operations::BaseOperations::$op_method(self, <$t>::from(rhs))
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
                $crate::real_field_operations::BaseOperations::$op_method(self, <$t>::from(*rhs))
            }
        }

        // &T op f64
        impl $op_trait<f64> for &$t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: f64) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(*self, <$t>::from(rhs))
            }
        }

        // &T op &f64
        impl $op_trait<&f64> for &$t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &f64) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(*self, <$t>::from(*rhs))
            }
        }

        // T op= f64
        impl $assign_trait<f64> for $t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: f64) {
                *self = $crate::real_field_operations::BaseOperations::$op_method(
                    *self,
                    <$t>::from(rhs),
                );
            }
        }

        // T op= &f64
        impl $assign_trait<&f64> for $t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            #[inline]
            fn $assign_method(&mut self, rhs: &f64) {
                *self = $crate::real_field_operations::BaseOperations::$op_method(
                    *self,
                    <$t>::from(*rhs),
                );
            }
        }
    };
}

/// This macro implements the following operations for a type `T` that implements
/// [`crate::real_field_operations::BaseOperations`]:
///
/// * `f64 op T`
/// * `f64 op &T`
/// * `&f64 op T`
/// * `&f64 op &T`
///
/// where `op` is each of `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`.
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
        $op_method:ident
    ) => {
        // f64 op T
        impl $op_trait<$t> for f64
        where
            $t: From<f64>,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: $t) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(<$t>::from(self), rhs)
            }
        }

        // f64 op &T
        impl $op_trait<&$t> for f64
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &$t) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(<$t>::from(self), *rhs)
            }
        }

        // &f64 op T
        impl $op_trait<$t> for &f64
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: $t) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(<$t>::from(*self), rhs)
            }
        }

        // &f64 op &T
        impl $op_trait<&$t> for &f64
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            type Output = $t;

            #[inline]
            fn $op_method(self, rhs: &$t) -> Self::Output {
                $crate::real_field_operations::BaseOperations::$op_method(<$t>::from(*self), *rhs)
            }
        }
    };
}

/// This macro implements the following operations for a type `T` that implements
/// [`crate::real_field_operations::BaseOperations`]:
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
/// // `linalg_traits::real_field_operations::BaseOperations`
/// impl_real_field_operations!(MyType);
/// ```
#[macro_export]
macro_rules! impl_real_field_operations {
    ($t:ty) => {
        use std::ops::{
            Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
        };

        // ------------
        // Comparisons.
        // ------------

        // ==, !=
        impl PartialEq for $t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                $crate::real_field_operations::BaseOperations::eq(*self, *other)
            }
        }

        // <, <=, >, >=
        impl PartialOrd for $t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                $crate::real_field_operations::BaseOperations::partial_cmp(*self, *other)
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
                $crate::real_field_operations::BaseOperations::neg(self)
            }
        }

        // -&T
        impl Neg for &$t
        where
            $t: $crate::real_field_operations::BaseOperations,
        {
            type Output = $t;

            #[inline]
            fn neg(self) -> Self::Output {
                $crate::real_field_operations::BaseOperations::neg(*self)
            }
        }

        // ----------------------------------------------------
        // Operations with `T` LHS and either `T` or `f64` RHS.
        // ----------------------------------------------------

        // Addition between `T`'s: T + T, T + &T, &T + T, &T + &T
        // Addition-assignemnt for `T`'s: T += T, T += &T
        // Addition between `T` and `f64`: T + f64, T + &f64, &T + f64, &T + &f64
        $crate::__impl_base_op!($t, Add, add, AddAssign, add_assign);

        // Subtraction between `T`'s: T - T, T - &T, &T - T, &T - &T
        // Subtraction-assignemnt for `T`'s: T -= T, T -= &T
        // Subtraction between `T` and `f64`: T - f64, T - &f64, &T - f64, &T - &f64
        $crate::__impl_base_op!($t, Sub, sub, SubAssign, sub_assign);

        // Multiplication between `T`'s: T * T, T * &T, &T * T, &T * &T
        // Multiplication-assignemnt for `T`'s: T *= T, T *= &T
        // Multiplication between `T` and `f64`: T * f64, T * &f64, &T * f64, &T * &f64
        $crate::__impl_base_op!($t, Mul, mul, MulAssign, mul_assign);

        // Division between `T`'s: T / T, T / &T, &T / T, &T / &T
        // Division-assignemnt for `T`'s: T /= T, T /= &T
        // Division between `T` and `f64`: T / f64, T / &f64, &T / f64, &T / &f64
        $crate::__impl_base_op!($t, Div, div, DivAssign, div_assign);

        // Remainder between `T`'s: T % T, T % &T, &T % T, &T % &T
        // Remainder-assignemnt for `T`'s: T %= T, T %= &T
        // Remainder between `T` and `f64`: T % f64, T % &f64, &T % f64, &T % &f64
        $crate::__impl_base_op!($t, Rem, rem, RemAssign, rem_assign);

        // --------------------------------------
        // Operations with `f64` LHS and `T` RHS.
        // --------------------------------------

        // Addition between `f64` and `T`: f64 + T, f64 + &T, &f64 + T, &f64 + &T
        $crate::__impl_base_op_f64_lhs!($t, Add, add);

        // Subtraction between `f64` and `T`: f64 - T, f64 - &T, &f64 - T, &f64 - &T
        $crate::__impl_base_op_f64_lhs!($t, Sub, sub);

        // Multiplication between `f64` and `T`: f64 * T, f64 * &T, &f64 * T, &f64 * &T
        $crate::__impl_base_op_f64_lhs!($t, Mul, mul);

        // Division between `f64` and `T`: f64 / T, f64 / &T, &f64 / T, &f64 / &T
        $crate::__impl_base_op_f64_lhs!($t, Div, div);

        // Remainder between `f64` and `T`: f64 % T, f64 % &T, &f64 % T, &f64 % &T
        $crate::__impl_base_op_f64_lhs!($t, Rem, rem);
    };
}
