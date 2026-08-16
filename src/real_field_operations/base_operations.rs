/// The minimum set of arithmetic operations that need to be implemented for a type `T` when
/// implementing the [`crate::RealField`] trait.
///
/// All additional arithmetic operations can be derived from these fundamental operations using the
/// [`crate::impl_real_field_operations!`] macro.
pub trait BaseOperations: Sized + From<f64> + Copy {
    /// Unary negation of `self`.
    #[must_use]
    fn neg(self) -> Self;

    /// Addition (`self + rhs`).
    #[must_use]
    fn add(self, rhs: Self) -> Self;

    /// Subtraction (`self - rhs`).
    #[must_use]
    fn sub(self, rhs: Self) -> Self;

    /// Multiplication (`self * rhs`).
    #[must_use]
    fn mul(self, rhs: Self) -> Self;

    /// Division (`self / rhs`).
    #[must_use]
    fn div(self, rhs: Self) -> Self;

    /// Remainder after division (`self % rhs`).
    #[must_use]
    fn rem(self, rhs: Self) -> Self;

    /// Equality comparison (`self == rhs`).
    #[must_use]
    fn eq(self, rhs: Self) -> bool;

    /// [(Partial) ordering](https://doc.rust-lang.org/std/cmp/trait.PartialOrd.html)
    /// (`self.partial_cmp(&rhs)`).
    #[must_use]
    fn partial_cmp(self, rhs: Self) -> Option<std::cmp::Ordering>;
}
