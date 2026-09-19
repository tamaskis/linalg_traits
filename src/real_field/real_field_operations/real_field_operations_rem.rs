use imply_hack::Imply;
use std::ops::{Rem, RemAssign};

/// Remainder operations.
///
/// This trait defines the following operations:
///
/// * `T % T`
/// * `T %= T`
/// * `T % &T`
/// * `T %= &T`
pub trait RealFieldOperationsRem:
    Sized
    + Rem<Output = Self>
    + RemAssign
    + for<'a> Rem<&'a Self, Output = Self>
    + for<'a> RemAssign<&'a Self>
    + for<'a> Imply<&'a Self, Is: Rem<Output = Self> + Rem<Self, Output = Self>>
{
}

// Blanket implementation.
impl<T> RealFieldOperationsRem for T
where
    T: Rem<Output = Self>
        + RemAssign
        + for<'a> Rem<&'a T, Output = Self>
        + for<'a> RemAssign<&'a T>,
    for<'a> T: Imply<&'a T, Is: Rem<Output = T> + Rem<T, Output = T>>,
{
}
