use std::ops::{Index, IndexMut};

/// Trait defining common vector methods and operations.
///
/// # Note
///
/// In addition to the methods defined by this trait, this trait also forces that the implementor
/// also support indexing ([`std::ops::Index`]) and mutable indexing ([`std::ops::IndexMut`]).
pub trait Vector:
    Index<usize, Output = f64>      // Indexing via square brackets.
    + IndexMut<usize, Output = f64> // Index-assignment via square brackets.
    + Clone                         // Copying (compatible with dynamically-sized types).
{
    /// Create a vector with the specified length, with each element set to 0.
    ///
    /// # Arguments
    ///
    /// * `len` - Desired length of the vector.
    ///
    /// # Returns
    ///
    /// Vector with the specified length, with each element set to 0.
    fn new_with_length(len: usize) -> Self;

    /// Get the length of the vector.
    ///
    /// # Returns
    ///
    /// Length of the vector.
    fn len(&self) -> usize;

    /// Determine if the vector is empty.
    ///
    /// # Returns
    ///
    /// `true` if the vector is empty, `false` if it is not empty.
    fn is_empty(&self) -> bool;

    /// Create a vector from a slice of [`f64`].
    ///
    /// # Arguments
    ///
    /// * `slice` - The slice of [`f64`] values to initialize the vector.
    ///
    /// # Returns
    ///
    /// A vector containing the elements from the slice.
    fn from_slice(slice: &[f64]) -> Self;

    /// Return a slice view of the vector's elements.
    ///
    /// # Returns
    ///
    /// A slice of the vector's elements.
    fn as_slice(&self) -> &[f64];

    /// Assert that this vector and another vector have the same length. 
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other vector whose length we are comparing with this vector.
    /// 
    /// # Panics
    /// 
    /// * If the length of the other vector is not equal to the length of this vector.
    fn assert_same_length(&self, other: &Self) {
        assert_eq!(
            self.len(),
            other.len(), 
            "Length of the other vector ({}) does not match the length of this vector ({}).",
            self.len(),
            other.len()
        );
    }

    /// Vector addition (elementwise).
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other vector to add to this vector.
    /// 
    /// # Returns
    /// 
    /// Sum of this vector with the other vector (i.e. `self + other`).
    /// 
    /// # Panics
    /// 
    /// * If `self` and `other` are dynamically-sized vectors and do not have the same length.
    fn add(&self, other: &Self) -> Self;

    /// In-place vector addition (elementwise) (`self += other`).
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other vector to add to this vector.
    /// 
    /// # Panics
    /// 
    /// * If `self` and `other` are dynamically-sized vectors and do not have the same length.
    fn add_assign(&mut self, other: &Self);

    /// Vector subtraction (elementwise).
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other vector to subtract from this vector.
    /// 
    /// # Returns
    /// 
    /// The difference of vector with the other vector (i.e. `self - other`).
    /// 
    /// # Panics
    /// 
    /// * If `self` and `other` are dynamically-sized vectors and do not have the same length.
    fn sub(&self, other: &Self) -> Self;

    /// In-place vector subtraction (elementwise) (`self -= other`).
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other vector to subtract from this vector.
    /// 
    /// # Panics
    /// 
    /// * If `self` and `other` are dynamically-sized vectors and do not have the same length.
    fn sub_assign(&mut self, other: &Self);

    /// Vector-scalar multiplication.
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to multiply each element of this vector by.
    /// 
    /// # Returns
    /// 
    /// Product of this vector with the scalar (i.e. `self * scalar` or `scalar * self`).
    fn mul(&self, scalar: f64) -> Self;

    /// In-place vector-scalar multiplication (`self * scalar` or `scalar * self`).
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to multiply each element of this vector by.
    fn mul_assign(&mut self, scalar: f64);

    /// Vector-scalar division.
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to divide each element of this vector by.
    /// 
    /// # Returns
    /// 
    /// This vector divided by the scalar (i.e. `self / scalar).
    fn div(&self, scalar: f64) -> Self;

    /// In-place vector-scalar division (`self / scalar).
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to divide each element of this vector by.
    fn div_assign(&mut self, scalar: f64);

}
