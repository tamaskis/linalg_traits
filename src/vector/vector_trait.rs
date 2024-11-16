use std::ops::{Index, IndexMut};
use crate::scalar::Scalar;

/// Trait defining common vector methods and operations.
///
/// # Note
///
/// In addition to the methods defined by this trait, this trait also forces that the implementor
/// also support indexing ([`Index`]) and mutable indexing ([`IndexMut`]).
/// 
/// # Using [`Vector`] as a trait bound
/// 
/// Say I want to write a function that is generic over all vectors of [`f64`], e.g. I want it to be
/// compatible both with [`Vec<f64>`] and with [`nalgebra::Vector1<f64>`]. I can define this
/// function as
/// 
/// ```ignore
/// fn my_function<V: Vector<f64>>(input_vector: &V) -> V { ... }
/// ```
/// 
/// Since the [`Vector`] trait is generic over types that implement the [`Scalar`] trait, any
/// function that is generic over [`Vector`]s can also be made generic over the type of their
/// elements. In this case, if we want `my_function` to be compatible with vectors of any scalar
/// type (i.e. types that implement the [`Scalar`] trait), and not just matrices of [`f64`]s, we can
/// include an additional generic parameter `S`.
/// 
/// ```ignore
/// fn my_function<S: Scalar, M: Matrix<S>>(input_vector: &M) -> M { ... }
/// ```
/// 
/// ## Warning
///
/// When working with arrays from [`ndarray`], elements of the array must also implement the
/// following traits in addition to the [`Scalar`] trait:
/// 
/// * [`ndarray::ScalarOperand`]
/// * [`ndarray::LinalgScalar`]
/// 
/// For example, consider the case where we define the struct `CustomType` and implement the 
/// [`Scalar`] trait for `CustomType`. If we want to be able to pass an
/// [`ndarray::Array2<CustomType>`] into `my_function` from the example above, then we must also
/// implement the [`ndarray::ScalarOperand`] and [`ndarray::LinalgScalar`] traits for `CustomType`.
pub trait Vector<S: Scalar>:
    Index<usize, Output = S>        // Indexing via square brackets.
    + IndexMut<usize, Output = S>   // Index-assignment via square brackets.
    + Clone                         // Copying (compatible with dynamically-sized types).
{
    /// `N x N` matrix type implementing the [`crate::Matrix`] trait that is compatible with this
    /// vector type. An instance of this matrix type with shape `(N, N)` can be multiplied either
    /// from the right or the left by an instance of this vector type with length-`N`, resulting in
    /// another length-`N` vector in either case.
    /// 
    /// # Note
    /// 
    /// 1. When multiplying this `N x N` matrix type from the right by an instance of this vector
    ///    type with length-`N`, the resulting length-`N` vector mathematically represents a column
    ///    vector.
    /// 2. When multiplying this `N x N` matrix type from the left by an instance of this vector
    ///    type with length-`N`, the resulting length-`N` vector mathematically represents a row
    ///    vector.
    type MatrixNxN;
    
    /// `M x N` matrix type implementing the [`crate::Matrix`] trait that is compatible with this
    /// vector type. An instance of this matrix type with shape `(M, N)` can be multiplied either
    /// from the right by an instance of this vector type with length-`N` (resulting in a length-`M`
    /// vector), or from the left by an instance of this vector type with length-`M` (resulting in
    /// alength-`N`) vector.
    /// 
    /// 1. When multiplying this `M x N` matrix type from the right by an instance of this vector
    ///    type with length-`N`, the resulting length-`M` vector mathematically represents a column
    ///    vector.
    /// 2. When multiplying this `M x N` matrix type from the left by an instance of this vector
    ///    type with length-`M`, the resulting length-`N`` vector mathematically represents a row
    ///    vector.
    type MatrixMxN<const M: usize>;

    /// Create a vector with the specified length, with each element set to 0.
    ///
    /// # Arguments
    ///
    /// * `len` - Desired length of the vector.
    ///
    /// # Returns
    ///
    /// Vector with the specified length, with each element set to 0.
    /// 
    /// # Panics
    /// 
    /// * If `len` does not match the length of the vector (for statically-sized vectors only).
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

    /// Create a vector from a slice of scalars.
    ///
    /// # Arguments
    ///
    /// * `slice` - The slice of scalar values to initialize the vector.
    ///
    /// # Returns
    ///
    /// A vector containing the elements from the slice.
    fn from_slice(slice: &[S]) -> Self;

    /// Return a slice view of the vector's elements.
    ///
    /// # Returns
    ///
    /// A slice of the vector's elements.
    fn as_slice(&self) -> &[S];

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
    fn mul(&self, scalar: S) -> Self;

    /// In-place vector-scalar multiplication (`self * scalar` or `scalar * self`).
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to multiply each element of this vector by.
    fn mul_assign(&mut self, scalar: S);

    /// Vector-scalar division.
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to divide each element of this vector by.
    /// 
    /// # Returns
    /// 
    /// This vector divided by the scalar (i.e. `self / scalar`).
    fn div(&self, scalar: S) -> Self;

    /// In-place vector-scalar division (`self / scalar`).
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to divide each element of this vector by.
    fn div_assign(&mut self, scalar: S);

    /// Dot product of two vectors.
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other vector to take the dot product with.
    /// 
    /// # Returns
    /// 
    /// Dot product of this vector with the other vector.
    /// 
    /// # Panics
    /// 
    /// * If the two vectors do not have the same length.
    fn dot(&self, other: &Self) -> S;
}