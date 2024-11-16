use std::ops::{Index, IndexMut};
use crate::scalar::Scalar;

/// Trait defining common matrix methods and operations.
///
/// # Note
///
/// In addition to the methods defined by this trait, this trait also forces that the implementor
/// also support indexing ([`Index`]) and mutable indexing ([`IndexMut`]) by a 2-element tuple of
/// [`usize`]s (1st element defining the row index, 2nd element defining the column index).
/// 
/// # Using [`Matrix`] as a trait bound
/// 
/// Say I want to write a function that is generic over all matrices of [`f64`], e.g. I want it to
/// be compatible both with [`ndarray::Array2<f64>`] and with [`nalgebra::DVector<f64>`]. I can
/// define this function as
/// 
/// ```ignore
/// fn my_function<M: Matrix<f64>>(input_vector: &M) -> M { ... }
/// ```
/// 
/// Since the [`Matrix`] trait is generic over types that implement the [`Scalar`] trait, any
/// function that is generic over [`Matrix`]es can also be made generic over the type of their
/// elements. In this case, if we want `my_function` to be compatible with vectors of any scalar
/// type (i.e. types that implement the [`Scalar`] trait), and not just vectors of [`f64`]s, we can
/// include an additional generic parameter `S`.
/// 
/// ```ignore
/// fn my_function<S: Scalar, V: Vector<S>>(input_vector: &V) -> V { ... }
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
pub trait Matrix<S: Scalar>:
    Index<(usize, usize), Output = S>       // Indexing via square brackets.
    + IndexMut<(usize, usize), Output = S>  // Index-assignment via square brackets.
    + Clone                                 // Copying (compatible with dynamically-sized types).
{
    /// Length-`N` Vector type implementing the [`crate::Vector`] trait that is compatible with this
    /// matrix type. An instance of this matrix type with shape `(M, N)` can be multiplied from the
    /// right by an instance of this vector type with length `N`, resulting in an instance of this
    /// vector type with length `M`.
    type VectorN;

    /// Length-`M` Vector type implementing the [`crate::Vector`] trait that is compatible with this
    /// matrix type. An instance of this matrix type with shape `(M, N)` can be multiplied from the
    /// left by an instance of this vector type with length `M`, resulting in an instance of this
    /// vector type with length `N`.
    type VectorM;

    /// Determine whether or not the matrix is row-major.
    /// 
    /// # Returns
    /// 
    /// `true` if the matrix is row-major, `false` if the matrix is column-major.
    fn is_row_major() -> bool;

    /// Determine whether or not the matrix is column-major.
    /// 
    /// # Returns
    /// 
    /// `true` if the matrix is column-major, `false` if the matrix is row-major.
    fn is_column_major() -> bool;

    /// Create a matrix with the specified size, with each element set to 0.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows.
    /// * `cols` - Number of columns.
    ///
    /// # Returns
    ///
    /// Matrix with the specified size, with each element set to 0.
    /// 
    /// # Panics
    /// 
    /// * If `rows` does not match the number of rows in the matrix (for statically-sized matrices
    ///   only).
    fn new_with_shape(rows: usize, cols: usize) -> Self;

    /// Get the shape of the matrix.
    ///
    /// # Returns
    ///
    /// * `rows` - Number of rows.
    /// * `cols` - Number of columns.
    fn shape(&self) -> (usize, usize);

    /// Create a matrix from a slice of scalars arranged in row-major order.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows.
    /// * `cols` - Number of columns.
    /// * `slice` - The slice of scalar values to initialize the matrix.
    ///
    /// # Returns
    ///
    /// A matrix containing the elements from the slice.
    /// 
    /// # Panics
    /// 
    /// * If `rows` does not match the number of rows in the matrix (for statically-sized matrices
    ///   only)
    fn from_row_slice(rows: usize, cols: usize, slice: &[S]) -> Self;

    /// Create a matrix from a slice of scalars arranged in column-major order.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows.
    /// * `cols` - Number of columns.
    /// * `slice` - The slice of scalar values to initialize the matrix.
    ///
    /// # Returns
    ///
    /// A matrix containing the elements from the slice.
    /// 
    /// # Panics
    /// 
    /// * If `rows` does not match the number of rows in the matrix (for statically-sized matrices
    ///   only)/
    /// * If the slice length is not compatible with the shape of the matrix (for dynamically-sized
    ///   matrices only).
    fn from_col_slice(rows: usize, cols: usize, slice: &[S]) -> Self;

    /// Return a slice view of the matrix's elements.
    ///
    /// # Returns
    ///
    /// A slice of the matrix's elements.
    /// 
    /// # Note
    /// 
    /// The order of the elements depends on whether the matrix is row-major or column-major. This
    /// can be programmatically determined via the [`Matrix::is_row_major`] and
    /// [`Matrix::is_column_major`] methods.
    fn as_slice(&self) -> &[S];

    /// Assert that this matrix and another matrix have the same shape. 
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other matrix whose shape we are comparing with this matrix.
    /// 
    /// # Panics
    /// 
    /// * If the shape of the other matrix is not equal to the shape of this matrix.
    fn assert_same_shape(&self, other: &Self) {
        assert_eq!(
            self.shape(),
            other.shape(), 
            "Matrices have incompatible shapes.",
        );
    }

    /// Matrix addition (elementwise).
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other matrix to add to this matrix.
    /// 
    /// # Returns
    /// 
    /// Sum of this matrix with the other matrix (i.e. `self + other`).
    /// 
    /// # Panics
    /// 
    /// * If `self` and `other` are dynamically-sized matrices and do not have the same shape.
    fn add(&self, other: &Self) -> Self;

    /// In-place matrix addition (elementwise) (`self += other`).
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other matrix to add to this matrix.
    /// 
    /// # Panics
    /// 
    /// * If `self` and `other` are dynamically-sized matrices and do not have the same shape.
    fn add_assign(&mut self, other: &Self);

    /// Matrix subtraction (elementwise).
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other matrix to subtract from this matrix.
    /// 
    /// # Returns
    /// 
    /// The difference of matrix with the other matrix (i.e. `self - other`).
    /// 
    /// # Panics
    /// 
    /// * If `self` and `other` are dynamically-sized matrices and do not have the same shape.
    fn sub(&self, other: &Self) -> Self;

    /// In-place matrix subtraction (elementwise) (`self -= other`).
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other matrix to subtract from this matrix.
    /// 
    /// # Panics
    /// 
    /// * If `self` and `other` are dynamically-sized matrices and do not have the same shape.
    fn sub_assign(&mut self, other: &Self);

    /// Matrix-scalar multiplication.
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to multiply each element of this matrix by.
    /// 
    /// # Returns
    /// 
    /// Product of this matrix with the scalar (i.e. `self * scalar` or `scalar * self`).
    fn mul(&self, scalar: S) -> Self;

    /// In-place matrix-scalar multiplication (`self * scalar` or `scalar * self`).
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to multiply each element of this matrix by.
    fn mul_assign(&mut self, scalar: S);

    /// Matrix-scalar division.
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to divide each element of this matrix by.
    /// 
    /// # Returns
    /// 
    /// This matrix divided by the scalar (i.e. `self / scalar`).
    fn div(&self, scalar: S) -> Self;

    /// In-place matrix-scalar division (`self / scalar`).
    /// 
    /// # Arguments
    /// 
    /// * `scalar` - The scalar to divide each element of this matrix by.
    fn div_assign(&mut self, scalar: S);

}