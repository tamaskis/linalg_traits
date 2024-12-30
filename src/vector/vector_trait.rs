use std::ops::{Index, IndexMut};
use std::fmt::Debug;
use crate::scalar::Scalar;
use crate::matrix::matrix_trait::Matrix;

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
    + Debug                         // Debug printing.
    + PartialEq                     // Equality comparisons.
{
    // -----------------
    // Associated types.
    // -----------------


    /// Vector type that is of the same "outer" vector type (i.e. the `Vector` part of `Vector<S>`
    /// where `S: Scalar`), but where the element type can be any other type that implements the
    /// [`crate::Scalar`] trait.
    /// 
    /// # Note
    /// 
    /// * This associated type could be either statically or dynamically-sized, as long as it is
    ///   compatible with this vector type.
    /// * We recommend that statically-sized vectors choose a compatible statically-sized matrix for
    ///   this associated type, and the dynamically-sized vectors choose a compatible
    ///   dynamically-sized matrix for this associated type.
    /// * For [`ndarray::Array1`], we define this associated type as a [`Vec`]. This is because
    ///   elements of an [`ndarray::Array1`] must also implement [`ndarray::ScalarOperand`] and
    ///   [`ndarray::LinalgScalar`], but we cannot apply these trait bounds in the definition of
    ///   this associated type if we wish to keep [`crate::Vector`] independent of any external
    ///   crate.
    type VectorT<T: Scalar>: Vector<T>;

    /// Dynamically-sized vector type that is compatible with this "outer" vector type (i.e. the
    /// `Vector` part of `Vector<S>` where `S: Scalar`), but where the element type can be any other
    /// type that implements the [`crate::Scalar`] trait.
    /// 
    /// # Note
    /// 
    /// For [`ndarray::Array1`], we define this associated type as a [`Vec`]. This is because
    /// elements of an [`ndarray::Array1`] must also implement [`ndarray::ScalarOperand`] and
    /// [`ndarray::LinalgScalar`], but we cannot apply these trait bounds in the definition of this
    /// associated type if we wish to keep [`crate::Vector`] independent of any external crate.
    type DVectorT<T: Scalar>: Vector<T>;

    /// Length-`N` vector type that is of the same "outer" vector type (i.e. the `Vector` part of
    /// `Vector<S>` where `S: Scalar`), but where the elements are of type [`f64`].
    type Vectorf64: Vector<f64>;

    /// `N x N` matrix type implementing the [`crate::Matrix`] trait that is compatible with this
    /// vector type. An instance of this matrix type with shape `(N, N)` can be multiplied either
    /// from the right or the left by an instance of this vector type with length-`N`, resulting in
    /// another length-`N` vector in either case.
    /// 
    /// * When multiplied from the right by a vector, the resulting vector mathematically represents
    ///   a column vector.
    /// * When multiplied from the left by a vector, the resulting vector mathematically represents
    ///   a row vector.
    /// 
    /// # Note
    /// 
    /// * This associated type could be either statically or dynamically-sized, as long as it is
    ///   compatible with this vector type.
    /// * We recommend that statically-sized vectors choose a compatible statically-sized matrix for
    ///   this associated type, and the dynamically-sized vectors choose a compatible
    ///   dynamically-sized matrix for this associated type.
    type MatrixNxN: Matrix<S>;

    /// `M x N` matrix type implementing the [`crate::Matrix`] trait that is compatible with this
    /// vector type. An instance of this matrix type with shape `(M, N)` can be multiplied from the
    /// right by an instance of this vector type with length-`N`, resulting in a length-`M` vector
    /// (which mathematically represents a column vector).
    /// 
    /// # Note
    /// 
    /// * We say that the instance of the type implementing the [`Vector`] trait has length `N`.
    /// * Therefore, we already know one of the dimensions (`N`) of this `M`-by-`N` matrix.
    /// * This associated type could be either statically or dynamically-sized, as long as it is
    ///   compatible with this vector type.
    /// * For statically-sized matrices, to know the other dimension (`M`) at compile time, we need
    ///   to provide `M` as a const generic.
    /// * For dynamically-sized matrices, the const generic `M` is not used.
    type MatrixMxN<const M: usize>: Matrix<S>;
    
    /// Dynamically-sized `M x N` matrix type implementing the [`crate::Matrix`] trait that is
    /// compatible with this vector type. An instance of this matrix type with shape `(M, N)` can be
    /// multiplied from the right by an instance of this vector type with length-`N`, resulting in a
    /// length-`M` vector (which mathematically represents a column vector).
    /// 
    /// # Note
    /// 
    /// * We say that the instance of the type implementing the [`Vector`] trait has length `N`.
    /// * Therefore, we already know one of the dimensions (`N`) of this `M`-by-`N` matrix.
    /// * The other dimension (`M`) is determined at runtime, so this type must be
    ///   dynamically-sized.
    type DMatrixMxN: Matrix<S>;

    /// Dynamically-sized `M x N` matrix type of the same "outer" type as
    /// [`crate::Vector::DMatrixMxN`], but with elements of type [`f64`].
    type DMatrixMxNf64: Matrix<f64>;

    /// `N x M` matrix type implementing the [`crate::Matrix`] trait that is compatible with this
    /// vector type. An instance of this matrix type with shape `(N, M)` can be multiplied from the
    /// left by an instance of this vector type with length-`N`, resulting in a length-`M` vector
    /// (which mathematically represents a row vector).
    /// 
    /// # Note
    /// 
    /// * We say that the instance of the type implementing the [`Vector`] trait has length `N`.
    /// * Therefore, we already know one of the dimensions (`N`) of this `N`-by-`M` matrix.
    /// * This associated type could be either statically or dynamically-sized, as long as it is
    ///   compatible with this vector type.
    /// * For statically-sized matrices, to know the other dimension (`M`) at compile time, we need
    ///   to provide `M` as a const generic.
    /// * For dynamically-sized matrices, the const generic `M` is not used.
    type MatrixNxM<const M: usize>: Matrix<S>;

    /// Dynmically-sized `N x M` matrix type implementing the [`crate::Matrix`] trait that is
    /// compatible with this vector type. An instance of this matrix type with shape `(N, M)` can be
    /// multiplied from the left by an instance of this vector type with length-`N`, resulting in a
    /// length-`M` vector (which mathematically represents a row vector).
    /// 
    /// # Note
    /// 
    /// * We say that the instance of the type implementing the [`Vector`] trait has length `N`.
    /// * Therefore, we already know one of the dimensions (`N`) of this `N`-by-`M` matrix.
    /// * The other dimension (`M`) is determined at runtime, so this type must be
    ///   dynamically-sized.
    type DMatrixNxM: Matrix<S>;

    // -------------------------------
    // Default method implementations.
    // -------------------------------

    /// Given an instance of a vector, create a new vector of the same type and length, but where
    /// each element of the vector is an [`f64`] initialized to `0.0`.
    /// 
    /// # Returns
    /// 
    /// New vector of same type and length, but where each element of the vector is an [`f64`]
    /// initialized to `0.0`.
    fn new_vector_f64(&self) -> Self::Vectorf64 {
        Self::Vectorf64::new_with_length(self.len())
    }

    /// Given an instance of a vector, create a new vector of the same type and length, but where
    /// each element of the vector is an [`f64`] initialized to `0.0`.
    /// 
    /// # Returns
    /// 
    /// New vector of same type and length, but where each element of the vector is an [`f64`]
    /// initialized to `0.0`.
    fn new_matrix_m_by_n_f64(&self) -> Self::Vectorf64 {
        Self::Vectorf64::new_with_length(self.len())
    }
    
    /// Create an `N x N` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Returns
    /// 
    /// `N x N` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Examples
    /// 
    /// ## Creating a statically-sized matrix compatible with a statically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{SMatrix, SVector};
    /// 
    /// // Create a statically-sized vector of length-2.
    /// let vec: SVector<f64, 2> = SVector::new_with_length(2);
    /// 
    /// // Create a statically-sized 2x2 matrix.
    /// let mat: SMatrix<f64, 2, 2> = vec.new_matrix_n_by_n();
    /// assert_eq!(mat.shape(), (2, 2));
    /// ```
    /// 
    /// ## Creating a dynamically-sized matrix compatible with a dynamically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{DMatrix, DVector};
    /// 
    /// // Create a dynamically-sized vector of length-2.
    /// let vec: DVector<f64> = DVector::new_with_length(2);
    /// 
    /// // Create a dynamically-sized 2x2 matrix.
    /// let mat: DMatrix<f64> = vec.new_matrix_n_by_n();
    /// assert_eq!(mat.shape(), (2, 2));
    /// ```
    fn new_matrix_n_by_n(&self) -> Self::MatrixNxN {
        let n = self.len();
        Self::MatrixNxN::new_with_shape(n, n)
    }

    /// Create an `M x N` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Arguments
    /// 
    /// * `m` - Number of rows of the `M x N` matrix. Input as `None` for statically-sized vectors,
    ///         and as `Some(m)` for dynamically-sized vectors.
    /// 
    /// # Returns
    /// 
    /// `M x N` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Note
    /// 
    /// * Dynamically-sized vectors will determine the number of rows (`m`) for the corresponding
    ///   compatible dynamically-sized matrix through the argument `m`. The const generic parameter,
    ///   `M`, can be specified as `0` at compile time since it is not used anyways.
    /// * Statically-sized vectors will determine the number of rows (`m`) for the corresponding
    ///   compatible statically-sized matrix through the const generic parameter `M`. The argument
    ///   `m` should be input as `None` at compile time since it is not used anyways.
    /// 
    /// # Examples
    /// 
    /// ## Creating a statically-sized matrix compatible with a statically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{SMatrix, SVector};
    /// 
    /// // Specify the dimensions.
    /// const M: usize = 3;
    /// const N: usize = 2;
    /// 
    /// // Create a statically-sized vector of length-2.
    /// let vec: SVector<f64, N> = SVector::new_with_length(N);
    /// 
    /// // Create a statically-sized 3x2 matrix.
    /// let mat: SMatrix<f64, M, N> = vec.new_matrix_m_by_n::<M>(None);
    /// assert_eq!(mat.shape(), (M, N));
    /// ```
    /// 
    /// ## Creating a dynamically-sized matrix compatible with a dynamically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{DMatrix, DVector};
    /// 
    /// // Specify the dimensions.
    /// const M: usize = 3;
    /// const N: usize = 2;
    /// 
    /// // Create a dynamically-sized vector of length-2.
    /// let vec: DVector<f64> = DVector::new_with_length(N);
    /// 
    /// // Create a dynamically-sized 3x2 matrix.
    /// let mat: DMatrix<f64> = vec.new_matrix_m_by_n::<0>(Some(M));
    /// assert_eq!(mat.shape(), (M, N));
    /// ```
    fn new_matrix_m_by_n<const M: usize>(&self, m: Option<usize>) -> Self::MatrixMxN<M> {
        let n = self.len();
        let m = if Self::is_statically_sized() {
            M
        } else {
            m.unwrap()
        };
        Self::MatrixMxN::new_with_shape(m, n)
    }

    /// Create a dynamically-sized `M x N` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Arguments
    /// 
    /// * `m` - Number of rows of the `M x N` matrix.
    /// 
    /// # Returns
    /// 
    /// Dynamically-sized `M x N` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Examples
    /// 
    /// ## Creating a dynamically-sized matrix compatible with a statically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{DMatrix, SVector};
    /// 
    /// // Create a statically-sized vector of length-2.
    /// let vec: SVector<f64, 2> = SVector::new_with_length(2);
    /// 
    /// // Create a dynamically-sized 3x2 matrix.
    /// let mat: DMatrix<f64> = vec.new_dmatrix_m_by_n(3);
    /// assert_eq!(mat.shape(), (3, 2));
    /// ```
    /// 
    /// ## Creating a dynamically-sized matrix compatible with a dynamically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{DMatrix, DVector};
    /// 
    /// // Create a dynamically-sized vector of length-2.
    /// let vec: DVector<f64> = DVector::new_with_length(2);
    /// 
    /// // Create a dynamically-sized 3x2 matrix.
    /// let mat: DMatrix<f64> = vec.new_dmatrix_m_by_n(3);
    /// assert_eq!(mat.shape(), (3, 2));
    /// ```
    fn new_dmatrix_m_by_n(&self, m: usize) -> Self::DMatrixMxN {
        Self::DMatrixMxN::new_with_shape(m, self.len())
    }

    /// Create a dynamically-sized `M x N` matrix (where this vector is length-`N`) filled with
    /// `0.0_f64`.
    /// 
    /// # Arguments
    /// 
    /// * `m` - Number of rows of the `M x N` matrix.
    /// 
    /// # Returns
    /// 
    /// Dynamically-sized `M x N` matrix filled with `0.0_f64`.
    fn new_dmatrix_m_by_n_f64(&self, m: usize) -> Self::DMatrixMxNf64 {
        Self::DMatrixMxNf64::new_with_shape(m, self.len())
    }

    /// Create an `N x M` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Arguments
    /// 
    /// * `m` - Number of columns of the `N x M` matrix. Input as `None` for statically-sized
    ///         vectors, and as `Some(m)` for dynamically-sized vectors.
    /// 
    /// # Returns
    /// 
    /// `N x M` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Note
    /// 
    /// * Dynamically-sized vectors will determine the number of columns (`m`) for the corresponding
    ///   compatible dynamically-sized matrix through the argument `m`. The const generic parameter,
    ///   `M`, can be specified as `0` at compile time since it is not used anyways.
    /// * Statically-sized vectors will determine the number of columns (`m`) for the corresponding
    ///   compatible statically-sized matrix through the const generic parameter `M`. The argument
    ///   `m` should be input as `None` at compile time since it is not used anyways.
    /// 
    /// # Examples
    /// 
    /// ## Creating a statically-sized matrix compatible with a statically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{SMatrix, SVector};
    /// 
    /// // Specify the dimensions.
    /// const N: usize = 2;
    /// const M: usize = 3;
    /// 
    /// // Create a statically-sized vector of length-2.
    /// let vec: SVector<f64, N> = SVector::new_with_length(N);
    /// 
    /// // Create a statically-sized 2x3 matrix.
    /// let mat: SMatrix<f64, N, M> = vec.new_matrix_n_by_m::<M>(None);
    /// assert_eq!(mat.shape(), (N, M));
    /// ```
    /// 
    /// ## Creating a dynamically-sized matrix compatible with a dynamically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{DMatrix, DVector};
    /// 
    /// // Specify the dimensions.
    /// const N: usize = 2;
    /// const M: usize = 3;
    /// 
    /// // Create a dynamically-sized vector of length-2.
    /// let vec: DVector<f64> = DVector::new_with_length(N);
    /// 
    /// // Create a dynamically-sized 2x3 matrix.
    /// let mat: DMatrix<f64> = vec.new_matrix_n_by_m::<0>(Some(M));
    /// assert_eq!(mat.shape(), (N, M));
    /// ```
    fn new_matrix_n_by_m<const M: usize>(&self, m: Option<usize>) -> Self::MatrixNxM<M> {
        let n = self.len();
        let m = if Self::is_statically_sized() {
            M
        } else {
            m.unwrap()
        };
        Self::MatrixNxM::new_with_shape(n, m)
    }

    /// Create a dynamically-sized `N x M` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Arguments
    /// 
    /// * `m` - Number of columns of the `N x M` matrix. Input as `None` for statically-sized
    ///         vectors, and as `Some(m)` for dynamically-sized vectors.
    /// 
    /// # Returns
    /// 
    /// Dynamically-sized `N x M` matrix that is compatible with this length-`N` vector.
    /// 
    /// # Examples
    /// 
    /// ## Creating a dynamically-sized matrix compatible with a statically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{DMatrix, SVector};
    /// 
    /// // Create a statically-sized vector of length-2.
    /// let vec: SVector<f64, 2> = SVector::new_with_length(2);
    /// 
    /// // Create a dynamically-sized 2x3 matrix.
    /// let mat: DMatrix<f64> = vec.new_dmatrix_n_by_m(3);
    /// assert_eq!(mat.shape(), (2, 3));
    /// ```
    /// 
    /// ## Creating a dynamically-sized matrix compatible with a dynamically-sized vector
    /// 
    /// ```
    /// use linalg_traits::Vector;
    /// use nalgebra::{DMatrix, DVector};
    /// 
    /// // Create a dynamically-sized vector of length-2.
    /// let vec: DVector<f64> = DVector::new_with_length(2);
    /// 
    /// // Create a dynamically-sized 2x3 matrix.
    /// let mat: DMatrix<f64> = vec.new_dmatrix_n_by_m(3);
    /// assert_eq!(mat.shape(), (2, 3));
    /// ```
    fn new_dmatrix_n_by_m(&self, m: usize) -> Self::DMatrixNxM {
        Self::DMatrixNxM::new_with_shape(self.len(), m)
    }

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

    // -----------------------------
    // Required method declarations.
    // -----------------------------

    /// Determine whether or not the vector is statically-sized.
    /// 
    /// # Returns
    /// 
    /// `true` if the vector is statically-sized, `false` if the vector is dynamically-sized.
    fn is_statically_sized() -> bool;
    
    /// Determine whether or not the vector is dynamically-sized.
    /// 
    /// # Returns
    /// 
    /// `true` if the vector is dynamically-sized, `false` if the vector is statically-sized.
    fn is_dynamically_sized() -> bool;

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