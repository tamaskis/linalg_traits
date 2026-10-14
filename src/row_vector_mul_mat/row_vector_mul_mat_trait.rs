use crate::{Matrix, RealField, RowVector};

/// Trait defining row vector-matrix multiplication.
///
/// # Note
///
/// The row vector and matrix must be from the same compatibility class (e.g. `nalgebra` with
/// `nalgebra`, `ndarray` with `ndarray`). Within a compatibility class, the operands may be
/// statically-sized, dynamically-sized, or a mix of both.
pub trait RowVectorMulMat<R: RealField, M: Matrix<R>>: RowVector<R> {
    /// Row vector type resulting from multiplying this row vector by a matrix of type `M`.
    type Output: RowVector<R>;

    /// Assert that this length-`M` row vector can be multiplied (from the left) by an `M x N`
    /// matrix.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The matrix that this row vector would be multiplied by.
    ///
    /// # Panics
    ///
    /// * If the length of this row vector is not equal to the number of rows of `rhs`.
    fn assert_mul_mat_compatible(&self, rhs: &M) {
        let (m, n) = rhs.shape();
        assert_eq!(
            self.len(),
            m,
            "Row vector and matrix have incompatible shapes for multiplication (length {} times {m}x{n}).",
            self.len(),
        );
    }

    /// Multiply this length-`M` row vector by an `M x N` matrix.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `M x N` matrix to multiply this row vector by (from the left).
    ///
    /// # Returns
    ///
    /// The length-`N` row vector `self * rhs`.
    ///
    /// # Panics
    ///
    /// * If the length of this row vector is not equal to the number of rows of `rhs`.
    ///
    /// # Examples
    ///
    /// ## Multiplying a statically-sized row vector by a dynamically-sized matrix
    ///
    /// ```
    /// # #[cfg(feature = "nalgebra")]
    /// # {
    /// use linalg_traits::RowVectorMulMat;
    /// use nalgebra::{DMatrix, RowDVector, RowSVector};
    ///
    /// // Create a statically-sized length-2 row vector.
    /// let v: RowSVector<f64, 2> = RowSVector::from_row_slice(&[1.0, 2.0]);
    ///
    /// // Create a dynamically-sized 2x3 matrix.
    /// let a: DMatrix<f64> = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    ///
    /// // The result is a dynamically-sized length-3 row vector.
    /// let w: RowDVector<f64> = v.mul_mat(&a);
    /// assert_eq!(w, RowDVector::from_row_slice(&[9.0, 12.0, 15.0]));
    /// # }
    /// ```
    ///
    /// ## Multiplying a `Vec` by a [`crate::Mat`]
    ///
    /// ```
    /// use linalg_traits::{Mat, Matrix, RowVectorMulMat};
    ///
    /// let a = Mat::<f64>::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let v = vec![1.0, 2.0];
    ///
    /// assert_eq!(v.mul_mat(&a), vec![9.0, 12.0, 15.0]);
    /// ```
    fn mul_mat(&self, rhs: &M) -> <Self as RowVectorMulMat<R, M>>::Output;
}
