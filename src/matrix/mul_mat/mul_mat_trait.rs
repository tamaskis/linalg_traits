use crate::{Matrix, RealField};

/// Trait defining matrix-matrix multiplication.
///
/// # Note
///
/// Both operands must be [`Matrix`] types from the same compatibility class (e.g. `nalgebra` with
/// `nalgebra`, `ndarray` with `ndarray`). Within a compatibility class, the operands may be
/// statically-sized, dynamically-sized, or a mix of both.
pub trait MulMat<R: RealField, Rhs: Matrix<R>>: Matrix<R> {
    /// Matrix type resulting from multiplying this matrix by a matrix of type `Rhs`.
    type Output: Matrix<R>;

    /// Assert that this `M x K` matrix can be multiplied (from the right) by a `K x N` matrix.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The matrix that this matrix would be multiplied by.
    ///
    /// # Panics
    ///
    /// * If the number of columns of this matrix is not equal to the number of rows of `rhs`.
    fn assert_mul_mat_compatible(&self, rhs: &Rhs) {
        let (m, k) = self.shape();
        let (k_rhs, n) = rhs.shape();
        assert_eq!(
            k, k_rhs,
            "Matrices have incompatible shapes for multiplication ({m}x{k} times {k_rhs}x{n}).",
        );
    }

    /// Multiply this `M x K` matrix by a `K x N` matrix.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `K x N` matrix to multiply this matrix by (from the right).
    ///
    /// # Returns
    ///
    /// The `M x N` matrix `self * rhs`.
    ///
    /// # Panics
    ///
    /// * If the number of columns of this matrix is not equal to the number of rows of `rhs`.
    ///
    /// # Examples
    ///
    /// ## Multiplying a statically-sized matrix by a dynamically-sized matrix
    ///
    /// ```
    /// # #[cfg(feature = "nalgebra")]
    /// # {
    /// use linalg_traits::MulMat;
    /// use nalgebra::{DMatrix, SMatrix};
    ///
    /// // Create a statically-sized 2x3 matrix.
    /// let a: SMatrix<f64, 2, 3> = SMatrix::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    ///
    /// // Create a dynamically-sized 3x2 matrix.
    /// let b: DMatrix<f64> = DMatrix::from_row_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    ///
    /// // The result is a dynamically-sized 2x2 matrix.
    /// let c: DMatrix<f64> = a.mul_mat(&b);
    /// assert_eq!(c, DMatrix::from_row_slice(2, 2, &[58.0, 64.0, 139.0, 154.0]));
    /// # }
    /// ```
    ///
    /// ## Multiplying two dynamically-sized matrices
    ///
    /// ```
    /// # #[cfg(feature = "ndarray")]
    /// # {
    /// use linalg_traits::MulMat;
    /// use ndarray::array;
    ///
    /// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];
    ///
    /// assert_eq!(a.mul_mat(&b), array![[58.0, 64.0], [139.0, 154.0]]);
    /// # }
    /// ```
    fn mul_mat(&self, rhs: &Rhs) -> <Self as MulMat<R, Rhs>>::Output;
}
