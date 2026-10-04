use crate::{Matrix, RealField, Vector};

/// Trait defining matrix-vector multiplication.
///
/// # Note
///
/// The matrix and vector must be from the same compatibility class (e.g. `nalgebra` with
/// `nalgebra`, `ndarray` with `ndarray`). Within a compatibility class, the operands may be
/// statically-sized, dynamically-sized, or a mix of both.
pub trait MulVec<R: RealField, V: Vector<R>>: Matrix<R> {
    /// Vector type resulting from multiplying this matrix by a vector of type `V`.
    type Output: Vector<R>;

    /// Assert that this `M x N` matrix can be multiplied (from the right) by a length-`N` vector.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector that this matrix would be multiplied by.
    ///
    /// # Panics
    ///
    /// * If the number of columns of this matrix is not equal to the length of `rhs`.
    fn assert_mul_vec_compatible(&self, rhs: &V) {
        let (m, n) = self.shape();
        assert_eq!(
            n,
            rhs.len(),
            "Matrix and vector have incompatible shapes for multiplication ({m}x{n} times length {}).",
            rhs.len(),
        );
    }

    /// Multiply this `M x N` matrix by a length-`N` vector.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The length-`N` vector to multiply this matrix by (from the right).
    ///
    /// # Returns
    ///
    /// The length-`M` vector `self * rhs`.
    ///
    /// # Panics
    ///
    /// * If the number of columns of this matrix is not equal to the length of `rhs`.
    ///
    /// # Examples
    ///
    /// ## Multiplying a statically-sized matrix by a dynamically-sized vector
    ///
    /// ```
    /// # #[cfg(feature = "nalgebra")]
    /// # {
    /// use linalg_traits::MulVec;
    /// use nalgebra::{DVector, SMatrix};
    ///
    /// // Create a statically-sized 2x3 matrix.
    /// let a: SMatrix<f64, 2, 3> = SMatrix::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    ///
    /// // Create a dynamically-sized length-3 vector.
    /// let v: DVector<f64> = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
    ///
    /// // The result is a dynamically-sized length-2 vector.
    /// let w: DVector<f64> = a.mul_vec(&v);
    /// assert_eq!(w, DVector::from_row_slice(&[14.0, 32.0]));
    /// # }
    /// ```
    ///
    /// ## Multiplying a dynamically-sized matrix by a dynamically-sized vector
    ///
    /// ```
    /// # #[cfg(feature = "ndarray")]
    /// # {
    /// use linalg_traits::MulVec;
    /// use ndarray::array;
    ///
    /// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// let v = array![1.0, 2.0, 3.0];
    ///
    /// assert_eq!(a.mul_vec(&v), array![14.0, 32.0]);
    /// # }
    /// ```
    fn mul_vec(&self, rhs: &V) -> <Self as MulVec<R, V>>::Output;
}
