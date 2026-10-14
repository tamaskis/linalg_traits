use crate::{MulMat, RealField};
use nalgebra::{DMatrix, SMatrix};

// SMatrix * SMatrix.
impl<R, const M: usize, const K: usize, const N: usize> MulMat<R, SMatrix<R, K, N>>
    for SMatrix<R, M, K>
where
    R: RealField,
{
    type Output = SMatrix<R, M, N>;

    fn mul_mat(&self, rhs: &SMatrix<R, K, N>) -> SMatrix<R, M, N> {
        self * rhs
    }
}

// SMatrix * DMatrix.
impl<R, const M: usize, const K: usize> MulMat<R, DMatrix<R>> for SMatrix<R, M, K>
where
    R: RealField,
{
    type Output = DMatrix<R>;

    fn mul_mat(&self, rhs: &DMatrix<R>) -> DMatrix<R> {
        self.assert_mul_mat_compatible(rhs);
        let product = self * rhs;
        DMatrix::from_column_slice(product.nrows(), product.ncols(), product.as_slice())
    }
}
