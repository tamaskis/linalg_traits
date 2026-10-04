use crate::{MulMat, RealField};
use nalgebra::{DMatrix, SMatrix};

// DMatrix * SMatrix.
impl<R, const K: usize, const N: usize> MulMat<R, SMatrix<R, K, N>> for DMatrix<R>
where
    R: RealField,
{
    type Output = DMatrix<R>;

    fn mul_mat(&self, rhs: &SMatrix<R, K, N>) -> DMatrix<R> {
        self.assert_mul_mat_compatible(rhs);
        let product = self * rhs;
        DMatrix::from_column_slice(product.nrows(), product.ncols(), product.as_slice())
    }
}

// DMatrix * DMatrix.
impl<R> MulMat<R, DMatrix<R>> for DMatrix<R>
where
    R: RealField,
{
    type Output = DMatrix<R>;

    fn mul_mat(&self, rhs: &DMatrix<R>) -> DMatrix<R> {
        self.assert_mul_mat_compatible(rhs);
        self * rhs
    }
}
