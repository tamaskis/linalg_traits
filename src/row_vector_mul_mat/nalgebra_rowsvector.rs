use crate::{RealField, RowVectorMulMat};
use nalgebra::{DMatrix, RowDVector, RowSVector, SMatrix};

// RowSVector * SMatrix.
impl<R, const M: usize, const N: usize> RowVectorMulMat<R, SMatrix<R, M, N>> for RowSVector<R, M>
where
    R: RealField,
{
    type Output = RowSVector<R, N>;

    fn mul_mat(&self, rhs: &SMatrix<R, M, N>) -> RowSVector<R, N> {
        self * rhs
    }
}

// RowSVector * DMatrix.
impl<R, const M: usize> RowVectorMulMat<R, DMatrix<R>> for RowSVector<R, M>
where
    R: RealField,
{
    type Output = RowDVector<R>;

    fn mul_mat(&self, rhs: &DMatrix<R>) -> RowDVector<R> {
        self.assert_mul_mat_compatible(rhs);
        let product = self * rhs;
        RowDVector::from_row_slice(product.as_slice())
    }
}
