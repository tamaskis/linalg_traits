use crate::{RealField, RowVectorMulMat};
use nalgebra::{DMatrix, RowDVector, SMatrix};

// RowDVector * SMatrix.
impl<R, const M: usize, const N: usize> RowVectorMulMat<R, SMatrix<R, M, N>> for RowDVector<R>
where
    R: RealField,
{
    type Output = RowDVector<R>;

    fn mul_mat(&self, rhs: &SMatrix<R, M, N>) -> RowDVector<R> {
        self.assert_mul_mat_compatible(rhs);
        let product = self * rhs;
        RowDVector::from_row_slice(product.as_slice())
    }
}

// RowDVector * DMatrix.
impl<R> RowVectorMulMat<R, DMatrix<R>> for RowDVector<R>
where
    R: RealField,
{
    type Output = RowDVector<R>;

    fn mul_mat(&self, rhs: &DMatrix<R>) -> RowDVector<R> {
        self.assert_mul_mat_compatible(rhs);
        self * rhs
    }
}
