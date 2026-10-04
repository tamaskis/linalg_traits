use crate::{MulVec, RealField};
use nalgebra::{DVector, SMatrix, SVector};

// SMatrix * SVector.
impl<R, const M: usize, const N: usize> MulVec<R, SVector<R, N>> for SMatrix<R, M, N>
where
    R: RealField,
{
    type Output = SVector<R, M>;

    fn mul_vec(&self, rhs: &SVector<R, N>) -> SVector<R, M> {
        self * rhs
    }
}

// SMatrix * DVector.
impl<R, const M: usize, const N: usize> MulVec<R, DVector<R>> for SMatrix<R, M, N>
where
    R: RealField,
{
    type Output = DVector<R>;

    fn mul_vec(&self, rhs: &DVector<R>) -> DVector<R> {
        self.assert_mul_vec_compatible(rhs);
        let product = self * rhs;
        DVector::from_column_slice(product.as_slice())
    }
}
