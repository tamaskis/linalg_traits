use crate::{MulVec, RealField};
use nalgebra::{DMatrix, DVector, SVector};

// DMatrix * SVector.
impl<R, const N: usize> MulVec<R, SVector<R, N>> for DMatrix<R>
where
    R: RealField,
{
    type Output = DVector<R>;

    fn mul_vec(&self, rhs: &SVector<R, N>) -> DVector<R> {
        self.assert_mul_vec_compatible(rhs);
        let product = self * rhs;
        DVector::from_column_slice(product.as_slice())
    }
}

// DMatrix * DVector.
impl<R> MulVec<R, DVector<R>> for DMatrix<R>
where
    R: RealField,
{
    type Output = DVector<R>;

    fn mul_vec(&self, rhs: &DVector<R>) -> DVector<R> {
        self.assert_mul_vec_compatible(rhs);
        self * rhs
    }
}
