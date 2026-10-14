use crate::{Mat, Matrix, RealField, RowVectorMulMat};

impl<R: RealField> RowVectorMulMat<R, Mat<R>> for Vec<R> {
    type Output = Vec<R>;

    fn mul_mat(&self, rhs: &Mat<R>) -> Vec<R> {
        let (m, n) = rhs.shape();
        self.assert_mul_mat_compatible(rhs);
        (0..n)
            .map(|j| {
                let mut sum = R::_zero();
                for i in 0..m {
                    sum += self[i] * rhs[(i, j)];
                }
                sum
            })
            .collect()
    }
}
