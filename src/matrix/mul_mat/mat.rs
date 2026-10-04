use crate::{Mat, Matrix, MulMat, RealField};

impl<R: RealField> MulMat<R, Mat<R>> for Mat<R> {
    type Output = Mat<R>;

    fn mul_mat(&self, rhs: &Mat<R>) -> Mat<R> {
        let (m, k) = self.shape();
        let (_, n) = rhs.shape();
        self.assert_mul_mat_compatible(rhs);
        let mut out = Mat::<R>::new_with_shape(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = R::_zero();
                for l in 0..k {
                    sum += self[(i, l)] * rhs[(l, j)];
                }
                out[(i, j)] = sum;
            }
        }
        out
    }
}
