use crate::{Mat, Matrix, MulVec, RealField};

impl<R: RealField> MulVec<R, Vec<R>> for Mat<R> {
    type Output = Vec<R>;

    fn mul_vec(&self, rhs: &Vec<R>) -> Vec<R> {
        let (m, n) = self.shape();
        self.assert_mul_vec_compatible(rhs);
        (0..m)
            .map(|i| {
                let mut sum = R::_zero();
                for j in 0..n {
                    sum += self[(i, j)] * rhs[j];
                }
                sum
            })
            .collect()
    }
}
