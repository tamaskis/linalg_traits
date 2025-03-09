use crate::scalar::Scalar;
use crate::vector::vector_trait::Vector;

#[cfg(feature = "faer")]
use faer::{Mat, Scale};

#[cfg(feature = "faer-traits")]
use faer_traits::RealField;

#[cfg(feature = "faer")]
#[cfg(feature = "faer-traits")]
impl<S: Scalar + RealField> Vector<S> for Mat<S> {
    // Cannot apply RealField trait bound on T because it would be more restrictive than the trait
    // definition.
    type VectorT<T: Scalar> = Vec<T>;

    // Cannot apply RealField trait bound on T because it would be more restrictive than the trait
    // definition.
    type DVectorT<T: Scalar> = Vec<T>;

    type Vectorf64 = Mat<f64>;

    type DVectorf64 = Mat<f64>;

    type MatrixNxN = Mat<S>;

    type MatrixMxN<const M: usize> = Mat<S>;

    type DMatrixMxN = Mat<S>;

    type DMatrixMxNf64 = Mat<f64>;

    type MatrixNxM<const M: usize> = Mat<S>;

    type DMatrixNxM = Mat<S>;

    fn vget(&self, idx: usize) -> S {
        self[(idx, 0)]
    }

    fn vset(&mut self, idx: usize, value: S) {
        self[(idx, 0)] = value;
    }

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn new_with_length(len: usize) -> Self {
        Mat::<S>::zeros(len, 1)
    }

    fn len(&self) -> usize {
        self.nrows()
    }

    fn is_empty(&self) -> bool {
        self.nrows() == 0
    }

    fn from_slice(slice: &[S]) -> Self {
        Mat::<S>::from_fn(slice.len(), 1, |i, _| slice[i])
    }

    fn as_slice(&self) -> &[S] {
        self.col_as_slice(0)
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn add_assign(&mut self, other: &Self) {
        *self += other;
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn sub_assign(&mut self, other: &Self) {
        *self -= other;
    }

    fn mul(&self, scalar: S) -> Self {
        self * Scale(scalar)
    }

    fn mul_assign(&mut self, scalar: S) {
        *self *= Scale(scalar);
    }

    fn div(&self, scalar: S) -> Self {
        self / Scale(scalar)
    }

    fn div_assign(&mut self, scalar: S) {
        *self /= Scale(scalar);
    }

    fn dot(&self, other: &Self) -> S {
        self.assert_same_length(other);
        let mut dot_product = S::zero();
        for i in 0..self.len() {
            dot_product += self.vget(i) * other.vget(i);
        }
        dot_product
    }
}
