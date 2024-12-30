use crate::scalar::Scalar;
use crate::vector::vector_trait::Vector;

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, DVector};

#[cfg(feature = "nalgebra")]
impl<S: Scalar> Vector<S> for DVector<S> {
    type VectorT<T: Scalar> = DVector<T>;

    type DVectorT<T: Scalar> = DVector<T>;

    type Vectorf64 = DVector<f64>;

    type DVectorf64 = DVector<f64>;

    type MatrixNxN = DMatrix<S>;

    type MatrixMxN<const M: usize> = DMatrix<S>;

    type DMatrixMxN = DMatrix<S>;

    type DMatrixMxNf64 = DMatrix<f64>;

    type MatrixNxM<const M: usize> = DMatrix<S>;

    type DMatrixNxM = DMatrix<S>;

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn new_with_length(len: usize) -> DVector<S> {
        DVector::from_element(len, S::zero())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn from_slice(slice: &[S]) -> Self {
        let mut result = DVector::new_with_length(slice.len());
        for (i, &item) in slice.iter().enumerate() {
            result[i] = item;
        }
        result
    }

    fn as_slice(&self) -> &[S] {
        self.as_slice()
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
        self * scalar
    }

    fn mul_assign(&mut self, scalar: S) {
        *self *= scalar;
    }

    fn div(&self, scalar: S) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: S) {
        *self /= scalar;
    }

    fn dot(&self, other: &Self) -> S {
        self.dot(other)
    }
}
