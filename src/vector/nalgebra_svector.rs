use crate::scalar::Scalar;
use crate::vector::vector_trait::Vector;

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, SMatrix, SVector};

#[cfg(feature = "nalgebra")]
impl<const N: usize, S: Scalar> Vector<S> for SVector<S, N> {
    type MatrixNxN = SMatrix<S, N, N>;

    type MatrixMxN<const M: usize> = SMatrix<S, M, N>;

    type DMatrixMxN = DMatrix<S>;

    type MatrixNxM<const M: usize> = SMatrix<S, N, M>;

    type DMatrixNxM = DMatrix<S>;

    fn is_statically_sized() -> bool {
        true
    }

    fn is_dynamically_sized() -> bool {
        false
    }

    fn new_with_length(len: usize) -> Self {
        assert_eq!(len, N, "Length must match the fixed size of the SVector.");
        SVector::from_element(S::zero())
    }

    fn len(&self) -> usize {
        N
    }

    fn is_empty(&self) -> bool {
        false // SVector is never empty because it's fixed size.
    }

    fn from_slice(slice: &[S]) -> Self {
        let mut result = SVector::new_with_length(slice.len());
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
