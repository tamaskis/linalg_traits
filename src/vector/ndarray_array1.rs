use crate::scalar::Scalar;
use crate::vector::vector_trait::Vector;

#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2, LinalgScalar, ScalarOperand};

#[cfg(feature = "ndarray")]
impl<S: Scalar + ScalarOperand + LinalgScalar> Vector<S> for Array1<S> {
    // Cannot apply ScalarOperand + LinalgScalar trait bounds on T because it would be more
    // restrictive than the trait definition.
    type VectorT<T: Scalar> = Vec<T>;

    // Cannot apply ScalarOperand + LinalgScalar trait bounds on T because it would be more
    // restrictive than the trait definition.
    type DVectorT<T: Scalar> = Vec<T>;

    type Vectorf64 = Array1<f64>;

    type DVectorf64 = Array1<f64>;

    type MatrixNxN = Array2<S>;

    type MatrixMxN<const M: usize> = Array2<S>;

    type DMatrixMxN = Array2<S>;

    type DMatrixMxNf64 = Array2<f64>;

    type MatrixNxM<const M: usize> = Array2<S>;

    type DMatrixNxM = Array2<S>;

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn new_with_length(len: usize) -> Self {
        Array1::<S>::zeros(len)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn from_slice(slice: &[S]) -> Self {
        Array1::from(slice.to_vec())
    }

    fn as_slice(&self) -> &[S] {
        self.as_slice()
            .expect("The array's data is either not contiguous or not in standard order.")
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn add_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a += *b;
        }
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn sub_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a -= *b;
        }
    }

    fn mul(&self, scalar: S) -> Self {
        self * scalar
    }

    fn mul_assign(&mut self, scalar: S) {
        for a in self.iter_mut() {
            *a *= scalar;
        }
    }

    fn div(&self, scalar: S) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: S) {
        for a in self.iter_mut() {
            *a /= scalar;
        }
    }

    fn dot(&self, other: &Self) -> S {
        Array1::<S>::dot(self, other)
    }
}
