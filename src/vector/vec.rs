use crate::matrix::mat::Mat;
use crate::scalar::Scalar;
use crate::vector::vector_trait::Vector;

impl<S: Scalar> Vector<S> for Vec<S> {
    type VectorT<T: Scalar> = Vec<T>;

    type DVectorT<T: Scalar> = Vec<T>;

    type Vectorf64 = Vec<f64>;

    type MatrixNxN = Mat<S>;

    type MatrixMxN<const M: usize> = Mat<S>;

    type DMatrixMxN = Mat<S>;

    type DMatrixMxNf64 = Mat<f64>;

    type MatrixNxM<const M: usize> = Mat<S>;

    type DMatrixNxM = Mat<S>;

    fn is_statically_sized() -> bool {
        false
    }

    fn is_dynamically_sized() -> bool {
        true
    }

    fn new_with_length(len: usize) -> Vec<S> {
        vec![S::zero(); len]
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn from_slice(slice: &[S]) -> Vec<S> {
        slice.to_vec()
    }

    fn as_slice(&self) -> &[S] {
        self.as_slice()
    }

    fn add(&self, other: &Self) -> Self {
        self.assert_same_length(other);
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| *a + *b)
            .collect()
    }

    fn add_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a += *b;
        }
    }

    fn sub(&self, other: &Self) -> Self {
        self.assert_same_length(other);
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| *a - *b)
            .collect()
    }

    fn sub_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a -= *b;
        }
    }

    fn mul(&self, scalar: S) -> Self {
        self.iter().map(|a| *a * scalar).collect()
    }

    fn mul_assign(&mut self, scalar: S) {
        for a in self.iter_mut() {
            *a *= scalar;
        }
    }

    fn div(&self, scalar: S) -> Self {
        self.iter().map(|a| *a / scalar).collect()
    }

    fn div_assign(&mut self, scalar: S) {
        for a in self.iter_mut() {
            *a /= scalar;
        }
    }

    fn dot(&self, other: &Self) -> S {
        if self.len() != other.len() {
            panic!("Cannot evaluate the dot product of vectors with different lengths.");
        }
        let mut result = S::zero();
        for i in 0..self.len() {
            result += self[i] * other[i];
        }
        result
    }
}
