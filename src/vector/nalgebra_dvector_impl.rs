use crate::vector::vector_trait::Vector;

#[cfg(feature = "with_nalgebra")]
use nalgebra::DVector;

#[cfg(feature = "with_nalgebra")]
impl Vector for DVector<f64> {
    fn new_with_length(len: usize) -> DVector<f64> {
        DVector::from_element(len, 0.0)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn from_slice(slice: &[f64]) -> Self {
        let mut result = DVector::new_with_length(slice.len());
        for (i, &item) in slice.iter().enumerate() {
            result[i] = item;
        }
        result
    }

    fn as_slice(&self) -> &[f64] {
        self.as_slice()
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn add_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a += b;
        }
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn sub_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a -= b;
        }
    }

    fn mul(&self, scalar: f64) -> Self {
        self * scalar
    }

    fn mul_assign(&mut self, scalar: f64) {
        for a in self.iter_mut() {
            *a *= scalar;
        }
    }

    fn div(&self, scalar: f64) -> Self {
        self / scalar
    }

    fn div_assign(&mut self, scalar: f64) {
        for a in self.iter_mut() {
            *a /= scalar;
        }
    }
}
