use crate::vector::vector_trait::Vector;

#[cfg(feature = "with_ndarray")]
use ndarray::Array1;

#[cfg(feature = "with_ndarray")]
impl Vector for Array1<f64> {
    fn new_with_length(len: usize) -> Self {
        Array1::<f64>::zeros(len)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn from_slice(slice: &[f64]) -> Self {
        Array1::from(slice.to_vec())
    }

    fn as_slice(&self) -> &[f64] {
        self.as_slice()
            .expect("The array's data is either not contiguous or not in standard order.")
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
