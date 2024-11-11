use crate::vector::vector_trait::Vector;

impl Vector for Vec<f64> {
    fn new_with_length(len: usize) -> Vec<f64> {
        vec![0.0; len]
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn from_slice(slice: &[f64]) -> Vec<f64> {
        slice.to_vec()
    }

    fn as_slice(&self) -> &[f64] {
        self.as_slice()
    }

    fn add(&self, other: &Self) -> Self {
        self.assert_same_length(other);
        self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
    }

    fn add_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a += b;
        }
    }

    fn sub(&self, other: &Self) -> Self {
        self.assert_same_length(other);
        self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }

    fn sub_assign(&mut self, other: &Self) {
        self.assert_same_length(other);
        for (a, b) in self.iter_mut().zip(other.iter()) {
            *a -= b;
        }
    }

    fn mul(&self, scalar: f64) -> Self {
        self.iter().map(|a| a * scalar).collect()
    }

    fn mul_assign(&mut self, scalar: f64) {
        for a in self.iter_mut() {
            *a *= scalar;
        }
    }

    fn div(&self, scalar: f64) -> Self {
        self.iter().map(|a| a / scalar).collect()
    }

    fn div_assign(&mut self, scalar: f64) {
        for a in self.iter_mut() {
            *a /= scalar;
        }
    }
}
