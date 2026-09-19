use std::fmt::{Debug, Display};

/// Low-level base trait.
pub trait Base: Copy + Clone + Debug + Default + Display {}

// Blanket implementation.
impl<T> Base for T where T: Copy + Clone + Debug + Default + Display {}

#[cfg(test)]
mod tests {
    use super::Base;

    #[test]
    fn test_f64_implements_base() {
        fn assert_base<T: Base>() {}

        assert_base::<f64>();
        assert_eq!(f64::default(), 0.0);
        assert_eq!(format!("{}", 1.5_f64), "1.5");
        assert_eq!(format!("{:?}", 1.5_f64), "1.5");
    }
}
