use std::fmt::Display;

#[derive(Copy, Clone, Debug, Default)]
pub struct MyScalar {
    pub(crate) x: f64,
}

impl MyScalar {
    pub fn new(x: f64) -> Self {
        Self { x }
    }
}

impl Display for MyScalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.x)
    }
}

impl From<f64> for MyScalar {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl From<MyScalar> for f64 {
    fn from(value: MyScalar) -> Self {
        value.x
    }
}

// `MyScalar` automatically satisfies `linalg_traits::real_field::Base`'s blanket implementation
// (`Copy + Clone + Debug + Default + Display`), so no explicit `impl` is required.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy() {
        let scalar = MyScalar { x: 1.5 };
        let copied_scalar = scalar;
        assert_eq!(scalar.x, copied_scalar.x);
    }

    #[test]
    fn test_clone() {
        let scalar = MyScalar { x: 1.5 };
        #[allow(clippy::clone_on_copy)]
        let cloned_scalar = scalar.clone();
        assert_eq!(scalar.x, cloned_scalar.x);
    }

    #[test]
    fn test_debug() {
        let scalar = MyScalar { x: 1.5 };
        assert_eq!(format!("{scalar:?}"), "MyScalar { x: 1.5 }");
    }
}
