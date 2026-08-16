//! Defines the [`MyReal`] struct.

#[derive(Copy, Clone, Debug, Default)]
pub struct MyReal {
    pub(crate) x: f64,
}

impl MyReal {
    pub fn new(x: f64) -> Self {
        Self { x }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let x = MyReal::new(1.5);
        let y = MyReal { x: 1.5 };
        assert_eq!(x.x, y.x);
    }
}
