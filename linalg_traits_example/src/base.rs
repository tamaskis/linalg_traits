//! Implements [`linalg_traits::real_field::Base`] for [`crate::my_real::MyReal`].

use crate::my_real::MyReal;
use std::fmt::Display;

// If we didn't have a `#[derive(Copy)]`, we'd have to do:
// impl Copy for MyReal {}

// If we didn't have a `#[derive(Clone)]`, we'd have to do:
// impl Clone for MyReal {
//     fn clone(&self) -> Self {
//         Self { x: self.x }
//     }
// }

// If we didn't have a `#[derive(Debug)]`, we'd have to do:
// impl std::fmt::Debug for MyReal {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("MyReal")
//             .field("x", &self.x)
//             .finish()
//     }
// }

// If we didn't have a `#[derive(Default)]`, we'd have to do:
// impl Default for MyReal {
//     fn default() -> Self {
//         Self { x: 0.0 }
//     }
// }

impl Display for MyReal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linalg_traits::real_field::assert_base;

    #[test]
    fn test_base() {
        assert_base::<MyReal>(
            MyReal::new(1.5),
            MyReal::new(0.0),
            "MyReal { x: 1.5 }",
            "1.5",
        );
    }
}
