use std::fmt::{Debug, Display};

/// Low-level base trait.
pub trait Base: Copy + Clone + Debug + Default + Display {}

// Blanket implementation.
impl<T> Base for T where T: Copy + Clone + Debug + Default + Display {}
