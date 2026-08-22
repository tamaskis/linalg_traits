use std::fmt::{Debug, Display};

/// Low-level base trait.
///
/// All of the traits required for this trait can be automatically derived except for [`Display`].
/// For example, a typical implementation often looks like
///
/// ```rust,ignore
/// #[derive(Copy, Clone, Debug, Default)]
/// pub struct MyStruct {
///     ...
/// }
///
/// impl Display for MyStruct {
///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
///         write!(f, ...)
///     }
/// }
/// ```
pub trait Base: Copy + Clone + Debug + Default + Display {}

// Provide a blanket implementation of `Base` for any type that implements `Copy`, `Clone`, `Debug`,
// `Default`, and `Display`.
impl<T> Base for T where T: Copy + Clone + Debug + Default + Display {}
