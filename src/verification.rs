/// Verify at compile time that a type implements a trait.
///
/// This macro is primarily geared towards verifying that traits implemented via blanket
/// implementations are indeed implemented for a specific type.
///
/// # Arguments
///
/// * `$type` - The type to verify that implements the trait.
/// * `$trait` - The trait to verify that is implemented for the type.
///
/// # Returns
///
/// `true` if `$type` implements `$trait`; otherwise, it fails to compile.
///
/// # Usage
///
/// ```rust,ignore
/// const _: bool = verify_trait_implemented!($type: $trait);
/// ```
///
/// # Example
///
/// ```
/// use linalg_traits::{verify_trait_implemented, RealField};
///
/// const _: bool = verify_trait_implemented!(f64: RealField);
/// ```
#[macro_export]
macro_rules! verify_trait_implemented {
    ($type:ty: $trait:path) => {{
        #[allow(clippy::extra_unused_type_parameters)]
        const fn verify<T: $trait>() -> bool {
            true
        }
        verify::<$type>()
    }};
}
