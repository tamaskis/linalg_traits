use crate::real_field::real_field_base::RealFieldBase;
use crate::real_field::real_field_faer::RealFieldFaer;
use crate::real_field::real_field_nalgebra::RealFieldNalgebra;
use crate::real_field::real_field_ndarray::RealFieldNdarray;

// TODO: define implementation sequence

/// Trait defining a real number.
///
/// # Overview
///
/// This trait defines the core functionality expected from any real number type, with the following
/// operators and methods implemented:
///
/// * All operations provided by [`crate::real_field_operations::RealFieldOperations`].
/// * All methods provided by [`crate::real_field::RealFieldBase`].
/// * All methods provided by [`num_traits::Zero`].
/// * All methods provided by [`num_traits::One`].
///
/// This set of operations and methods is a (99%) complete superset of all functionality provided by
/// [`f64`], [`num_traits::Float`] (and its supertraits), [`nalgebra::RealField`]
/// (and its supertraits), and [`faer_traits::RealField`] (and its supertraits). The methods that
/// are not included are ones that are complex number specific or more of a backend features of
/// [`nalgebra::RealField`] or [`faer_traits::RealField`].
///
/// # Background
///
/// This trait serves as the foundational interface for real number types that is cross-compatible
/// across different linear algebra and numerical computation crates. Consequently, it originated as
/// a trait that defined the superset of functionality provided by [`nalgebra::RealField`] and
/// [`faer_traits::RealField`]. However, real numbers are used well beyond these specific crates,
/// with many common operations not required by these crates (for example,
/// [`faer_traits::RealField`] doesn't require that a real number implement a `sin` method, which is
/// one of the most widely used mathematical functions). As a result, this trait defines the
/// superset (mostly; there are some methods missing) of functionality provided by
/// [`num_traits::Float`], [`nalgebra::RealField`], and [`faer_traits::RealField`].
///
/// # Interoperability with [`f64`]s.
///
/// We enforce that real types be interoperable with [`f64`]s. Some common differentiation methods,
/// notably forward-mode automatic differentation and complex-step differentiation, rely on
/// replacing real numbers with a custom type of number that has its own arithmetic (dual numbers
/// for forward-mode automatic differentiation, complex numbers for complex-step differentiation).
/// Forcing scalars to have this interoperability with [`f64`]s built-in helps enable downstream
/// crates to write functions in way that can be used with both plain [`f64`]s for most use cases,
/// and with custom types when the functions need to be differentiated.
///
/// Additionally, we chose to restrict this interoperability to be with [`f64`]s since
/// double-precision floating point numbers are the de facto standard for numerical computations.
///
/// This interoperability is mandated by [`crate::real_field_operations::F64Interop`] (a supertrait
/// of [`crate::real_field_operations::RealFieldOperations`]).
pub trait RealField: RealFieldBase + RealFieldFaer + RealFieldNalgebra + RealFieldNdarray {
    /// Construct an instance of this scalar from an [`f64`].
    ///
    /// # Arguments
    ///
    /// * `x` - An [`f64`].
    ///
    /// # Return
    ///
    /// An instance of this scalar type constructed from an [`f64`].
    #[must_use]
    #[inline]
    fn new(x: f64) -> Self {
        <Self as From<f64>>::from(x)
    }
}

// Blanket implementation.
impl<T> RealField for T where T: RealFieldBase + RealFieldFaer + RealFieldNalgebra + RealFieldNdarray
{}
