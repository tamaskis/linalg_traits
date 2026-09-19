use crate::real_field::real_field_operations::f64_interop::f64_interop::F64Interop;
use crate::real_field::real_field_operations::real_field_operations_ref_ops::RealFieldOperationsRefOps;
use crate::real_field::real_field_operations::real_field_operations_rem::RealFieldOperationsRem;

/// Trait defining all arithmetic operations that need to be supported by real fields.
///
/// This trait extends far beyond what is required by popular linear algebra crates (e.g.
/// [`nalgebra`], [`faer`]) and is instead geared towards making generic real numbers as ergonomic
/// as possible (including being highly interoperable with [`f64`], which is the de facto standard
/// floating point representation). This is especially useful for fully-generic implementations that
/// are required for forward-mode automatic differentiation (see the
/// [`numdiff` crate documentation](https://docs.rs/numdiff/latest/numdiff/)).
///
/// # Supported Operators
///
/// These operators are supported by the [`RealFieldOperations`] trait and can be substituted in for
/// "`op`" in all the definitions below:
///
/// `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`
///
/// # Supported Operations
///
/// ## Unary negation
/// * `-T`
/// * `-&T`
///
/// ## Binary operations between two `T`'s
///
/// * `T op T`
/// * `T op &T`
/// * `&T op T`
/// * `&T op &T`
///
/// ## Binary-assignment operations between two `T`'s
///
/// * `T op= T`
/// * `T op= &T`
///
/// ## Binary operations between `T` and `f64` with `T` on the left-hand side
///
/// * `T op f64`
/// * `T op &f64`
/// * `&T op f64`
/// * `&T op &f64`
///
/// ## Binary-assignment operations between `T` and `f64` with `T` on the left-hand side
///
/// * `T op= f64`
/// * `T op= &f64`
///
/// ## Binary operations between `f64` and `T` with `f64` on the left-hand side
///
/// * `f64 op T`
/// * `f64 op &T`
/// * `&f64 op T`
/// * `&f64 op &T`
///
/// ## Binary-assignment operations between `T` and `f64` with `f64` on the left-hand side
///
/// **These operations are NOT implemented.** This is because the primary use case of
/// [`crate::RealField`] (for which [`RealFieldOperations`] is a supertrait) is for use in a generic
/// autodiff backend (e.g. [`numdiff` crate documentation](https://docs.rs/numdiff/latest/numdiff/))
/// where the types that will be substituted in for `T` are typically more complex types (e.g.
/// [`numdiff::Dual`](https://docs.rs/numdiff/latest/numdiff/struct.Dual.html),
/// [`numdiff::HyperDual`](https://docs.rs/numdiff/latest/numdiff/struct.HyperDual.html)). It is
/// perfectly safe "up-casting" from an `f64` to a one of these types (no information is lost), but
/// the reverse is not true (i.e., information will be lost). This is why binary-assignment
/// operations with an `f64` on the left-hand side are not implemented, because doing something like
/// `1.0 += Dual::new(2.0, 3.0)` would lose information.
///
/// # How To Implement
///
/// To implement this trait for a type `T`:
///
/// 1. Implement [`crate::real_field::RealFieldBase`] (which includes the base arithmetic methods).
/// 2. Call [`crate::impl_real_field_operations!`] on the type `T`.
///
/// This trait definition is quite complex, and is further complicated by having to satisfy multiple
/// optional third party dependencies (e.g. see the [`RealFieldOperationsRefOps`] docs as an example
/// for where things get complicated). If someone really wanted to, they _could_ trace through the
/// extensive set of trait requirements and manually implement all of the required traits manually,
/// but implementing [`crate::real_field::RealFieldBase`] and then
/// letting [`crate::impl_real_field_operations!`] take care of the rest is far easier.
pub trait RealFieldOperations:
    RealFieldOperationsRefOps + RealFieldOperationsRem + PartialEq + PartialOrd + F64Interop
{
}

// Blanket implementation.
impl<T> RealFieldOperations for T where
    T: RealFieldOperationsRefOps + RealFieldOperationsRem + PartialEq + PartialOrd + F64Interop
{
}
