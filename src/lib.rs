//! [![github]](https://github.com/tamaskis/linalg_traits)&ensp;[![crates-io]](https://crates.io/crates/linalg-traits)&ensp;[![docs-rs]](https://docs.rs/linalg-traits)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! Traits for generic linear algebra.
//!
//! # Purpose
//!
//! The purpose of this crate is to provide traits defining common interfaces for linear algebra
//! types defined in various popular numerical computing crates in the Rust ecosystem. This is to
//! facilitate the development of crates requiring some linear algebra features (e.g. ODE solvers,
//! state estimation, etc.) without forcing users to use a specific numerical computing crate.
//!
//! _See [Additional notes on use cases](#additional-notes-on-use-cases)._
//!
//! #### Constraints
//!
//! 1. **Compatibility with [`Vec<f64>`], [`nalgebra`] types, and [`ndarray`] types.**
//!
//!     As a result, this crate does not require any operator overloads to be implemented for linear
//!     alegebra types. Different numerical computing crates may have different implementations for
//!     operator overloads (e.g. [`ndarray`] overloads `*` for elementwise multiplication, while
//!     [`nalgebra`] overloads `*` for matrix multiplication). This means that anyone writing generic
//!     linear algebra code using `linalg-traits` should use the arithmetic methods defined on the
//!     [`Vector`] and [`Matrix`] traits. <br><br>
//!
//! 1. **Compatibility with both statically-sized and dynamically-sized types.**
//!
//!     Most linear algebra types implement the [`Clone`] trait. Statically-sized linear algebra
//!     types (e.g. `nalgebra::SVector`) may also implement the [`Copy`] trait, which is often
//!     preferable to use for those types. However, it can be unsafe to copy dynamically-sized
//!     types, so to keep [`Vector`] and [`Matrix`] compatible with both statically and
//!     dynamically-sized linear algebra types, they only require that [`Clone`] be implemented.
//!
//! # Linear Algebra Traits
//!
//! This crate provides the following traits along with their implementations for the following
//! types:
//!
//! | Trait | Implementations on Foreign Types | Implementations on Local Types |
//! | ----- | -------------------------------- | ------------------------------ |
//! | [`Scalar`] | [`f64`] and all other types that satisfy its trait bounds. | N/A |
//! | [`Vector`] | [`Vec<S>`] <BR> [`nalgebra::DVector<S>`] <BR> [`nalgebra::SVector<S, N>`] <BR> [`ndarray::Array1<T>`] <BR><BR> Note:<BR>   • `S: Scalar` <BR>   • `T: Scalar + ndarray::ScalarOperand` + `ndarray::LinalgScalar` <BR>   • `N: usize` | N/A |
//! | [`Matrix`] | [`nalgebra::DMatrix<S>`] <BR> [`nalgebra::SMatrix<S, M, N>`] <BR> [`ndarray::Array2<T>`] <BR><BR> Note:<BR>   • `S: Scalar` <BR>   • `T: Scalar + ndarray::ScalarOperand` + `ndarray::LinalgScalar` <BR>   • `M: usize` <BR>   • `N: usize` | [`Mat<S>`] <BR><BR> Note:<BR>   • `S: Scalar` |
//!
//! See the [Using with `nalgebra` and `ndarray`](#using-with-nalgebra-and-ndarray) section further
//! down on this page for information on using the `linalg-traits` crate with types defined in
//! [`nalgebra`] and/or [`ndarray`].
//!
//! # Example
//!
//! Let's define a function that takes in a vector and returns a new vector with all the elements
//! repeated twice. Using the [`Scalar`] and [`Vector`] traits, we can write it in a way that makes
//! it independent of what types we use to represent scalars and vectors.
//!
//! ```
//! use linalg_traits::{Scalar, Vector};
//! use ndarray::{array, Array1};
//! use numtest::*;
//!
//! // Define the function for repeating the elements.
//! fn repeat_elements<S: Scalar, V: Vector<S>>(v: &V) -> V {
//!     // Create a new vector of the same type but with twice the length.
//!     let mut v_repeated = V::new_with_length(v.len() * 2);
//!
//!     // Populate the vector.
//!     for i in 0..v.len() {
//!         v_repeated[2 * i] = v[i];
//!         v_repeated[2 * i + 1] = v[i];
//!     }
//!
//!     v_repeated
//! }
//!
//! // Define the vector to be repeated.
//! let v: Array1<f64> = array![1.0, 2.0, 3.0];
//!
//! // Repeat the elements.
//! let v_repeated: Array1<f64> = repeat_elements(&v);
//!
//! // Check that the elements were properly repeated.
//! assert_arrays_equal!(v_repeated, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
//! ```
//!
//! # Using with `nalgebra` and `ndarray`
//!
//! `linalg-traits` provides implementations of the [`Vector`] and [`Matrix`] traits for linear
//! algebra types defined by [`nalgebra`] and [`ndarray`]. However, since you may not one to use one or
//! either of these crates in your project, `linalg-traits` has specifies both [`nalgebra`] and
//! [`ndarray`] as optional dependencies.
//!
//! If you _are_ using either of these crates in your project and want linear algebra types defined
//! by these crates to be identified as either [`Vector`]s or [`Matrix`]es, you should specify these
//! crates as features in the `linalg-traits` dependency in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! linalg-traits = { version = "x.y.z", features = ["nalgebra", "ndarray"] }
//! ```
//!
//! # Additional notes on use cases
//!
//! Say I have an ODE solver crate `my-ode-solver`. I want this crate to be compatible with both
//! [`ndarray`] and [`nalgebra`]. In the backend, I implement everything requiring vectors/matrices in
//! `my-ode-solver` using generic types with trait bounds on either [`Vector`] or [`Matrix`] (e.g.
//! `T: Vector`, `T: Matrix`). Now, any downstream user of `my-ode-solver` can choose whether they
//! want to use [`ndarray`] or [`nalgebra`]. Alternatively, if they have some custom linear algebra
//! types, they could implement [`Vector`] and [`Matrix`] for those custom types and use them
//! directly with `my-ode-solver`.
//!
//! This crate is _not_ trying to replace existing APIs in all situations. Continuing the example
//! from above, `my-ode-solver` will use the methods defined by the traits in `linalg-traits`.
//! However, if I'm a user of `my-ode-solver` and I'm using it with [`ndarray`], in my project I
//! should still use the APIs defined by [`ndarray`] (and not those defined by `linalg-traits`).

// Linter setup.
#![warn(missing_docs)]

// Module declarations.
pub(crate) mod matrix;
pub(crate) mod scalar;
pub(crate) mod vector;

// Re-exports.
pub use crate::matrix::mat::Mat;
pub use crate::matrix::matrix_trait::Matrix;
pub use crate::scalar::Scalar;
pub use crate::vector::vector_trait::Vector;
