//! [![github]](https://github.com/tamaskis/linalg_traits)&ensp;[![crates-io]](https://crates.io/crates/linalg-traits)&ensp;[![docs-rs]](https://docs.rs/linalg-traits)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! Traits for generic linear algebra..
//!
//! # Purpose
//!
//! The purpose of this crate is to provide traits defining common interfaces for linear algebra
//! types defined in various popular numerical computing crates in the Rust ecosystem. This is to
//! facilitate the development of crates requiring some linear algebra features (e.g. ODE solvers,
//! state estimation, etc.) without forcing users to use a specific numerical computing crate.
//!
//! # Traits
//!
//! This crate provides the following traits along with their implementations for the following
//! types:
//!
//! | Trait | Implementations on Foreign Types |
//! | ----- | -------------------------------- |
//! | [`Vector`] | [`Vec<f64>`], `nalgebra::DVector<f64>`, `nalgebra::SVector<f64, D>` |
//! | [`Matrix`] | _not yet implemented_ |
//!
//! See the [Using with `nalgebra` and `ndarray`](#using-with-nalgebra-and-ndarray) section further
//! down on this page for information on using the `linalg-traits` crate with types defined in
//! `nalgebra` and/or `ndarray`.
//!
//! # Examples
//!
//! Let's define a function that takes in a vector and returns a new vector with all the elements
//! repeated twice. Using the [`Vector`] trait, we can write it in a way that makes it independent
//! of what struct we use to represent a vector.
//!
//! ```
//! use linalg_traits::Vector;
//! use numtest::*;
//!
//! // Define the function for repeating the elements.
//! fn repeat_elements<T: Vector>(v: &T) -> T {
//!     // Create a new vector of the same type but with twice the length.
//!     let mut v_repeated = T::new_with_length(v.len() * 2);
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
//! let v: Vec<f64> = vec![1.0, 2.0, 3.0];
//!
//! // Repeat the elements.
//! let v_repeated = repeat_elements(&v);
//!
//! // Check that the elements were properly repeated.
//! assert_arrays_equal!(v_repeated, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
//! ```
//!
//! # Using with `nalgebra` and `ndarray`
//!
//! `linalg-traits` provides implementations of the [`Vector`] and [`Matrix`] traits for linear
//! algebra types defined by `nalgebra` and `ndarray`. However, since you may not one to use one or
//! either of these crates in your project, `linalg-traits` has specifies both `nalgebra` and
//! `ndarray` as optional dependencies.
//!
//! If you _are_ using either of these crates in your project and want linear algebra types defined
//! by these crates to be identified as either [`Vector`]s or [`Matrix`]es, you should specify these
//! crates as features in the `linalg-traits` dependency in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! linalg-traits = { version = "x.y.z", features = ["nalgebra", "ndarray"] }
//! ```

// Linter setup.
#![warn(missing_docs)]

// Linking project modules.
pub(crate) mod matrix;
pub(crate) mod vector;

// Re-exports.
pub use crate::matrix::matrix_trait::Matrix;
pub use crate::vector::vector_trait::Vector;
