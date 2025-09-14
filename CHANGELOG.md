# Changelog

## 0.12.1

1. Made some lifetimes explicit to address new lints.
1. Updated `nalgebra` optional dependency from `0.33.2` to `0.34.0`.
1. Updated `numtest` dev dependency from `0.3.0` to `0.3.1`.
1. Added `tracing-subscriber` dependency at `0.3.20` (since `<0.3.20` had a security vulnerability).

## 0.12.0

1. Updated `rust` version to 2024.
1. Updated `faer` optional dependency from `0.21.7` to `0.22.6`.
1. Updated `faer-traits` optional dependency from `0.21.0` to `0.22.1`.
1. Updated `numtest` dev dependency from `0.2.2` to `0.3.0`.

## 0.11.1

1. Added `as_row_slice` and `as_col_slice` methods to the `Matrix` trait.

## 0.11.0

1. Fixed issue with `faer` feature flags.

## 0.10.0

1. Implementation of `Vector` and `Matrix` traits for the `faer::Mat` struct.
1. No longer require implementations of `Index` and `IndexMut` for `Vector` trait.
1. `Vector` trait now defines the `vget` ("vector get") and `vset` ("vector set") methods for accessing and setting elements of the vector (replaces `Index` and `IndexMut`).
1. `Matrix::as_slice` now returns a `Cow<[S]>` instead of a `&[S]`.
    * This provides support for matrix types with non-contiguous data storage (e.g. `faer::Mat`).
1. Added `faer` optional dependency at `0.21.7`.
1. Updated `nalgebra` optional dependency from `0.33.0` to `0.33.2`.
1. Updated `ndarray` optional dependency from `0.16.0` to `0.16.1`.
1. Updated `num-traits` dependency from `0.2.18` to `0.2.19`.
1. Updated `numtest` dev dependency from `0.2.1` to `0.2.2`.

## 0.9.1

1. Added `DVectorf64` associated type to `Vector` trait.

## 0.9.0

1. Renamed the `Vector::VectorT` associated type to `Vector::VectorT`.
1. Added `DVectorT` associated type to `Vector` trait.

## 0.8.0

1. The `Vector` and `Matrix` traits now also require that types implementing them also implement `Debug` and `PartialEq`.

## 0.7.5

1. Added `new` method to `Scalar` trait for constructing a `Scalar` from an `f64`.

## 0.7.4

1. Added `GenericVector` associated type to `Vector` trait.

## 0.7.3

1. Added `DMatrixMxNf64` associated type to `Vector` trait.
1. Added `new_dmatrix_m_by_n_f64` method to `Vector` trait.

## 0.7.2

1. Added `Vectorf64` associated type to `Vector` trait.
1. Added `new_vector_f64` method to `Vector` trait.

## 0.7.1

1. Added `RemAssign<f64>` trait bound to `linalg_traits::Scalar` trait.

## 0.7.0

1. Re-wrote `Scalar` trait as an extension of the `num_traits::Float` trait.

## 0.6.0

1. Added specific associated types and methods to the `Vector` trait for constructing compatible dynamically-sized matrices for both statically-sized and dynamically-sized vectors.

## 0.5.0

1. Updated `nalgebra` optional dependency from `0.32.5` to `0.33.0`.
1. Updated `ndarray` optional dependency from `0.15.6` to `0.16.0`.
1. Updated `numtest` dev dependency from `0.1.6` to `0.2.0`.
1. Reverted module visibility updates from version `0.4.2`.

## 0.4.2

1. Updated visibility of trait implementations on foreign types to be public, instead of just public to this crate.

## 0.4.1

1. Switched from named features to implicit features for optional dependencies.

## 0.4.0

1. Added ability to construct matrices of compatible types from vectors, and vectors of compatible types from matrices.
1. Added methods to both the `Matrix` and `Vector` traits to determine if an instance of a type implementing either trait is statically-sized or dynamically-sized.

## 0.3.0

1. Initial definition of the `Matrix` trait.
1. Defined the `Mat` struct representing an extremely basic/minimal matrix type.
1. Initial implementation of the `Vector` trait for the following types:
    * `Mat<S>`
    * `nalgebra::SMatrix<S, M, N>`
    * `nalgebra::DMatrix<S>`
    * `ndarray::Array2<S>`

## 0.2.1.

1. Add the `dot` method to the `Vector` trait.

## 0.2.0

1. Added the `Scalar` trait.
1. Updated the `Vector` trait to be generic over all types that implement the `Scalar` trait.

## 0.1.3

1. Added the following methods to the `Vector` trait:
    * `from_slice`
    * `as_slice`
    * `add` / `add_assign`
    * `sub` / `sub_assign`
    * `mul` / `mul_assign`
    * `div` / `div_assign`

## 0.1.2

1. Added `clone` trait bound to the `Vector` trait.

## 0.1.1

1. Implemented the `Vector` trait for `ndarray::Array1<f64>`.

## 0.1.0

1. Initial definition of the `Vector` trait with `new_with_length`, `len`, and `is_empty` methods.
1. Initial implementation of the `Vector` trait for the following types:
    * `Vec<f64>`
    * `nalgebra::SVector<f64, N>`
    * `nalgebra::DVector<f64>`