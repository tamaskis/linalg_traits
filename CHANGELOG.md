# Changelog

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