# linalg-traits

[<img alt="github" src="https://img.shields.io/badge/github-tamaskis/linalg_traits-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/tamaskis/linalg_traits)
[<img alt="crates.io" src="https://img.shields.io/crates/v/linalg-traits.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/linalg-traits)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-linalg_traits-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/linalg-traits)

Traits for generic linear algebra.

## Documentation

Please see https://docs.rs/linalg-traits.

## Examples

Let's define a function that takes in a vector and returns a new vector with all the elements repeated twice. Using the `Scalar` and `Vector` traits, we can write it in a way that makes it independent of what types we use to represent scalars and vectors.

```rust
use linalg_traits::{Scalar, Vector};
use ndarray::{array, Array1};
use numtest::*;

// Define the function for repeating the elements.
fn repeat_elements<S: Scalar, V: Vector<S>>(v: &V) -> V {
    // Create a new vector of the same type but with twice the length.
    let mut v_repeated = V::new_with_length(v.len() * 2);

    // Populate the vector.
    for i in 0..v.len() {
        v_repeated[2 * i] = v[i];
        v_repeated[2 * i + 1] = v[i];
    }

    v_repeated
}

// Define the vector to be repeated.
let v: Array1<f64> = array![1.0, 2.0, 3.0];

// Repeat the elements.
let v_repeated: Array1<f64> = repeat_elements(&v);

// Check that the elements were properly repeated.
assert_arrays_equal!(v_repeated, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
```

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version 2.0</a> or 
<a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
</sub>