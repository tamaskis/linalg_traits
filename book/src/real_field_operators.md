# Operators

<!-- toc -->

## Addition

The following addition operations are supported for `T: RealField`:

* `T + T`
* `T + &T`
* `&T + T`
* `&T + &T`

**Example**

```rust
{{#include ../../examples/real_field_addition.rs}}
```

## Subtraction

The following subtraction operations are supported for `T: RealField`:

* `T - T`
* `T - &T`
* `&T - T`
* `&T - &T`

**Example**

```rust
{{#include ../../examples/real_field_subtraction.rs}}
```

## Multiplication

The following multiplication operations are supported for `T: RealField`:

* `T * T`
* `T * &T`
* `&T * T`
* `&T * &T`

**Example**

```rust
{{#include ../../examples/real_field_multiplication.rs}}
```

## Division

The following division operations are supported for `T: RealField`:

* `T / T`
* `T / &T`
* `&T / T`
* `&T / &T`

**Example**

```rust
{{#include ../../examples/real_field_division.rs}}
```

## Remainder after division

The following remainder after division operations are supported for `T: RealField`:

* `T % T`
* `T % &T`
* `&T % T`
* `&T % &T`

**Example**

```rust
{{#include ../../examples/real_field_remainder.rs}}
```

## Addition-assignment

The following addition-assignment operations are supported for `T: RealField`:

* `T += T`
* `T += &T`

**Example**

```rust
{{#include ../../examples/real_field_add_assign.rs}}
```

## Subtraction-assignment

The following subtraction-assignment operations are supported for `T: RealField`:

* `T -= T`
* `T -= &T`

**Example**

```rust
{{#include ../../examples/real_field_sub_assign.rs}}
```

## Multiplication-assignment

The following multiplication-assignment operations are supported for `T: RealField`:

* `T *= T`
* `T *= &T`

**Example**

```rust
{{#include ../../examples/real_field_mul_assign.rs}}
```

## Division-assignment

The following division-assignment operations are supported for `T: RealField`:

* `T /= T`
* `T /= &T`

**Example**

```rust
{{#include ../../examples/real_field_div_assign.rs}}
```

## Remainder-assignment

The following remainder-assignment operations are supported for `T: RealField`:

* `T %= T`
* `T %= &T`

**Example**

```rust
{{#include ../../examples/real_field_rem_assign.rs}}
```

## Equality and Inequality

The following equality/inequality operations are supported for `T: RealField`:

* `T == T`
* `T != T`

**Example**

```rust
{{#include ../../examples/real_field_equality.rs}}
```

## Ordering

The following ordering operations are supported for `T: RealField`:

* `T < T`
* `T <= T`
* `T > T`
* `T >= T`

**Example**

```rust
{{#include ../../examples/real_field_ordering.rs}}
```