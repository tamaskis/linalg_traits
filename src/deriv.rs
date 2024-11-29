#[cfg(test)]
mod tests {
    use crate::dual::Dual;
    use crate::scalar::Scalar;

    macro_rules! sderivative {
        ($f:expr, $x0:expr) => {{
            let x = Dual::new($x0, 1.0);
            let f_eval = $f(x);
            f_eval.get_dual()
        }};
    }

    pub fn func<S: Scalar>(x: S) -> S {
        x.powi(2)
    }

    #[test]
    fn example() {
        let result = sderivative!(func, 2.0);
        assert_eq!(result, 4.0);
        // println!("Derivative: {}", result);
    }
}
