#[cfg(test)]
mod tests {
    use crate::dual::Dual;
    use crate::scalar::Scalar;
    use crate::vector::vector_trait::Vector;

    macro_rules! sderivative {
        ($f:expr, $x0:expr) => {{
            let x = Dual::new($x0, 1.0);
            let f_eval = $f(x);
            f_eval.get_dual()
        }};
    }

    macro_rules! vderivative {
        ($f:expr, $x0:expr) => {{
            let x = Dual::new($x0, 1.0);
            let f_eval = $f(x);
            // might work but need to swap internal type
            for i in 0..f_eval.len() {}
            f_eval.get_dual()
        }};
    }

    pub fn func<S: Scalar>(x: S) -> S {
        x.powi(2)
    }

    pub fn func2<S: Scalar>(x: S) -> Vec<S> {
        vec![x.sin(), x.cos()]
    }

    #[test]
    fn example() {
        let result = sderivative!(func, 5.0);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn example_2() {
        let result = vderivative!(func2, 5.0);
        assert_eq!(result, 10.0);
    }
}
