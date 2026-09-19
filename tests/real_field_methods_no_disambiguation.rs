use linalg_traits::real_field::assert_no_trait_disambiguation_needed;

#[test]
fn test_real_field_methods_no_disambiguation() {
    assert_no_trait_disambiguation_needed::<f64>(1.0, 2.0, 3.0);
}
