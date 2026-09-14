add_executable(trueform_cpp_spatial_tests
  test_closest_metric_point.cpp
  test_distance.cpp
  test_form_python_parity.cpp
  test_gather_ids.cpp
  test_intersects.cpp
  test_neighbor_search.cpp
  test_primitive.cpp
  test_primitive_template_defaults.cpp
  test_python_parity.cpp
  test_ray_cast.cpp
  test_signed_distance.cpp
  test_transformed.cpp
)
target_link_libraries(trueform_cpp_spatial_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_spatial_tests
  TEST_PREFIX "cpp::spatial::"
  PROPERTIES LABELS "cpp-spatial"
)
