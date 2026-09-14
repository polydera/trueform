add_executable(trueform_cpp_intersect_tests
  test_intersect.cpp
  test_intersection_curves_matrix.cpp
  test_python_parity.cpp
)
target_link_libraries(trueform_cpp_intersect_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_intersect_tests
  TEST_PREFIX "cpp::intersect::"
  PROPERTIES LABELS "cpp-intersect"
)
