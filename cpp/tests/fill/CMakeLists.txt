add_executable(trueform_cpp_fill_tests
  test_fill.cpp
)
target_link_libraries(trueform_cpp_fill_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_fill_tests
  TEST_PREFIX "cpp::fill::"
  PROPERTIES LABELS "cpp-fill"
)
