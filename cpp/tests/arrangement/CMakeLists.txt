add_executable(trueform_cpp_arrangement_tests
  test_arrangements.cpp
  test_python_parity.cpp
)
target_link_libraries(trueform_cpp_arrangement_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_arrangement_tests
  TEST_PREFIX "cpp::arrangement::"
  PROPERTIES LABELS "cpp-arrangement"
)
