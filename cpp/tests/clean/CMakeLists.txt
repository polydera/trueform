add_executable(trueform_cpp_clean_tests
  test_clean.cpp
  test_python_parity.cpp
)
target_link_libraries(trueform_cpp_clean_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_clean_tests
  TEST_PREFIX "cpp::clean::"
  PROPERTIES LABELS "cpp-clean"
)
