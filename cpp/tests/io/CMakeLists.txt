add_executable(trueform_cpp_io_tests
  test_io.cpp
  test_python_parity.cpp
)
target_link_libraries(trueform_cpp_io_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_io_tests
  TEST_PREFIX "cpp::io::"
  PROPERTIES LABELS "cpp-io"
)
