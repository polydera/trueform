add_executable(trueform_cpp_iso_tests
  test_isobands.cpp
  test_isocontours.cpp
  test_python_parity.cpp
)
target_link_libraries(trueform_cpp_iso_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_iso_tests
  TEST_PREFIX "cpp::iso::"
  PROPERTIES LABELS "cpp-iso"
)
