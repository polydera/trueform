add_executable(trueform_cpp_remesh_tests
  test_python_parity.cpp
  test_remesh.cpp
)
target_link_libraries(trueform_cpp_remesh_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_remesh_tests
  TEST_PREFIX "cpp::remesh::"
  PROPERTIES LABELS "cpp-remesh"
)
