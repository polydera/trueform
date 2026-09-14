add_executable(trueform_cpp_csg_tests
  test_boolean.cpp
  test_boolean_python_parity.cpp
  test_csg_graph.cpp
  test_python_parity.cpp
  test_outer_shell.cpp
)
target_link_libraries(trueform_cpp_csg_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_csg_tests
  TEST_PREFIX "cpp::csg::"
  PROPERTIES LABELS "cpp-csg"
)
