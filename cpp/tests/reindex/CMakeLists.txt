add_executable(trueform_cpp_reindex_tests
  test_python_parity.cpp
  test_reindex.cpp
  test_split_components_matrix.cpp
)
target_link_libraries(trueform_cpp_reindex_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_reindex_tests
  TEST_PREFIX "cpp::reindex::"
  PROPERTIES LABELS "cpp-reindex"
)
