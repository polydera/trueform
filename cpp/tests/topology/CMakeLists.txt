add_executable(trueform_cpp_topology_tests
  test_domain_labels_matrix.cpp
  test_neighborhoods.cpp
  test_python_parity.cpp
  test_topology.cpp
  test_topology_analysis_matrix.cpp
)
target_link_libraries(trueform_cpp_topology_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_topology_tests
  TEST_PREFIX "cpp::topology::"
  PROPERTIES LABELS "cpp-topology"
)
