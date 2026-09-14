add_executable(trueform_cpp_geometry_tests
  test_curvature.cpp
  test_curvature_matrix.cpp
  test_edges.cpp
  test_measurements.cpp
  test_mesh_measurement_matrix.cpp
  test_mesh_primitives.cpp
  test_normals.cpp
  test_orientation.cpp
  test_orientation_matrix.cpp
  test_python_parity.cpp
  test_registration.cpp
  test_smoothing.cpp
  test_triangulate.cpp
)
target_link_libraries(trueform_cpp_geometry_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_geometry_tests
  TEST_PREFIX "cpp::geometry::"
  PROPERTIES LABELS "cpp-geometry"
)
