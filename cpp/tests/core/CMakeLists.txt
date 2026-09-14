add_executable(trueform_cpp_core_tests
  test_async.cpp
  test_cache_preparation.cpp
  test_common_helpers.cpp
  test_elementwise.cpp
  test_edge_mesh.cpp
  test_histogram.cpp
  test_index_storage.cpp
  test_matrix.cpp
  test_mesh_door.cpp
  test_mesh_template_defaults.cpp
  test_nd_array_creation_structure.cpp
  test_nd_array_indexing.cpp
  test_nd_array_sorting.cpp
  test_obb.cpp
  test_offset_blocked_python_parity.cpp
  test_point_cloud_template_defaults.cpp
  test_read_after_shrink.cpp
  test_reductions.cpp
  test_storage_handles.cpp
  test_to_matrix.cpp
  test_views.cpp
)
target_link_libraries(trueform_cpp_core_tests PRIVATE
  trueform_cpp_test_common
)

catch_discover_tests(
  trueform_cpp_core_tests
  TEST_PREFIX "cpp::core::"
  PROPERTIES LABELS "cpp-core"
)
