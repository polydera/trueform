add_executable(trueform_topology_tests
  test_boundary.cpp
  test_connectivity.cpp
  test_half_edges.cpp
  test_neighborhoods.cpp
  test_mesh_analysis.cpp
  test_connected_components.cpp
  test_duplicate_faces.cpp
  test_stitching.cpp
  test_planar_graph_regions.cpp
  test_face_hole_relations.cpp
  test_face_split_by_edges.cpp
  test_hole_patcher.cpp
  test_ear_cutter.cpp
)

target_link_libraries(trueform_topology_tests PRIVATE trueform_test_common)

target_compile_options(trueform_topology_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_topology_tests TEST_PREFIX "topology::")
