add_executable(trueform_cut_tests
  test_arrangement_curves.cpp
  test_component_labels.cpp
  test_boolean.cpp
  test_boolean_pair.cpp
  test_consolidate_region_merges.cpp
  test_coplanar_loop_pairs.cpp
  test_cut_graph.cpp
  test_cut_graph_self.cpp
  test_embedded_intersection_curves.cpp
  test_face_cutter.cpp
  test_face_cuts.cpp
  test_face_regions.cpp
  test_has_adjacent_region_split.cpp
  test_identify_resolved_crossings.cpp
  test_region_refinement_identity.cpp
  test_region_split_proposals.cpp
  test_region_triangulator.cpp
  test_mesh_arrangements.cpp
  test_segment_arrangements.cpp
  test_self_arrangements.cpp
  test_settle_region_triangulations.cpp
)

target_link_libraries(trueform_cut_tests PRIVATE trueform_test_common)

target_compile_options(trueform_cut_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_cut_tests TEST_PREFIX "cut::")
