add_executable(trueform_topology_tests
  test_boundary.cpp
  test_connectivity.cpp
  test_tet_connectivity.cpp
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
  test_constrained_delaunay_triangulator.cpp
  test_delaunay_tetrahedralizer.cpp
  test_tet_collapse.cpp
  test_tet_segment_recovery.cpp
  test_tet_face_recovery.cpp
  test_tet_empty_sites.cpp
  test_tet_hull_order.cpp
  test_tet_row_admission.cpp
  test_cdt_constraint_owner_api.cpp
  test_cdt_meeting_landings.cpp
  test_cdt_half_edge_carrier_lifecycle.cpp
  test_cdt_parallel_topology_equivalence.cpp
  test_cdt_unconstrained_face_emission.cpp
  test_cdt_refiner.cpp
  test_make_cdt.cpp
  test_domain_labels.cpp
  test_face_orientation.cpp
  test_euler_characteristic.cpp
  test_boundary_rims.cpp
  test_hole_flap_predicates.cpp
  test_vertex_split.cpp
)

target_link_libraries(trueform_topology_tests PRIVATE trueform_test_common)

target_compile_options(trueform_topology_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

tf_discover_tests(trueform_topology_tests "topology::")

# The deep-lattice suites over the house int128 class; they read no builder.
if(TARGET trueform_test_int128_class)
  add_executable(trueform_topology_int128_class_tests
    test_delaunay_tetrahedralizer.cpp
    test_tet_segment_recovery.cpp
    test_tet_face_recovery.cpp
  )
  target_link_libraries(trueform_topology_int128_class_tests PRIVATE
    trueform_test_int128_class)
  tf_discover_tests(trueform_topology_int128_class_tests
    "topology_int128_class::")
endif()
