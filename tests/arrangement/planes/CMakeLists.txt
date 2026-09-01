add_executable(trueform_arrangement_planes_tests
  test_mesh_plane_producers.cpp
  test_mesh_triangulation.cpp
  test_plane_arrangement_laws.cpp
  test_plane_canonicalize_wave.cpp
  test_plane_collision_parity.cpp
  test_plane_components.cpp
  test_plane_graph_fan_bits.cpp
  test_plane_mesh_carrier.cpp
  test_plane_radial_fans.cpp
  test_plane_refined_walls.cpp
  test_plane_refusal_recovery.cpp
  test_plane_wave_entrants.cpp
)

target_link_libraries(trueform_arrangement_planes_tests PRIVATE trueform_test_common)

target_compile_options(trueform_arrangement_planes_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_arrangement_planes_tests TEST_PREFIX "arrangement::planes::")
