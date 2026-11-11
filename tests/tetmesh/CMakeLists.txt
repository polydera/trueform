add_executable(trueform_tetmesh_tests
  test_carry_tet_regions.cpp
  test_count_tet_drain_member_openness.cpp
  test_land_tet_reach_points.cpp
  test_tet_side_points.cpp
  test_tet_drain_names.cpp
  test_tet_drain_measure.cpp
  test_make_tet_mesh.cpp
  test_tet_mesh_api.cpp
  test_tet_mesh_boxes.cpp
  test_tet_mesh_covers.cpp
  test_tet_weld_entrants.cpp
  test_tet_identity_authority.cpp
)

target_link_libraries(trueform_tetmesh_tests PRIVATE trueform_test_common)
target_compile_definitions(trueform_tetmesh_tests PRIVATE
  TF_TETMESH_DRAGON="${PROJECT_SOURCE_DIR}/benchmarks/data/dragon-50k.stl")

target_compile_options(trueform_tetmesh_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

# make_tet_mesh runs the parallel Delaunay build at the machine's full
# concurrency, so the whole suite runs alone.
catch_discover_tests(trueform_tetmesh_tests TEST_PREFIX "tetmesh::"
  PROPERTIES RUN_SERIAL TRUE)

# The same suite over the house int128 class, its builders compiled with it.
if(TARGET trueform_test_int128_class)
  add_executable(trueform_tetmesh_int128_class_tests
    test_carry_tet_regions.cpp
    test_count_tet_drain_member_openness.cpp
    test_land_tet_reach_points.cpp
    test_tet_side_points.cpp
    test_tet_drain_names.cpp
    test_tet_drain_measure.cpp
    test_make_tet_mesh.cpp
    test_tet_mesh_api.cpp
    test_tet_mesh_covers.cpp
  test_tet_weld_entrants.cpp
    ../builders/tetmesh_entrance_float.cpp
    ../builders/tetmesh_float_default.cpp
    ../builders/tetmesh_float_framed.cpp
    ../builders/tetmesh_float_link.cpp
    ../builders/tetmesh_float_tagged.cpp
    ../builders/tetmesh_float_tree.cpp
    ../builders/tetmesh_float.cpp
    ../builders/tetmesh_float_int64.cpp
    ../builders/tetmesh_double.cpp
    ../builders/tetmesh_float_double.cpp
  )
  target_link_libraries(trueform_tetmesh_int128_class_tests PRIVATE
    trueform_test_int128_class)
  target_compile_definitions(trueform_tetmesh_int128_class_tests PRIVATE
    TF_TETMESH_DRAGON="${PROJECT_SOURCE_DIR}/benchmarks/data/dragon-50k.stl")
  catch_discover_tests(trueform_tetmesh_int128_class_tests
    TEST_PREFIX "tetmesh_int128_class::" PROPERTIES RUN_SERIAL TRUE)
endif()
