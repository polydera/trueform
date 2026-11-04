add_executable(trueform_tetmesh_tests
  test_make_tet_mesh.cpp
)

target_link_libraries(trueform_tetmesh_tests PRIVATE trueform_test_common)
target_compile_definitions(trueform_tetmesh_tests PRIVATE
  TF_TETMESH_DRAGON="${PROJECT_SOURCE_DIR}/benchmarks/data/dragon-50k.stl")

target_compile_options(trueform_tetmesh_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_tetmesh_tests TEST_PREFIX "tetmesh::")
