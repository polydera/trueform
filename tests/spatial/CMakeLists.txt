add_executable(trueform_spatial_tests
  test_neighbor_search.cpp
  test_ray_cast.cpp
  test_intersects.cpp
  test_gather_ids.cpp
  test_mod_tree.cpp
)

target_link_libraries(trueform_spatial_tests PRIVATE trueform_test_common)

target_compile_options(trueform_spatial_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_spatial_tests TEST_PREFIX "spatial::")
