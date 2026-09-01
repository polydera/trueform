add_executable(trueform_exact_tests
  test_det2_sign.cpp
  test_dot_sign.cpp
  test_edge_parameter.cpp
  test_classify_wedge.cpp
  test_dyadic_blend.cpp
  test_dyadic_ratio.cpp
  test_wide_dyadic_ratio.cpp
  test_is_between_on_segment.cpp
  test_rebase_parameter.cpp
  test_canonical_plane.cpp
  test_snap_parameter_to_edge.cpp
  test_int128_msvc.cpp
  test_int256.cpp
  test_segment_hits_aabb.cpp
  test_door_placement.cpp
)

target_link_libraries(trueform_exact_tests PRIVATE trueform_test_common)

target_compile_options(trueform_exact_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_exact_tests TEST_PREFIX "exact::")
