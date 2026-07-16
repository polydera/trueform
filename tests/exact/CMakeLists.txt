add_executable(trueform_exact_tests
  test_dot_sign.cpp
  test_classify_wedge.cpp
  test_int128_msvc.cpp
  test_int256.cpp
  test_segment_hits_aabb.cpp
)

target_link_libraries(trueform_exact_tests PRIVATE trueform_test_common)

target_compile_options(trueform_exact_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_exact_tests TEST_PREFIX "exact::")
