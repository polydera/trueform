add_executable(trueform_core_tests
  test_classify.cpp
  test_closest_metric_point_pair.cpp
  test_intersects.cpp
  test_ray_cast.cpp
)

target_link_libraries(trueform_core_tests PRIVATE trueform_test_common)

target_compile_options(trueform_core_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3>
)

catch_discover_tests(trueform_core_tests TEST_PREFIX "core::")
