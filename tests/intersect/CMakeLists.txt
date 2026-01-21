add_executable(trueform_intersect_tests
  test_intersection_curves.cpp
  test_self_intersection.cpp
  test_isocontours.cpp
)

target_link_libraries(trueform_intersect_tests PRIVATE trueform_test_common)

target_compile_options(trueform_intersect_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_intersect_tests TEST_PREFIX "intersect::")
