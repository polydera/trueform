add_executable(trueform_clean_tests
  test_clean_points.cpp
  test_clean_segments.cpp
  test_clean_polygons.cpp
  test_clean_curves.cpp
)

target_link_libraries(trueform_clean_tests PRIVATE trueform_test_common)

target_compile_options(trueform_clean_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_clean_tests TEST_PREFIX "clean::")
