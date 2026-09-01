add_executable(trueform_arrangement_tests
  test_arrangement_curves.cpp
  test_quantized_weld.cpp
  test_has_adjacent_split.cpp
  test_mesh_arrangements.cpp
  test_segment_arrangements.cpp
  test_self_arrangements.cpp
)

target_link_libraries(trueform_arrangement_tests PRIVATE trueform_test_common)

target_compile_options(trueform_arrangement_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_arrangement_tests TEST_PREFIX "arrangement::")

add_subdirectory(planes)
