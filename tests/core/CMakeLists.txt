add_executable(trueform_core_tests
  test_compact_ids_and_make_map.cpp
  test_generate_offset_blocks.cpp
  test_classify.cpp
  test_determinant_of.cpp
  test_closest_metric_point_pair.cpp
  test_intersects.cpp
  test_least_squares_parallel.cpp
  test_memory.cpp
  test_ray_cast.cpp
  test_blocked_buffer.cpp
  test_buffer.cpp
  test_nth_element.cpp
  test_parallel_workers.cpp
  test_ranges.cpp
  test_tetrahedra.cpp
  test_transformed.cpp
  test_isolated_parallel_for_each.cpp
)

target_link_libraries(trueform_core_tests PRIVATE trueform_test_common)

target_compile_options(trueform_core_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

tf_discover_tests(trueform_core_tests "core::")
