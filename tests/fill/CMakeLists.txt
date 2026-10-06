add_executable(trueform_fill_tests
  test_hole_preflight.cpp
  test_hole_tiny_tier.cpp
  test_hole_table_tier.cpp
  test_hole_face_plans.cpp
  test_hole_refinement.cpp
  test_hole_fairing.cpp
  test_fill_holes.cpp
)

target_link_libraries(trueform_fill_tests PRIVATE trueform_test_common)

target_compile_options(trueform_fill_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_fill_tests TEST_PREFIX "fill::")
