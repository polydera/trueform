add_executable(trueform_csg_tests
  test_expression.cpp
  test_expression_fuzz.cpp
  test_csg_graph.cpp
  test_csg_solids.cpp
  test_csg_sheets.cpp
  test_csg_domains.cpp
  test_make_outer_shell.cpp
  test_triangulated_csg.cpp
)

target_link_libraries(trueform_csg_tests PRIVATE trueform_test_common)

target_compile_options(trueform_csg_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_csg_tests TEST_PREFIX "csg::")
