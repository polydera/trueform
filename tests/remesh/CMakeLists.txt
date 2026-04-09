add_executable(trueform_remesh_tests
  test_remesh.cpp
)

target_link_libraries(trueform_remesh_tests PRIVATE trueform_test_common)

target_compile_options(trueform_remesh_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_remesh_tests TEST_PREFIX "remesh::")
