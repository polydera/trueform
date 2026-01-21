add_executable(trueform_io_tests
  test_read_stl.cpp
  test_write_stl.cpp
  # test_read_obj.cpp
  # test_write_obj.cpp
)

target_link_libraries(trueform_io_tests PRIVATE trueform_test_common)

target_compile_options(trueform_io_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_io_tests TEST_PREFIX "io::")
