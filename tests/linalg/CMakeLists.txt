# The sparse Cholesky factor, its pattern and its solves, over the forms the
# Laplacian systems state.
add_executable(trueform_linalg_tests
  test_cholesky.cpp
)

target_link_libraries(trueform_linalg_tests PRIVATE trueform_test_common)

target_compile_options(trueform_linalg_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_linalg_tests TEST_PREFIX "linalg::")
