# The suite's compiled build tier. One translation unit per (index, real)
# combination, each independent, so the builds ride all cores instead of
# sitting on one test target's critical path. A test target links this and
# compiles calls; it instantiates no build kernel.
#
# NOT a unity target: the point is that these units are independent.
add_library(trueform_test_builders STATIC
  arrangement_builders_int32_double.cpp
  arrangement_builders_int32_float.cpp
  arrangement_builders_int64_double.cpp
  arrangement_pairs_int32_double.cpp
  arrangement_pairs_int32_float.cpp
  arrangement_pairs_int64_double.cpp
  arrangement_reads_int32_double.cpp
  arrangement_reads_int32_float.cpp
  arrangement_reads_int64_double.cpp
  boolean_builders_int32_float.cpp
  boolean_builders_int64_double.cpp
  csg_builders_int32_double.cpp
  csg_builders_int32_float.cpp
  csg_builders_int64_double.cpp
)

target_include_directories(trueform_test_builders PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(trueform_test_builders PUBLIC tf::trueform)

target_compile_options(trueform_test_builders PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -O1>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /bigobj /wd4244 /wd4267 /wd4458 /wd4701>
)
