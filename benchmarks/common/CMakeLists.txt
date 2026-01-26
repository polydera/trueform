# Benchmark common library
# Provides shared headers and utilities for all benchmarks

add_library(benchmark_common INTERFACE)
target_include_directories(benchmark_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark_common INTERFACE trueform)
