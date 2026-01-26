# Coal benchmark common library
# Provides shared Coal utilities and conversions

add_library(benchmark_coal_common INTERFACE)
target_include_directories(benchmark_coal_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark_coal_common INTERFACE benchmark_common coal)
