# FCL benchmark common library
# Provides shared FCL utilities and conversions

add_library(benchmark_fcl_common INTERFACE)
target_include_directories(benchmark_fcl_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark_fcl_common INTERFACE benchmark_common fcl::fcl)
