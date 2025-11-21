# libigl benchmark common library
# Provides shared libigl utilities and conversions

add_library(benchmark_igl_common INTERFACE)
target_include_directories(benchmark_igl_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark_igl_common INTERFACE benchmark_common igl::core)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_igl_common INTERFACE OpenMP::OpenMP_CXX)
endif()
