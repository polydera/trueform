# VTK benchmark common library
# Provides shared VTK utilities and conversions

add_library(benchmark_vtk_common INTERFACE)
target_include_directories(benchmark_vtk_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark_vtk_common INTERFACE benchmark_common ${VTK_LIBRARIES})
