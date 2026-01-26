# CGAL benchmark common library
# Provides shared CGAL utilities and conversions

add_library(benchmark_cgal_common INTERFACE)
target_include_directories(benchmark_cgal_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark_cgal_common INTERFACE benchmark_common CGAL::CGAL CGAL::TBB_support)
