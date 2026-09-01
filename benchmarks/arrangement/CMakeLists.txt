# Arrangement module benchmarks

# TrueForm boolean
add_library(arrangement-boolean-tf-impl OBJECT boolean-tf-impl.cpp)
target_link_libraries(arrangement-boolean-tf-impl benchmark_common)

add_executable(arrangement-boolean-tf boolean-tf.cpp)
target_link_libraries(arrangement-boolean-tf arrangement-boolean-tf-impl)

# CGAL boolean
add_library(arrangement-boolean-cgal-impl OBJECT boolean-cgal-impl.cpp)
target_link_libraries(arrangement-boolean-cgal-impl benchmark_cgal_common)

add_executable(arrangement-boolean-cgal boolean-cgal.cpp)
target_link_libraries(arrangement-boolean-cgal arrangement-boolean-cgal-impl)

# libigl boolean (requires copyleft cgal which uses GMP/MPFR)
add_library(arrangement-boolean-igl-impl OBJECT boolean-igl-impl.cpp)
target_link_libraries(arrangement-boolean-igl-impl benchmark_igl_common igl_copyleft::cgal)

add_executable(arrangement-boolean-igl boolean-igl.cpp)
target_link_libraries(arrangement-boolean-igl arrangement-boolean-igl-impl)

# Geogram boolean
add_library(arrangement-boolean-geogram-impl OBJECT boolean-geogram-impl.cpp)
target_link_libraries(arrangement-boolean-geogram-impl benchmark_geogram_common)

add_executable(arrangement-boolean-geogram boolean-geogram.cpp)
target_link_libraries(arrangement-boolean-geogram arrangement-boolean-geogram-impl)

# TrueForm polygon arrangements
add_library(arrangement-polygon_arrangements-tf-impl OBJECT polygon_arrangements-tf-impl.cpp)
target_link_libraries(arrangement-polygon_arrangements-tf-impl benchmark_common)

add_executable(arrangement-polygon_arrangements-tf polygon_arrangements-tf.cpp)
target_link_libraries(arrangement-polygon_arrangements-tf arrangement-polygon_arrangements-tf-impl)

# CGAL polygon arrangements (autorefine_triangle_soup)
add_library(arrangement-polygon_arrangements-cgal-impl OBJECT polygon_arrangements-cgal-impl.cpp)
target_link_libraries(arrangement-polygon_arrangements-cgal-impl benchmark_cgal_common)

add_executable(arrangement-polygon_arrangements-cgal polygon_arrangements-cgal.cpp)
target_link_libraries(arrangement-polygon_arrangements-cgal arrangement-polygon_arrangements-cgal-impl)

# libigl polygon arrangements (remesh_self_intersections)
add_library(arrangement-polygon_arrangements-igl-impl OBJECT polygon_arrangements-igl-impl.cpp)
target_link_libraries(arrangement-polygon_arrangements-igl-impl benchmark_igl_common igl_copyleft::cgal)

add_executable(arrangement-polygon_arrangements-igl polygon_arrangements-igl.cpp)
target_link_libraries(arrangement-polygon_arrangements-igl arrangement-polygon_arrangements-igl-impl)

# TrueForm embedded isocurves
add_library(arrangement-embedded_isocurves-tf-impl OBJECT embedded_isocurves-tf-impl.cpp)
target_link_libraries(arrangement-embedded_isocurves-tf-impl benchmark_common)

add_executable(arrangement-embedded_isocurves-tf embedded_isocurves-tf.cpp)
target_link_libraries(arrangement-embedded_isocurves-tf arrangement-embedded_isocurves-tf-impl)

# VTK embedded isocurves (optional)
if(HAVE_VTK)
    add_library(arrangement-embedded_isocurves-vtk-impl OBJECT embedded_isocurves-vtk-impl.cpp)
    target_link_libraries(arrangement-embedded_isocurves-vtk-impl benchmark_vtk_common)

    add_executable(arrangement-embedded_isocurves-vtk embedded_isocurves-vtk.cpp)
    target_link_libraries(arrangement-embedded_isocurves-vtk arrangement-embedded_isocurves-vtk-impl)
endif()
