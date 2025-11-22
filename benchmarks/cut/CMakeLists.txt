# Cut module benchmarks

# TrueForm boolean
add_executable(cut-boolean-tf boolean-tf.cpp)
target_link_libraries(cut-boolean-tf benchmark_common)

# CGAL boolean
add_executable(cut-boolean-cgal boolean-cgal.cpp)
target_link_libraries(cut-boolean-cgal benchmark_cgal_common)

# libigl boolean (requires copyleft cgal which uses GMP/MPFR)
add_executable(cut-boolean-igl boolean-igl.cpp)
target_link_libraries(cut-boolean-igl benchmark_igl_common igl_copyleft::cgal)

# TrueForm embedded self-intersection curves
add_executable(cut-embedded_self_intersection_curves-tf embedded_self_intersection_curves-tf.cpp)
target_link_libraries(cut-embedded_self_intersection_curves-tf benchmark_common)

# libigl remesh_self_intersections (requires copyleft cgal)
add_executable(cut-embedded_self_intersection_curves-igl embedded_self_intersection_curves-igl.cpp)
target_link_libraries(cut-embedded_self_intersection_curves-igl benchmark_igl_common igl_copyleft::cgal)

# TrueForm embedded isocurves
add_executable(cut-embedded_isocurves-tf embedded_isocurves-tf.cpp)
target_link_libraries(cut-embedded_isocurves-tf benchmark_common)

# VTK embedded isocurves (optional)
if(HAVE_VTK)
    add_executable(cut-embedded_isocurves-vtk embedded_isocurves-vtk.cpp)
    target_link_libraries(cut-embedded_isocurves-vtk benchmark_vtk_common)
endif()
