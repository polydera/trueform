# Cut module benchmarks

# TrueForm boolean
add_library(cut-boolean-tf-impl OBJECT boolean-tf-impl.cpp)
target_link_libraries(cut-boolean-tf-impl benchmark_common)

add_executable(cut-boolean-tf boolean-tf.cpp)
target_link_libraries(cut-boolean-tf cut-boolean-tf-impl)

# CGAL boolean
add_library(cut-boolean-cgal-impl OBJECT boolean-cgal-impl.cpp)
target_link_libraries(cut-boolean-cgal-impl benchmark_cgal_common)

add_executable(cut-boolean-cgal boolean-cgal.cpp)
target_link_libraries(cut-boolean-cgal cut-boolean-cgal-impl)

# libigl boolean (requires copyleft cgal which uses GMP/MPFR)
add_library(cut-boolean-igl-impl OBJECT boolean-igl-impl.cpp)
target_link_libraries(cut-boolean-igl-impl benchmark_igl_common igl_copyleft::cgal)

add_executable(cut-boolean-igl boolean-igl.cpp)
target_link_libraries(cut-boolean-igl cut-boolean-igl-impl)

# TrueForm embedded self-intersection curves
add_library(cut-embedded_self_intersection_curves-tf-impl OBJECT embedded_self_intersection_curves-tf-impl.cpp)
target_link_libraries(cut-embedded_self_intersection_curves-tf-impl benchmark_common)

add_executable(cut-embedded_self_intersection_curves-tf embedded_self_intersection_curves-tf.cpp)
target_link_libraries(cut-embedded_self_intersection_curves-tf cut-embedded_self_intersection_curves-tf-impl)

# libigl remesh_self_intersections (requires copyleft cgal)
add_library(cut-embedded_self_intersection_curves-igl-impl OBJECT embedded_self_intersection_curves-igl-impl.cpp)
target_link_libraries(cut-embedded_self_intersection_curves-igl-impl benchmark_igl_common igl_copyleft::cgal)

add_executable(cut-embedded_self_intersection_curves-igl embedded_self_intersection_curves-igl.cpp)
target_link_libraries(cut-embedded_self_intersection_curves-igl cut-embedded_self_intersection_curves-igl-impl)

# TrueForm embedded isocurves
add_library(cut-embedded_isocurves-tf-impl OBJECT embedded_isocurves-tf-impl.cpp)
target_link_libraries(cut-embedded_isocurves-tf-impl benchmark_common)

add_executable(cut-embedded_isocurves-tf embedded_isocurves-tf.cpp)
target_link_libraries(cut-embedded_isocurves-tf cut-embedded_isocurves-tf-impl)

# VTK embedded isocurves (optional)
if(HAVE_VTK)
    add_library(cut-embedded_isocurves-vtk-impl OBJECT embedded_isocurves-vtk-impl.cpp)
    target_link_libraries(cut-embedded_isocurves-vtk-impl benchmark_vtk_common)

    add_executable(cut-embedded_isocurves-vtk embedded_isocurves-vtk.cpp)
    target_link_libraries(cut-embedded_isocurves-vtk cut-embedded_isocurves-vtk-impl)
endif()
