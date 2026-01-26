# Intersect module benchmarks

# TrueForm mesh-mesh intersection curves
add_library(intersect-mesh_mesh_curves-tf-impl OBJECT mesh_mesh_curves-tf-impl.cpp)
target_link_libraries(intersect-mesh_mesh_curves-tf-impl benchmark_common)

add_executable(intersect-mesh_mesh_curves-tf mesh_mesh_curves-tf.cpp)
target_link_libraries(intersect-mesh_mesh_curves-tf intersect-mesh_mesh_curves-tf-impl)

# CGAL mesh-mesh intersection curves
add_library(intersect-mesh_mesh_curves-cgal-impl OBJECT mesh_mesh_curves-cgal-impl.cpp)
target_link_libraries(intersect-mesh_mesh_curves-cgal-impl benchmark_cgal_common)

add_executable(intersect-mesh_mesh_curves-cgal mesh_mesh_curves-cgal.cpp)
target_link_libraries(intersect-mesh_mesh_curves-cgal intersect-mesh_mesh_curves-cgal-impl)

# TrueForm isocontours
add_library(intersect-isocontours-tf-impl OBJECT isocontours-tf-impl.cpp)
target_link_libraries(intersect-isocontours-tf-impl benchmark_common)

add_executable(intersect-isocontours-tf isocontours-tf.cpp)
target_link_libraries(intersect-isocontours-tf intersect-isocontours-tf-impl)

# libigl isocontours
add_library(intersect-isocontours-igl-impl OBJECT isocontours-igl-impl.cpp)
target_link_libraries(intersect-isocontours-igl-impl benchmark_igl_common)

add_executable(intersect-isocontours-igl isocontours-igl.cpp)
target_link_libraries(intersect-isocontours-igl intersect-isocontours-igl-impl)

# VTK benchmarks (optional)
if(HAVE_VTK)
    # VTK mesh-mesh intersection curves
    add_library(intersect-mesh_mesh_curves-vtk-impl OBJECT mesh_mesh_curves-vtk-impl.cpp)
    target_link_libraries(intersect-mesh_mesh_curves-vtk-impl benchmark_vtk_common)

    add_executable(intersect-mesh_mesh_curves-vtk mesh_mesh_curves-vtk.cpp)
    target_link_libraries(intersect-mesh_mesh_curves-vtk intersect-mesh_mesh_curves-vtk-impl)

    # VTK isocontours
    add_library(intersect-isocontours-vtk-impl OBJECT isocontours-vtk-impl.cpp)
    target_link_libraries(intersect-isocontours-vtk-impl benchmark_vtk_common)

    add_executable(intersect-isocontours-vtk isocontours-vtk.cpp)
    target_link_libraries(intersect-isocontours-vtk intersect-isocontours-vtk-impl)
endif()
