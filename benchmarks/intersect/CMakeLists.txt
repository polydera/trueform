# Intersect module benchmarks

# TrueForm mesh-mesh intersection curves
add_executable(intersect-mesh_mesh_curves-tf mesh_mesh_curves-tf.cpp)
target_link_libraries(intersect-mesh_mesh_curves-tf benchmark_common)

# CGAL mesh-mesh intersection curves
add_executable(intersect-mesh_mesh_curves-cgal mesh_mesh_curves-cgal.cpp)
target_link_libraries(intersect-mesh_mesh_curves-cgal benchmark_cgal_common)

# TrueForm isocontours
add_executable(intersect-isocontours-tf isocontours-tf.cpp)
target_link_libraries(intersect-isocontours-tf benchmark_common)

# libigl isocontours
add_executable(intersect-isocontours-igl isocontours-igl.cpp)
target_link_libraries(intersect-isocontours-igl benchmark_igl_common)

# VTK benchmarks (optional)
if(HAVE_VTK)
    # VTK mesh-mesh intersection curves
    add_executable(intersect-mesh_mesh_curves-vtk mesh_mesh_curves-vtk.cpp)
    target_link_libraries(intersect-mesh_mesh_curves-vtk benchmark_vtk_common)

    # VTK isocontours
    add_executable(intersect-isocontours-vtk isocontours-vtk.cpp)
    target_link_libraries(intersect-isocontours-vtk benchmark_vtk_common)
endif()
