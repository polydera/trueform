# Topology module benchmarks

# CGAL connected components
add_executable(topology-connected_components-cgal connected_components-cgal.cpp)
target_link_libraries(topology-connected_components-cgal benchmark_cgal_common)

# TrueForm connected components
add_executable(topology-connected_components-tf connected_components-tf.cpp)
target_link_libraries(topology-connected_components-tf benchmark_common)

# CGAL boundary paths
add_executable(topology-boundary_paths-cgal boundary_paths-cgal.cpp)
target_link_libraries(topology-boundary_paths-cgal benchmark_cgal_common)

# TrueForm boundary paths
add_executable(topology-boundary_paths-tf boundary_paths-tf.cpp)
target_link_libraries(topology-boundary_paths-tf benchmark_common)

# libigl connected components
add_executable(topology-connected_components-igl connected_components-igl.cpp)
target_link_libraries(topology-connected_components-igl benchmark_igl_common)

# libigl boundary paths
add_executable(topology-boundary_paths-igl boundary_paths-igl.cpp)
target_link_libraries(topology-boundary_paths-igl benchmark_igl_common)

# VTK benchmarks (optional)
if(HAVE_VTK)
    # VTK connected components
    add_executable(topology-connected_components-vtk connected_components-vtk.cpp)
    target_link_libraries(topology-connected_components-vtk benchmark_vtk_common)

    # VTK boundary paths
    add_executable(topology-boundary_paths-vtk boundary_paths-vtk.cpp)
    target_link_libraries(topology-boundary_paths-vtk benchmark_vtk_common)
endif()
