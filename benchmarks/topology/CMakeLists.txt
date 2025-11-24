# Topology module benchmarks

# CGAL connected components
add_library(topology-connected_components-cgal-impl OBJECT connected_components-cgal-impl.cpp)
target_link_libraries(topology-connected_components-cgal-impl benchmark_cgal_common)

add_executable(topology-connected_components-cgal connected_components-cgal.cpp)
target_link_libraries(topology-connected_components-cgal topology-connected_components-cgal-impl)

# TrueForm connected components
add_library(topology-connected_components-tf-impl OBJECT connected_components-tf-impl.cpp)
target_link_libraries(topology-connected_components-tf-impl benchmark_common)

add_executable(topology-connected_components-tf connected_components-tf.cpp)
target_link_libraries(topology-connected_components-tf topology-connected_components-tf-impl)

# libigl connected components
add_library(topology-connected_components-igl-impl OBJECT connected_components-igl-impl.cpp)
target_link_libraries(topology-connected_components-igl-impl benchmark_igl_common)

add_executable(topology-connected_components-igl connected_components-igl.cpp)
target_link_libraries(topology-connected_components-igl topology-connected_components-igl-impl)

# CGAL boundary paths
add_library(topology-boundary_paths-cgal-impl OBJECT boundary_paths-cgal-impl.cpp)
target_link_libraries(topology-boundary_paths-cgal-impl benchmark_cgal_common)

add_executable(topology-boundary_paths-cgal boundary_paths-cgal.cpp)
target_link_libraries(topology-boundary_paths-cgal topology-boundary_paths-cgal-impl)

# TrueForm boundary paths
add_library(topology-boundary_paths-tf-impl OBJECT boundary_paths-tf-impl.cpp)
target_link_libraries(topology-boundary_paths-tf-impl benchmark_common)

add_executable(topology-boundary_paths-tf boundary_paths-tf.cpp)
target_link_libraries(topology-boundary_paths-tf topology-boundary_paths-tf-impl)

# libigl boundary paths
add_library(topology-boundary_paths-igl-impl OBJECT boundary_paths-igl-impl.cpp)
target_link_libraries(topology-boundary_paths-igl-impl benchmark_igl_common)

add_executable(topology-boundary_paths-igl boundary_paths-igl.cpp)
target_link_libraries(topology-boundary_paths-igl topology-boundary_paths-igl-impl)

# VTK benchmarks (optional)
if(HAVE_VTK)
    # VTK connected components
    add_library(topology-connected_components-vtk-impl OBJECT connected_components-vtk-impl.cpp)
    target_link_libraries(topology-connected_components-vtk-impl benchmark_vtk_common)

    add_executable(topology-connected_components-vtk connected_components-vtk.cpp)
    target_link_libraries(topology-connected_components-vtk topology-connected_components-vtk-impl)

    # VTK boundary paths
    add_library(topology-boundary_paths-vtk-impl OBJECT boundary_paths-vtk-impl.cpp)
    target_link_libraries(topology-boundary_paths-vtk-impl benchmark_vtk_common)

    add_executable(topology-boundary_paths-vtk boundary_paths-vtk.cpp)
    target_link_libraries(topology-boundary_paths-vtk topology-boundary_paths-vtk-impl)
endif()
