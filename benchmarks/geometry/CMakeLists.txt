# Geometry module benchmarks

# TrueForm principal curvatures
add_library(geometry-principal_curvatures-tf-impl OBJECT principal_curvatures-tf-impl.cpp)
target_link_libraries(geometry-principal_curvatures-tf-impl benchmark_common)

add_executable(geometry-principal_curvatures-tf principal_curvatures-tf.cpp)
target_link_libraries(geometry-principal_curvatures-tf geometry-principal_curvatures-tf-impl)

# libigl principal curvatures
add_library(geometry-principal_curvatures-igl-impl OBJECT principal_curvatures-igl-impl.cpp)
target_link_libraries(geometry-principal_curvatures-igl-impl benchmark_igl_common)

add_executable(geometry-principal_curvatures-igl principal_curvatures-igl.cpp)
target_link_libraries(geometry-principal_curvatures-igl geometry-principal_curvatures-igl-impl)

# TrueForm point normals
add_library(geometry-point_normals-tf-impl OBJECT point_normals-tf-impl.cpp)
target_link_libraries(geometry-point_normals-tf-impl benchmark_common)

add_executable(geometry-point_normals-tf point_normals-tf.cpp)
target_link_libraries(geometry-point_normals-tf geometry-point_normals-tf-impl)

# libigl point normals
add_library(geometry-point_normals-igl-impl OBJECT point_normals-igl-impl.cpp)
target_link_libraries(geometry-point_normals-igl-impl benchmark_igl_common)

add_executable(geometry-point_normals-igl point_normals-igl.cpp)
target_link_libraries(geometry-point_normals-igl geometry-point_normals-igl-impl)

# VTK point normals
add_library(geometry-point_normals-vtk-impl OBJECT point_normals-vtk-impl.cpp)
target_link_libraries(geometry-point_normals-vtk-impl benchmark_vtk_common)

add_executable(geometry-point_normals-vtk point_normals-vtk.cpp)
target_link_libraries(geometry-point_normals-vtk geometry-point_normals-vtk-impl)

# TrueForm ICP registration
add_library(geometry-icp_registration-tf-impl OBJECT icp_registration-tf-impl.cpp)
target_link_libraries(geometry-icp_registration-tf-impl benchmark_common)

add_executable(geometry-icp_registration-tf icp_registration-tf.cpp)
target_link_libraries(geometry-icp_registration-tf geometry-icp_registration-tf-impl)

# VTK ICP registration
add_library(geometry-icp_registration-vtk-impl OBJECT icp_registration-vtk-impl.cpp)
target_link_libraries(geometry-icp_registration-vtk-impl benchmark_vtk_common)

add_executable(geometry-icp_registration-vtk icp_registration-vtk.cpp)
target_link_libraries(geometry-icp_registration-vtk geometry-icp_registration-vtk-impl)

# libigl ICP registration
add_library(geometry-icp_registration-igl-impl OBJECT icp_registration-igl-impl.cpp)
target_link_libraries(geometry-icp_registration-igl-impl benchmark_igl_common)

add_executable(geometry-icp_registration-igl icp_registration-igl.cpp)
target_link_libraries(geometry-icp_registration-igl geometry-icp_registration-igl-impl)
