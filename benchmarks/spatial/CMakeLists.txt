# Spatial module benchmarks

# TrueForm point cloud tree building
add_executable(spatial-point_cloud-build_tree-tf point_cloud-build_tree-tf.cpp)
target_link_libraries(spatial-point_cloud-build_tree-tf benchmark_common)

# nanoflann point cloud tree building
add_executable(spatial-point_cloud-build_tree-nanoflann point_cloud-build_tree-nanoflann.cpp)
target_link_libraries(spatial-point_cloud-build_tree-nanoflann benchmark_common nanoflann::nanoflann)

# TrueForm point cloud kNN queries
add_executable(spatial-point_cloud-knn-tf point_cloud-knn-tf.cpp)
target_link_libraries(spatial-point_cloud-knn-tf benchmark_common)

# nanoflann point cloud kNN queries
add_executable(spatial-point_cloud-knn-nanoflann point_cloud-knn-nanoflann.cpp)
target_link_libraries(spatial-point_cloud-knn-nanoflann benchmark_common nanoflann::nanoflann)

# TrueForm polygons tree building
add_executable(spatial-polygons-build_tree-tf polygons-build_tree-tf.cpp)
target_link_libraries(spatial-polygons-build_tree-tf benchmark_common)

# CGAL polygons tree building
add_executable(spatial-polygons-build_tree-cgal polygons-build_tree-cgal.cpp)
target_link_libraries(spatial-polygons-build_tree-cgal benchmark_cgal_common)

# FCL polygons tree building
add_executable(spatial-polygons-build_tree-fcl polygons-build_tree-fcl.cpp)
target_link_libraries(spatial-polygons-build_tree-fcl benchmark_fcl_common)

# TrueForm polygons closest point queries
add_executable(spatial-polygons-closest_point-tf polygons-closest_point-tf.cpp)
target_link_libraries(spatial-polygons-closest_point-tf benchmark_common)

# CGAL polygons closest point queries
add_executable(spatial-polygons-closest_point-cgal polygons-closest_point-cgal.cpp)
target_link_libraries(spatial-polygons-closest_point-cgal benchmark_cgal_common)

# TrueForm polygons to point cloud closest point queries
add_executable(spatial-polygons_to_point_cloud-closest_point-tf polygons_to_point_cloud-closest_point-tf.cpp)
target_link_libraries(spatial-polygons_to_point_cloud-closest_point-tf benchmark_common)

# CGAL polygons to point cloud closest point queries
add_executable(spatial-polygons_to_point_cloud-closest_point-cgal polygons_to_point_cloud-closest_point-cgal.cpp)
target_link_libraries(spatial-polygons_to_point_cloud-closest_point-cgal benchmark_cgal_common)

# TrueForm polygons to polygons closest point queries
add_executable(spatial-polygons_to_polygons-closest_point-tf polygons_to_polygons-closest_point-tf.cpp)
target_link_libraries(spatial-polygons_to_polygons-closest_point-tf benchmark_common)

# FCL polygons to polygons closest point queries
add_executable(spatial-polygons_to_polygons-closest_point-fcl polygons_to_polygons-closest_point-fcl.cpp)
target_link_libraries(spatial-polygons_to_polygons-closest_point-fcl benchmark_fcl_common)
