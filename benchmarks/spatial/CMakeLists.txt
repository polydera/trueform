# Spatial module benchmarks

# TrueForm point cloud tree building
add_library(spatial-point_cloud-build_tree-tf-impl OBJECT point_cloud-build_tree-tf-impl.cpp)
target_link_libraries(spatial-point_cloud-build_tree-tf-impl benchmark_common)

add_executable(spatial-point_cloud-build_tree-tf point_cloud-build_tree-tf.cpp)
target_link_libraries(spatial-point_cloud-build_tree-tf spatial-point_cloud-build_tree-tf-impl)

# nanoflann point cloud tree building
add_library(spatial-point_cloud-build_tree-nanoflann-impl OBJECT point_cloud-build_tree-nanoflann-impl.cpp)
target_link_libraries(spatial-point_cloud-build_tree-nanoflann-impl benchmark_common nanoflann::nanoflann)

add_executable(spatial-point_cloud-build_tree-nanoflann point_cloud-build_tree-nanoflann.cpp)
target_link_libraries(spatial-point_cloud-build_tree-nanoflann spatial-point_cloud-build_tree-nanoflann-impl)

# TrueForm point cloud kNN queries
add_library(spatial-point_cloud-knn-tf-impl OBJECT point_cloud-knn-tf-impl.cpp)
target_link_libraries(spatial-point_cloud-knn-tf-impl benchmark_common)

add_executable(spatial-point_cloud-knn-tf point_cloud-knn-tf.cpp)
target_link_libraries(spatial-point_cloud-knn-tf spatial-point_cloud-knn-tf-impl)

# nanoflann point cloud kNN queries
add_library(spatial-point_cloud-knn-nanoflann-impl OBJECT point_cloud-knn-nanoflann-impl.cpp)
target_link_libraries(spatial-point_cloud-knn-nanoflann-impl benchmark_common nanoflann::nanoflann)

add_executable(spatial-point_cloud-knn-nanoflann point_cloud-knn-nanoflann.cpp)
target_link_libraries(spatial-point_cloud-knn-nanoflann spatial-point_cloud-knn-nanoflann-impl)

# TrueForm polygons tree building
add_library(spatial-polygons-build_tree-tf-impl OBJECT polygons-build_tree-tf-impl.cpp)
target_link_libraries(spatial-polygons-build_tree-tf-impl benchmark_common)

add_executable(spatial-polygons-build_tree-tf polygons-build_tree-tf.cpp)
target_link_libraries(spatial-polygons-build_tree-tf spatial-polygons-build_tree-tf-impl)

# CGAL polygons tree building
add_library(spatial-polygons-build_tree-cgal-impl OBJECT polygons-build_tree-cgal-impl.cpp)
target_link_libraries(spatial-polygons-build_tree-cgal-impl benchmark_cgal_common)

add_executable(spatial-polygons-build_tree-cgal polygons-build_tree-cgal.cpp)
target_link_libraries(spatial-polygons-build_tree-cgal spatial-polygons-build_tree-cgal-impl)

# FCL polygons tree building
add_library(spatial-polygons-build_tree-fcl-impl OBJECT polygons-build_tree-fcl-impl.cpp)
target_link_libraries(spatial-polygons-build_tree-fcl-impl benchmark_fcl_common)

add_executable(spatial-polygons-build_tree-fcl polygons-build_tree-fcl.cpp)
target_link_libraries(spatial-polygons-build_tree-fcl spatial-polygons-build_tree-fcl-impl)

# TrueForm polygons closest point queries
add_library(spatial-polygons-closest_point-tf-impl OBJECT polygons-closest_point-tf-impl.cpp)
target_link_libraries(spatial-polygons-closest_point-tf-impl benchmark_common)

add_executable(spatial-polygons-closest_point-tf polygons-closest_point-tf.cpp)
target_link_libraries(spatial-polygons-closest_point-tf spatial-polygons-closest_point-tf-impl)

# CGAL polygons closest point queries
add_library(spatial-polygons-closest_point-cgal-impl OBJECT polygons-closest_point-cgal-impl.cpp)
target_link_libraries(spatial-polygons-closest_point-cgal-impl benchmark_cgal_common)

add_executable(spatial-polygons-closest_point-cgal polygons-closest_point-cgal.cpp)
target_link_libraries(spatial-polygons-closest_point-cgal spatial-polygons-closest_point-cgal-impl)

# TrueForm polygons to polygons closest point queries
add_library(spatial-polygons_to_polygons-closest_point-tf-impl OBJECT polygons_to_polygons-closest_point-tf-impl.cpp)
target_link_libraries(spatial-polygons_to_polygons-closest_point-tf-impl benchmark_common)

add_executable(spatial-polygons_to_polygons-closest_point-tf polygons_to_polygons-closest_point-tf.cpp)
target_link_libraries(spatial-polygons_to_polygons-closest_point-tf spatial-polygons_to_polygons-closest_point-tf-impl)

# FCL polygons to polygons closest point queries
add_library(spatial-polygons_to_polygons-closest_point-fcl-impl OBJECT polygons_to_polygons-closest_point-fcl-impl.cpp)
target_link_libraries(spatial-polygons_to_polygons-closest_point-fcl-impl benchmark_fcl_common)

add_executable(spatial-polygons_to_polygons-closest_point-fcl polygons_to_polygons-closest_point-fcl.cpp)
target_link_libraries(spatial-polygons_to_polygons-closest_point-fcl spatial-polygons_to_polygons-closest_point-fcl-impl)

# TrueForm polygons to polygons collision queries
add_library(spatial-polygons_to_polygons-collision-tf-impl OBJECT polygons_to_polygons-collision-tf-impl.cpp)
target_link_libraries(spatial-polygons_to_polygons-collision-tf-impl benchmark_common)

add_executable(spatial-polygons_to_polygons-collision-tf polygons_to_polygons-collision-tf.cpp)
target_link_libraries(spatial-polygons_to_polygons-collision-tf spatial-polygons_to_polygons-collision-tf-impl)

# FCL polygons to polygons collision queries
add_library(spatial-polygons_to_polygons-collision-fcl-impl OBJECT polygons_to_polygons-collision-fcl-impl.cpp)
target_link_libraries(spatial-polygons_to_polygons-collision-fcl-impl benchmark_fcl_common)

add_executable(spatial-polygons_to_polygons-collision-fcl polygons_to_polygons-collision-fcl.cpp)
target_link_libraries(spatial-polygons_to_polygons-collision-fcl spatial-polygons_to_polygons-collision-fcl-impl)

# Coal polygons tree building
add_library(spatial-polygons-build_tree-coal-impl OBJECT polygons-build_tree-coal-impl.cpp)
target_link_libraries(spatial-polygons-build_tree-coal-impl benchmark_coal_common)

add_executable(spatial-polygons-build_tree-coal polygons-build_tree-coal.cpp)
target_link_libraries(spatial-polygons-build_tree-coal spatial-polygons-build_tree-coal-impl)

# Coal polygons to polygons closest point queries
add_library(spatial-polygons_to_polygons-closest_point-coal-impl OBJECT polygons_to_polygons-closest_point-coal-impl.cpp)
target_link_libraries(spatial-polygons_to_polygons-closest_point-coal-impl benchmark_coal_common)

add_executable(spatial-polygons_to_polygons-closest_point-coal polygons_to_polygons-closest_point-coal.cpp)
target_link_libraries(spatial-polygons_to_polygons-closest_point-coal spatial-polygons_to_polygons-closest_point-coal-impl)

# Coal polygons to polygons collision queries
add_library(spatial-polygons_to_polygons-collision-coal-impl OBJECT polygons_to_polygons-collision-coal-impl.cpp)
target_link_libraries(spatial-polygons_to_polygons-collision-coal-impl benchmark_coal_common)

add_executable(spatial-polygons_to_polygons-collision-coal polygons_to_polygons-collision-coal.cpp)
target_link_libraries(spatial-polygons_to_polygons-collision-coal spatial-polygons_to_polygons-collision-coal-impl)

# TrueForm mod_tree update
add_library(spatial-mod_tree-update-tf-impl OBJECT mod_tree-update-tf-impl.cpp)
target_link_libraries(spatial-mod_tree-update-tf-impl benchmark_common)

add_executable(spatial-mod_tree-update-tf mod_tree-update-tf.cpp)
target_link_libraries(spatial-mod_tree-update-tf spatial-mod_tree-update-tf-impl)
