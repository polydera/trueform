# Remesh module benchmarks

# TrueForm decimation
add_library(remesh-decimation-tf-impl OBJECT decimation-tf-impl.cpp)
target_link_libraries(remesh-decimation-tf-impl benchmark_common)

add_executable(remesh-decimation-tf decimation-tf.cpp)
target_link_libraries(remesh-decimation-tf remesh-decimation-tf-impl)

# CGAL decimation
add_library(remesh-decimation-cgal-impl OBJECT decimation-cgal-impl.cpp)
target_link_libraries(remesh-decimation-cgal-impl benchmark_cgal_common)

add_executable(remesh-decimation-cgal decimation-cgal.cpp)
target_link_libraries(remesh-decimation-cgal remesh-decimation-cgal-impl)

# TrueForm isotropic remeshing
add_library(remesh-isotropic_remeshing-tf-impl OBJECT isotropic_remeshing-tf-impl.cpp)
target_link_libraries(remesh-isotropic_remeshing-tf-impl benchmark_common)

add_executable(remesh-isotropic_remeshing-tf isotropic_remeshing-tf.cpp)
target_link_libraries(remesh-isotropic_remeshing-tf remesh-isotropic_remeshing-tf-impl)

# CGAL isotropic remeshing
add_library(remesh-isotropic_remeshing-cgal-impl OBJECT isotropic_remeshing-cgal-impl.cpp)
target_link_libraries(remesh-isotropic_remeshing-cgal-impl benchmark_cgal_common)

add_executable(remesh-isotropic_remeshing-cgal isotropic_remeshing-cgal.cpp)
target_link_libraries(remesh-isotropic_remeshing-cgal remesh-isotropic_remeshing-cgal-impl)

# Quality comparison tool (standalone, not part of unified runner)
add_executable(remesh-quality-decimation quality-decimation.cpp)
target_link_libraries(remesh-quality-decimation benchmark_cgal_common)
