# I/O module benchmarks

# TrueForm read_stl
add_library(io-read_stl-tf-impl OBJECT read_stl-tf-impl.cpp)
target_link_libraries(io-read_stl-tf-impl benchmark_common)

add_executable(io-read_stl-tf read_stl-tf.cpp)
target_link_libraries(io-read_stl-tf io-read_stl-tf-impl)

# libigl read_stl
add_library(io-read_stl-igl-impl OBJECT read_stl-igl-impl.cpp)
target_link_libraries(io-read_stl-igl-impl benchmark_igl_common)

add_executable(io-read_stl-igl read_stl-igl.cpp)
target_link_libraries(io-read_stl-igl io-read_stl-igl-impl)

# VTK read_stl
if(HAVE_VTK)
  add_library(io-read_stl-vtk-impl OBJECT read_stl-vtk-impl.cpp)
  target_link_libraries(io-read_stl-vtk-impl benchmark_vtk_common)

  add_executable(io-read_stl-vtk read_stl-vtk.cpp)
  target_link_libraries(io-read_stl-vtk io-read_stl-vtk-impl)
endif()
