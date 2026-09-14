include("${PROJECT_SOURCE_DIR}/cmake/TrueformCppMatrix.cmake")

set(TF_CPP_REALS "${TF_CPP_MATRIX_ALL_REALS}" CACHE STRING
  "Coordinate types the C++ facade archive carries")
set(TF_CPP_INDICES "${TF_CPP_MATRIX_ALL_INDICES}" CACHE STRING
  "Index widths the C++ facade archive carries")
set(TF_CPP_DIMS "${TF_CPP_MATRIX_ALL_DIMS}" CACHE STRING
  "Dimensions the C++ facade archive carries")
set(TF_CPP_LAYOUTS "${TF_CPP_MATRIX_ALL_LAYOUTS}" CACHE STRING
  "Face layouts the C++ facade archive carries")

tf_cpp_require_matrix_values(TF_CPP_REALS TF_CPP_MATRIX_ALL_REALS ${TF_CPP_REALS})
tf_cpp_require_matrix_values(TF_CPP_INDICES TF_CPP_MATRIX_ALL_INDICES ${TF_CPP_INDICES})
tf_cpp_require_matrix_values(TF_CPP_DIMS TF_CPP_MATRIX_ALL_DIMS ${TF_CPP_DIMS})
tf_cpp_require_matrix_values(TF_CPP_LAYOUTS TF_CPP_MATRIX_ALL_LAYOUTS ${TF_CPP_LAYOUTS})
tf_cpp_require_triangle_layout()

set(TF_CPP_MATRIX_HEADER
  "${CMAKE_CURRENT_BINARY_DIR}/generated/trueform/cpp/core/matrix.hpp")
tf_cpp_generate_matrix_header(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/trueform/cpp/core/matrix.hpp.in"
  "${TF_CPP_MATRIX_HEADER}")
message(STATUS
  "C++ facade matrix: reals=${TF_CPP_REALS} indices=${TF_CPP_INDICES} "
  "dims=${TF_CPP_DIMS} layouts=${TF_CPP_LAYOUTS}")

add_library(trueform_cpp STATIC)
tf_cpp_add_shards(
  src/core/async.cpp
  src/core/elementwise.cpp
  src/core/histogram.cpp
  src/core/index_map.cpp
  src/core/cache.cpp
  src/core/edge_mesh_cache.cpp
  src/core/point_cloud_cache.cpp
  src/core/nd_array.cpp
  src/core/nd_array_indexing.cpp
  src/core/nd_array_creation.cpp
  src/core/nd_array_structure.cpp
  src/core/nd_array_sorting.cpp
  src/core/obb.cpp
  src/core/offset_blocked_buffer.cpp
  src/core/reductions.cpp
  src/core/to_matrix.cpp
  src/geometry/curvature_float_int32_3d.cpp
  src/geometry/curvature_float_int64_3d.cpp
  src/geometry/curvature_double_int32_3d.cpp
  src/geometry/curvature_double_int64_3d.cpp
  src/geometry/edges_float_int32_3d.cpp
  src/geometry/edges_float_int64_3d.cpp
  src/geometry/edges_double_int32_3d.cpp
  src/geometry/edges_double_int64_3d.cpp
  src/geometry/mesh_measurements_float_int32_2d.cpp
  src/geometry/mesh_measurements_float_int32_3d.cpp
  src/geometry/mesh_measurements_float_int64_2d.cpp
  src/geometry/mesh_measurements_float_int64_3d.cpp
  src/geometry/mesh_measurements_double_int32_2d.cpp
  src/geometry/mesh_measurements_double_int32_3d.cpp
  src/geometry/mesh_measurements_double_int64_2d.cpp
  src/geometry/mesh_measurements_double_int64_3d.cpp
  src/geometry/mesh_primitives_float_int32_3d.cpp
  src/geometry/mesh_primitives_float_int64_3d.cpp
  src/geometry/mesh_primitives_double_int32_3d.cpp
  src/geometry/mesh_primitives_double_int64_3d.cpp
  src/geometry/normals_float_int32_3d.cpp
  src/geometry/normals_float_int64_3d.cpp
  src/geometry/normals_double_int32_3d.cpp
  src/geometry/normals_double_int64_3d.cpp
  src/geometry/orientation_float_int32_2d.cpp
  src/geometry/orientation_float_int32_3d.cpp
  src/geometry/orientation_float_int64_2d.cpp
  src/geometry/orientation_float_int64_3d.cpp
  src/geometry/orientation_double_int32_2d.cpp
  src/geometry/orientation_double_int32_3d.cpp
  src/geometry/orientation_double_int64_2d.cpp
  src/geometry/orientation_double_int64_3d.cpp
  src/geometry/primitive_geometry_float_2d.cpp
  src/geometry/primitive_geometry_float_3d.cpp
  src/geometry/primitive_geometry_double_2d.cpp
  src/geometry/primitive_geometry_double_3d.cpp
  src/geometry/registration_fit_float_2d.cpp
  src/geometry/registration_fit_float_3d.cpp
  src/geometry/registration_fit_double_2d.cpp
  src/geometry/registration_fit_double_3d.cpp
  src/geometry/smoothing_float_int32_2d.cpp
  src/geometry/smoothing_float_int32_3d.cpp
  src/geometry/smoothing_float_int64_2d.cpp
  src/geometry/smoothing_float_int64_3d.cpp
  src/geometry/smoothing_double_int32_2d.cpp
  src/geometry/smoothing_double_int32_3d.cpp
  src/geometry/smoothing_double_int64_2d.cpp
  src/geometry/smoothing_double_int64_3d.cpp
  src/geometry/triangulate_float_int32_2d.cpp
  src/geometry/triangulate_float_int32_3d.cpp
  src/geometry/triangulate_float_int64_2d.cpp
  src/geometry/triangulate_float_int64_3d.cpp
  src/geometry/triangulate_double_int32_2d.cpp
  src/geometry/triangulate_double_int32_3d.cpp
  src/geometry/triangulate_double_int64_2d.cpp
  src/geometry/triangulate_double_int64_3d.cpp
  src/io/obj_float_int32_3d.cpp
  src/io/obj_float_int64_3d.cpp
  src/io/obj_double_int32_3d.cpp
  src/io/obj_double_int64_3d.cpp
  src/io/stl_int32_3d.cpp
  src/io/stl_int64_3d.cpp
  src/intersect/intersection_curves_float_int32_int32_3d.cpp
  src/intersect/intersection_curves_float_int32_int64_3d.cpp
  src/intersect/intersection_curves_float_int64_int32_3d.cpp
  src/intersect/intersection_curves_float_int64_int64_3d.cpp
  src/intersect/intersection_curves_double_int32_int32_3d.cpp
  src/intersect/intersection_curves_double_int32_int64_3d.cpp
  src/intersect/intersection_curves_double_int64_int32_3d.cpp
  src/intersect/intersection_curves_double_int64_int64_3d.cpp
  src/spatial/closest_metric_point_float_2d.cpp
  src/spatial/closest_metric_point_float_3d.cpp
  src/spatial/closest_metric_point_double_2d.cpp
  src/spatial/closest_metric_point_double_3d.cpp
  src/spatial/closest_metric_point_mixed_2d.cpp
  src/spatial/closest_metric_point_mixed_3d.cpp
  src/spatial/closest_metric_point_pair_float_2d.cpp
  src/spatial/closest_metric_point_pair_float_3d.cpp
  src/spatial/closest_metric_point_pair_double_2d.cpp
  src/spatial/closest_metric_point_pair_double_3d.cpp
  src/spatial/closest_metric_point_pair_mixed_2d.cpp
  src/spatial/closest_metric_point_pair_mixed_3d.cpp
  src/spatial/distance_float_int32_2d.cpp
  src/spatial/distance_float_int32_3d.cpp
  src/spatial/distance_float_int64_2d.cpp
  src/spatial/distance_float_int64_3d.cpp
  src/spatial/distance_double_int32_2d.cpp
  src/spatial/distance_double_int32_3d.cpp
  src/spatial/distance_double_int64_2d.cpp
  src/spatial/distance_double_int64_3d.cpp
  src/spatial/distance_mixed_int32_2d.cpp
  src/spatial/distance_mixed_int32_3d.cpp
  src/spatial/distance_mixed_int64_2d.cpp
  src/spatial/distance_mixed_int64_3d.cpp
  src/spatial/gather_ids_float_2d.cpp
  src/spatial/gather_ids_float_3d.cpp
  src/spatial/gather_ids_double_2d.cpp
  src/spatial/gather_ids_double_3d.cpp
  src/spatial/intersection_result.cpp
  src/spatial/intersects_float_2d.cpp
  src/spatial/intersects_double_2d.cpp
  src/spatial/intersects_mixed_2d.cpp
  src/spatial/intersects_float_3d.cpp
  src/spatial/intersects_double_3d.cpp
  src/spatial/intersects_mixed_3d.cpp
  src/spatial/neighbor_search_float_2d.cpp
  src/spatial/neighbor_search_float_3d.cpp
  src/spatial/neighbor_search_double_2d.cpp
  src/spatial/neighbor_search_double_3d.cpp
  src/spatial/primitive_float_2d.cpp
  src/spatial/primitive_float_3d.cpp
  src/spatial/primitive_double_2d.cpp
  src/spatial/primitive_double_3d.cpp
  src/spatial/ray_cast_float_2d.cpp
  src/spatial/ray_cast_float_3d.cpp
  src/spatial/ray_cast_double_2d.cpp
  src/spatial/ray_cast_double_3d.cpp
  src/spatial/ray_cast_mixed_2d.cpp
  src/spatial/ray_cast_mixed_3d.cpp
  src/spatial/ray_cast_edge_mesh_double_int32_2d.cpp
  src/spatial/ray_cast_edge_mesh_double_int32_3d.cpp
  src/spatial/ray_cast_edge_mesh_double_int64_2d.cpp
  src/spatial/ray_cast_edge_mesh_double_int64_3d.cpp
  src/spatial/ray_cast_edge_mesh_float_int32_2d.cpp
  src/spatial/ray_cast_edge_mesh_float_int32_3d.cpp
  src/spatial/ray_cast_edge_mesh_float_int64_2d.cpp
  src/spatial/ray_cast_edge_mesh_float_int64_3d.cpp
  src/spatial/ray_cast_mesh_double_int32_2d.cpp
  src/spatial/ray_cast_mesh_double_int32_3d.cpp
  src/spatial/ray_cast_mesh_double_int64_2d.cpp
  src/spatial/ray_cast_mesh_double_int64_3d.cpp
  src/spatial/ray_cast_mesh_float_int32_2d.cpp
  src/spatial/ray_cast_mesh_float_int32_3d.cpp
  src/spatial/ray_cast_mesh_float_int64_2d.cpp
  src/spatial/ray_cast_mesh_float_int64_3d.cpp
  src/spatial/ray_cast_point_cloud_double_2d.cpp
  src/spatial/ray_cast_point_cloud_double_3d.cpp
  src/spatial/ray_cast_point_cloud_float_2d.cpp
  src/spatial/ray_cast_point_cloud_float_3d.cpp
  src/spatial/signed_distance_float_int32_3d.cpp
  src/spatial/signed_distance_float_int64_3d.cpp
  src/spatial/signed_distance_double_int32_3d.cpp
  src/spatial/signed_distance_double_int64_3d.cpp
  src/spatial/transformed_float_2d.cpp
  src/spatial/transformed_float_3d.cpp
  src/spatial/transformed_double_2d.cpp
  src/spatial/transformed_double_3d.cpp
)
add_library(tf::trueform_cpp ALIAS trueform_cpp)

# Emscripten 6.0.3 crashes in WebAssembly instruction selection while
# optimizing the native-only 2D OBB registration specializations at -O3.
# Compile only those independently extractable archive shards at -O1; the
# TypeScript bridge remains on the legacy 3D registration symbols.
if(EMSCRIPTEN)
  set_source_files_properties(
    src/geometry/registration_fit_float_2d.cpp
    src/geometry/registration_fit_double_2d.cpp
    PROPERTIES COMPILE_OPTIONS "-O1;-fno-lto"
  )
endif()

add_subdirectory(src/arrangement)
add_subdirectory(src/clean)
add_subdirectory(src/csg)
add_subdirectory(src/iso)
add_subdirectory(src/reindex)
add_subdirectory(src/remesh)
add_subdirectory(src/topology)

target_compile_features(trueform_cpp PUBLIC cxx_std_17)
set_target_properties(trueform_cpp PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(trueform_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(trueform_cpp PUBLIC tf::trueform)

target_compile_options(trueform_cpp PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Unoptimized facade template instantiations emit millions of weak symbols and
# can exceed Darwin's practical 4 GiB archive boundary even without debug
# information. Keep line-level debuggability while allowing the compiler to
# fold those duplicate implementation symbols in Debug and single-config
# builds with no explicit configuration. Release flags remain unchanged.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(trueform_cpp PRIVATE "$<$<CONFIG:Debug>:-Og>")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "")
    target_compile_options(trueform_cpp PRIVATE -Og)
  endif()
endif()

add_custom_command(TARGET trueform_cpp POST_BUILD
  COMMAND "${CMAKE_COMMAND}"
    "-DTRUEFORM_CPP_ARCHIVE=$<TARGET_FILE:trueform_cpp>"
    "-DTRUEFORM_CPP_ARCHIVE_MAX_BYTES=3758096384"
    -P "${PROJECT_SOURCE_DIR}/cmake/CheckCppArchiveSize.cmake"
  VERBATIM
)

if(TF_BUILD_CPP)
  install(
    TARGETS trueform_cpp
    EXPORT trueformTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "*.in" EXCLUDE
  )
  install(
    FILES "${TF_CPP_MATRIX_HEADER}"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trueform/cpp/core
  )
endif()

if(TF_BUILD_CPP AND TF_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(TF_BUILD_CPP_VTK_EXAMPLES)
  add_subdirectory(examples/vtk)
endif()

if(TF_BUILD_CPP_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
