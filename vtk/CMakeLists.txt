cmake_minimum_required(VERSION 3.16)
project(trueform_vtk LANGUAGES CXX)

find_package(Freetype QUIET)
find_package(VTK CONFIG REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
)

set(TRUEFORM_VTK_SOURCES
  src/filters/adapter.cpp
  src/filters/boolean.cpp
  src/filters/boundary_paths.cpp
  src/filters/connected_components.cpp
  src/filters/curvatures_generator.cpp
  src/filters/intersection_curves.cpp
  src/filters/isocontours.cpp
  src/filters/isobands.cpp
  src/filters/line_cleaner.cpp
  src/filters/non_simple_edges.cpp
  src/filters/normals_generator.cpp
  src/filters/obj_reader.cpp
  src/filters/polygon_cleaner.cpp
  src/filters/self_intersection_resolver.cpp
  src/filters/stl_reader.cpp
  src/filters/triangulator.cpp
  src/core/make_byte_blocks.cpp
  src/core/make_range.cpp
  src/core/make_blocked_range.cpp
  src/core/make_vtk_array.cpp
  src/core/make_vtk_array_reindexed.cpp
  src/core/make_vtk_data_set_attributes_reindexed.cpp
  src/core/make_curves.cpp
  src/core/make_lines.cpp
  src/core/make_normals.cpp
  src/core/make_paths.cpp
  src/core/make_points.cpp
  src/core/make_polygons.cpp
  src/core/make_polys.cpp
  src/core/make_vtk_cells.cpp
  src/core/make_frame.cpp
  src/core/make_vtk_matrix.cpp
  src/core/make_vtk_normals.cpp
  src/core/make_vtk_points.cpp
  src/core/make_vtk_transform.cpp
  src/core/make_vtk_polydata.cpp
  src/core/make_world_ray.cpp
  src/core/polydata.cpp
  src/functions/aabb_from.cpp
  src/functions/area.cpp
  src/functions/chamfer_error.cpp
  src/functions/cleaned_lines.cpp
  src/functions/cleaned_polygons.cpp
  src/functions/cleaned_points.cpp
  src/functions/compute_cell_normals.cpp
  src/functions/compute_point_normals.cpp
  src/functions/compute_principal_curvatures.cpp
  src/functions/ensure_positive_orientation.cpp
  src/functions/fit_knn_alignment.cpp
  src/functions/fit_obb_alignment.cpp
  src/functions/fit_rigid_alignment.cpp
  src/functions/make_boolean_pp.cpp
  src/functions/make_boolean_mp.cpp
  src/functions/make_boolean_pm.cpp
  src/functions/make_boolean_mm.cpp
  src/functions/make_boundary_edges.cpp
  src/functions/make_boundary_paths.cpp
  src/functions/make_connected_components.cpp
  src/functions/make_intersection_curves.cpp
  src/functions/make_isobands.cpp
  src/functions/make_isocontours.cpp
  src/functions/make_non_manifold_edges.cpp
  src/functions/make_non_simple_edges.cpp
  src/functions/neighbor_search.cpp
  src/functions/neighbor_search_batch.cpp
  src/functions/neighbor_search_k.cpp
  src/functions/neighbor_search_k_batch.cpp
  src/functions/intersects.cpp
  src/functions/obb_from.cpp
  src/functions/orient_faces_consistently.cpp
  src/functions/pick.cpp
  src/functions/ray_cast.cpp
  src/functions/ray_hit.cpp
  src/functions/read_obj.cpp
  src/functions/read_stl.cpp
  src/functions/resolved_self_intersections.cpp
  src/functions/signed_volume.cpp
  src/functions/split_into_components.cpp
  src/functions/triangulated.cpp
  src/functions/write_obj.cpp
  src/functions/write_stl.cpp
)

add_library(trueform_vtk ${TRUEFORM_VTK_SOURCES})
add_library(tf::vtk ALIAS trueform_vtk)
set_target_properties(trueform_vtk PROPERTIES EXPORT_NAME vtk)

target_compile_features(trueform_vtk PUBLIC cxx_std_17)

target_include_directories(trueform_vtk PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(trueform_vtk PUBLIC
  tf::trueform
  ${VTK_LIBRARIES}
)

vtk_module_autoinit(
  TARGETS trueform_vtk
  MODULES ${VTK_LIBRARIES}
)

target_compile_options(trueform_vtk PRIVATE -Wall -Wextra -Wpedantic -Wno-extra-semi)

if(TF_BUILD_VTK_EXAMPLES)
  add_subdirectory(examples)
endif()

# ==============================================================================
# Installation
# ==============================================================================
include(GNUInstallDirs)

# Install the library
install(TARGETS trueform_vtk
  EXPORT trueformVtkTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install headers
install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install CMake config
install(EXPORT trueformVtkTargets
  FILE trueformVtkTargets.cmake
  NAMESPACE tf::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trueform_vtk
)

# Generate and install package config file
include(CMakePackageConfigHelpers)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/trueform_vtkConfig.cmake
"include(CMakeFindDependencyMacro)
find_dependency(trueform REQUIRED)
find_dependency(VTK REQUIRED COMPONENTS CommonCore CommonDataModel)
include(\"\${CMAKE_CURRENT_LIST_DIR}/trueformVtkTargets.cmake\")
")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/trueform_vtkConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trueform_vtk
)
