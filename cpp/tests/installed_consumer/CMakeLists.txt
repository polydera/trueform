cmake_minimum_required(VERSION 3.16)
project(trueform_installed_consumer LANGUAGES CXX)

find_package(trueform CONFIG REQUIRED)

if(NOT TARGET tf::trueform_cpp)
  message(FATAL_ERROR "The installed trueform package does not export tf::trueform_cpp")
endif()

function(assert_not_under label candidate forbidden_root)
  if(candidate MATCHES "^\\$<")
    message(FATAL_ERROR
      "${label} unexpectedly contains an unresolved generator expression: ${candidate}")
  endif()

  get_filename_component(candidate_path "${candidate}" REALPATH)
  get_filename_component(forbidden_path "${forbidden_root}" REALPATH)
  string(FIND "${candidate_path}/" "${forbidden_path}/" prefix_position)
  if(prefix_position EQUAL 0)
    message(FATAL_ERROR
      "${label} exposes the producer tree: ${candidate_path} (under ${forbidden_path})")
  endif()
endfunction()

function(assert_under label candidate required_root)
  get_filename_component(candidate_path "${candidate}" REALPATH)
  get_filename_component(required_path "${required_root}" REALPATH)
  string(FIND "${candidate_path}/" "${required_path}/" prefix_position)
  if(NOT prefix_position EQUAL 0)
    message(FATAL_ERROR
      "${label} is outside the expected install prefix: ${candidate_path} (expected under ${required_path})")
  endif()
endfunction()

get_filename_component(expected_prefix "${TRUEFORM_EXPECTED_INSTALL_PREFIX}" REALPATH)

foreach(package_target IN ITEMS tf::trueform_cpp tf::trueform)
  get_target_property(package_includes ${package_target} INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT package_includes)
    message(FATAL_ERROR "${package_target} has no installed interface include directories")
  endif()

  foreach(package_include IN LISTS package_includes)
    if(NOT IS_DIRECTORY "${package_include}")
      message(FATAL_ERROR
        "${package_target} include directory does not exist: ${package_include}")
    endif()
    assert_under("${package_target} include directory" "${package_include}" "${expected_prefix}")
    foreach(forbidden_root IN LISTS TRUEFORM_FORBIDDEN_INCLUDE_ROOTS)
      assert_not_under("${package_target} include directory" "${package_include}" "${forbidden_root}")
    endforeach()
  endforeach()
endforeach()

get_filename_component(package_directory "${trueform_DIR}" REALPATH)
assert_under("trueform package directory" "${package_directory}" "${expected_prefix}")

# An installed consumer must link the installed archive, never an artifact left
# in the producer source/build trees. Check every location exported by CMake,
# including a generic IMPORTED_LOCATION when one is provided.
get_target_property(imported_configurations tf::trueform_cpp IMPORTED_CONFIGURATIONS)
if(NOT imported_configurations)
  set(imported_configurations "")
endif()

set(imported_location_properties IMPORTED_LOCATION)
foreach(imported_configuration IN LISTS imported_configurations)
  string(TOUPPER "${imported_configuration}" imported_configuration_upper)
  list(APPEND imported_location_properties "IMPORTED_LOCATION_${imported_configuration_upper}")
endforeach()
list(REMOVE_DUPLICATES imported_location_properties)

set(imported_archive_count 0)
foreach(location_property IN LISTS imported_location_properties)
  get_target_property(imported_archive tf::trueform_cpp "${location_property}")
  if(NOT imported_archive OR imported_archive MATCHES "-NOTFOUND$")
    if(NOT location_property STREQUAL "IMPORTED_LOCATION")
      message(FATAL_ERROR
        "tf::trueform_cpp declares ${location_property} but does not provide its archive location")
    endif()
    continue()
  endif()
  if(imported_archive MATCHES "^\\$<")
    message(FATAL_ERROR
      "tf::trueform_cpp ${location_property} contains an unresolved generator expression: ${imported_archive}")
  endif()
  if(NOT EXISTS "${imported_archive}" OR IS_DIRECTORY "${imported_archive}")
    message(FATAL_ERROR
      "tf::trueform_cpp ${location_property} does not name an existing archive: ${imported_archive}")
  endif()

  math(EXPR imported_archive_count "${imported_archive_count} + 1")
  assert_under("tf::trueform_cpp ${location_property}" "${imported_archive}" "${expected_prefix}")
  foreach(forbidden_root IN LISTS TRUEFORM_FORBIDDEN_PRODUCER_ROOTS)
    assert_not_under("tf::trueform_cpp ${location_property}" "${imported_archive}" "${forbidden_root}")
  endforeach()
  message(STATUS "Verified tf::trueform_cpp ${location_property}: ${imported_archive}")
endforeach()

if(imported_archive_count EQUAL 0)
  message(FATAL_ERROR "tf::trueform_cpp exports no imported archive locations")
endif()

try_compile(unsupported_outer_shell_real_compiles
  "${CMAKE_CURRENT_BINARY_DIR}/reject_outer_shell_long_double"
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/reject_outer_shell_long_double.cpp"
  CMAKE_FLAGS
    "-DCMAKE_CXX_STANDARD=17"
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
  LINK_LIBRARIES tf::trueform_cpp
  OUTPUT_VARIABLE unsupported_outer_shell_real_output)
if(unsupported_outer_shell_real_compiles)
  message(FATAL_ERROR
    "outer_shell<long double, int64_t> unexpectedly compiled")
endif()
if(NOT unsupported_outer_shell_real_output MATCHES
    "ambiguous|call to deleted function|use of deleted function|attempt to use a deleted function")
  message(FATAL_ERROR
    "outer_shell<long double, int64_t> failed for an unexpected reason:\n${unsupported_outer_shell_real_output}")
endif()
message(STATUS
  "Verified outer_shell<long double, int64_t> is rejected at compile time")

# The connectivity entries take carriers that name no index of their own, so
# the entry is what states the widths this build answers for. Whether the
# int32 overloads exist is the installed matrix's own statement, and the probe
# asks it: carried, the call compiles; left out, it fails where it is written.
find_file(trueform_installed_matrix_header trueform/cpp/core/matrix.hpp
  PATHS "${expected_prefix}/include" NO_DEFAULT_PATH)
if(NOT trueform_installed_matrix_header)
  message(FATAL_ERROR "The installed package exports no matrix header")
endif()
file(READ "${trueform_installed_matrix_header}" trueform_installed_matrix)
string(REGEX MATCH "#define TF_CPP_MATRIX_HAS_INT32 ([01])"
  trueform_matrix_int32_match "${trueform_installed_matrix}")
if(NOT trueform_matrix_int32_match)
  message(FATAL_ERROR
    "The installed matrix header does not state TF_CPP_MATRIX_HAS_INT32")
endif()
set(trueform_matrix_has_int32 "${CMAKE_MATCH_1}")

try_compile(int32_components_compile
  "${CMAKE_CURRENT_BINARY_DIR}/reject_disabled_index_components"
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/reject_disabled_index_components.cpp"
  CMAKE_FLAGS
    "-DCMAKE_CXX_STANDARD=17"
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
  LINK_LIBRARIES tf::trueform_cpp
  OUTPUT_VARIABLE int32_components_output)
if(trueform_matrix_has_int32)
  if(NOT int32_components_compile)
    message(FATAL_ERROR
      "This build carries int32, but its connectivity entries do not:\n${int32_components_output}")
  endif()
  message(STATUS
    "Verified the int32 connectivity entries exist where the matrix carries int32")
else()
  if(int32_components_compile)
    message(FATAL_ERROR
      "This build left int32 out, but its connectivity entries still compiled")
  endif()
  if(NOT int32_components_output MATCHES
      "no matching function|no viable conversion|cannot convert")
    message(FATAL_ERROR
      "The disabled int32 connectivity entries failed for an unexpected reason:\n${int32_components_output}")
  endif()
  message(STATUS
    "Verified the connectivity entries refuse int32 where the matrix left it out")
endif()

# A query's real is its own axis of the matrix, so a double ray against a float
# mesh exists only where this build carries double. The probe asks the installed
# matrix: carried, the cast compiles and links; left out, the query carrier
# refuses it where it is named.
string(REGEX MATCH "#define TF_CPP_MATRIX_HAS_DOUBLE ([01])"
  trueform_matrix_double_match "${trueform_installed_matrix}")
if(NOT trueform_matrix_double_match)
  message(FATAL_ERROR
    "The installed matrix header does not state TF_CPP_MATRIX_HAS_DOUBLE")
endif()
set(trueform_matrix_has_double "${CMAKE_MATCH_1}")

try_compile(double_ray_cast_compiles
  "${CMAKE_CURRENT_BINARY_DIR}/reject_disabled_real_ray_cast"
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/reject_disabled_real_ray_cast.cpp"
  CMAKE_FLAGS
    "-DCMAKE_CXX_STANDARD=17"
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
  LINK_LIBRARIES tf::trueform_cpp
  OUTPUT_VARIABLE double_ray_cast_output)
if(trueform_matrix_has_double)
  if(NOT double_ray_cast_compiles)
    message(FATAL_ERROR
      "This build carries double, but a double ray query does not:\n${double_ray_cast_output}")
  endif()
  message(STATUS
    "Verified the double ray query exists where the matrix carries double")
else()
  if(double_ray_cast_compiles)
    message(FATAL_ERROR
      "This build left double out, but a double ray query still compiled")
  endif()
  if(NOT double_ray_cast_output MATCHES "not in the built C\\+\\+ facade matrix")
    message(FATAL_ERROR
      "The disabled double ray query failed for an unexpected reason:\n${double_ray_cast_output}")
  endif()
  message(STATUS
    "Verified the query carrier refuses double where the matrix left it out")
endif()

add_executable(trueform_installed_consumer main.cpp)
target_compile_features(trueform_installed_consumer PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_consumer PRIVATE tf::trueform_cpp)

# The umbrellas are what a first-time caller includes, and no other translation
# unit in the repository compiles them.
add_executable(trueform_installed_umbrellas umbrellas.cpp)
target_compile_features(trueform_installed_umbrellas PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_umbrellas PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_legacy_carriers legacy_carriers.cpp)
target_compile_features(trueform_installed_legacy_carriers PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_legacy_carriers PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_split_components_double_int64_2d
  split_components_double_int64_2d.cpp)
target_compile_features(trueform_installed_split_components_double_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_split_components_double_int64_2d
  PRIVATE tf::trueform_cpp)


add_executable(trueform_installed_csg_graph_double_int64
  csg_graph_double_int64.cpp)
target_compile_features(trueform_installed_csg_graph_double_int64
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_csg_graph_double_int64
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_mesh_primitives_double_int64_3d
  mesh_primitives_double_int64_3d.cpp)
target_compile_features(trueform_installed_mesh_primitives_double_int64_3d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_mesh_primitives_double_int64_3d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_mesh_measurements_float_int64_2d
  mesh_measurements_float_int64_2d.cpp)
target_compile_features(trueform_installed_mesh_measurements_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_mesh_measurements_float_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_mesh_measurements_double_int64_3d
  mesh_measurements_double_int64_3d.cpp)
target_compile_features(trueform_installed_mesh_measurements_double_int64_3d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_mesh_measurements_double_int64_3d
  PRIVATE tf::trueform_cpp)


add_executable(trueform_installed_boundary_float_int64_2d
  boundary_float_int64_2d.cpp)
target_compile_features(trueform_installed_boundary_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_boundary_float_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_boundary_double_int64_3d
  boundary_double_int64_3d.cpp)
target_compile_features(trueform_installed_boundary_double_int64_3d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_boundary_double_int64_3d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_orientation_double_int64_3d
  orientation_double_int64_3d.cpp)
target_compile_features(trueform_installed_orientation_double_int64_3d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_orientation_double_int64_3d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_triangulate_float_int64_2d
  triangulate_float_int64_2d.cpp)
target_compile_features(trueform_installed_triangulate_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_triangulate_float_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_triangulate_double_int64_3d
  triangulate_double_int64_3d.cpp)
target_compile_features(trueform_installed_triangulate_double_int64_3d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_triangulate_double_int64_3d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_connect_edges_to_paths_int32
  connect_edges_to_paths_int32.cpp)
target_compile_features(trueform_installed_connect_edges_to_paths_int32
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_connect_edges_to_paths_int32
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_connect_edges_to_paths_int64
  connect_edges_to_paths_int64.cpp)
target_compile_features(trueform_installed_connect_edges_to_paths_int64
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_connect_edges_to_paths_int64
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_intersects_float_int64_2d
  intersects_float_int64_2d.cpp)
target_compile_features(trueform_installed_intersects_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_intersects_float_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_neighbor_search_float_int64_2d
  neighbor_search_float_int64_2d.cpp)
target_compile_features(trueform_installed_neighbor_search_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_neighbor_search_float_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_gather_ids_float_int64_2d
  gather_ids_float_int64_2d.cpp)
target_compile_features(trueform_installed_gather_ids_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_gather_ids_float_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_ray_cast_float_2d ray_cast_float_2d.cpp)
target_compile_features(trueform_installed_ray_cast_float_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_ray_cast_float_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_closest_metric_point_pair_float_2d
  closest_metric_point_pair_float_2d.cpp)
target_compile_features(
  trueform_installed_closest_metric_point_pair_float_2d PRIVATE cxx_std_17)
target_link_libraries(
  trueform_installed_closest_metric_point_pair_float_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_distance_primitive_float_3d
  distance_primitive_float_3d.cpp)
target_compile_features(trueform_installed_distance_primitive_float_3d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_distance_primitive_float_3d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_registration_float_2d
  registration_float_2d.cpp)
target_compile_features(trueform_installed_registration_float_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_registration_float_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_primitive_geometry_float_2d
  primitive_geometry_float_2d.cpp)
target_compile_features(trueform_installed_primitive_geometry_float_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_primitive_geometry_float_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_legacy_primitive_normals_abi
  legacy_primitive_normals_abi.cpp)
target_compile_features(trueform_installed_legacy_primitive_normals_abi
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_legacy_primitive_normals_abi
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_obj_float_int64_dynamic
  obj_float_int64_dynamic.cpp)
target_compile_features(trueform_installed_obj_float_int64_dynamic
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_obj_float_int64_dynamic
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_stl_int64 stl_int64.cpp)
target_compile_features(trueform_installed_stl_int64 PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_stl_int64 PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_mesh_normals_double_int64_3d
  mesh_normals_double_int64_3d.cpp)
target_compile_features(trueform_installed_mesh_normals_double_int64_3d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_mesh_normals_double_int64_3d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_topology_analysis_float_int64_2d
  topology_analysis_float_int64_2d.cpp)
target_compile_features(trueform_installed_topology_analysis_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_topology_analysis_float_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_topology_analysis_double_int64_3d
  topology_analysis_double_int64_3d.cpp)
target_compile_features(trueform_installed_topology_analysis_double_int64_3d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_topology_analysis_double_int64_3d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_topology_links_int64 topology_links_int64.cpp)
target_compile_features(trueform_installed_topology_links_int64
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_topology_links_int64
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_clean_forms_double_int64_2d
  clean_forms_double_int64_2d.cpp)
target_compile_features(trueform_installed_clean_forms_double_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_clean_forms_double_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_clean_points_float_2d
  clean_points_float_2d.cpp)
target_compile_features(trueform_installed_clean_points_float_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_clean_points_float_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_clean_soups_float_int64_2d
  clean_soups_float_int64_2d.cpp)
target_compile_features(trueform_installed_clean_soups_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_clean_soups_float_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_reindex_double_int64_2d
  reindex_double_int64_2d.cpp)
target_compile_features(trueform_installed_reindex_double_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_reindex_double_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_domain_labels_double_int64_dynamic
  domain_labels_double_int64_dynamic.cpp)
target_compile_features(trueform_installed_domain_labels_double_int64_dynamic
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_domain_labels_double_int64_dynamic
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_edge_mesh_float_int64_2d
  edge_mesh_float_int64_2d.cpp)
target_compile_features(trueform_installed_edge_mesh_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_edge_mesh_float_int64_2d
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_curvature_double_int64_3d_dynamic
  curvature_double_int64_3d_dynamic.cpp)
target_compile_features(trueform_installed_curvature_double_int64_3d_dynamic
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_curvature_double_int64_3d_dynamic
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_k_rings_int64 k_rings_int64.cpp)
target_compile_features(trueform_installed_k_rings_int64 PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_k_rings_int64
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_neighborhoods_float_int64_2d
  neighborhoods_float_int64_2d.cpp)
target_compile_features(trueform_installed_neighborhoods_float_int64_2d
  PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_neighborhoods_float_int64_2d
  PRIVATE tf::trueform_cpp)


add_executable(trueform_installed_remesh_float_int32 remesh_float_int32.cpp)
target_compile_features(trueform_installed_remesh_float_int32 PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_remesh_float_int32
  PRIVATE tf::trueform_cpp)

add_executable(trueform_installed_remesh_float_int64 remesh_float_int64.cpp)
target_compile_features(trueform_installed_remesh_float_int64 PRIVATE cxx_std_17)
target_link_libraries(trueform_installed_remesh_float_int64
  PRIVATE tf::trueform_cpp)

function(add_exact_operation_consumer target source)
  add_executable(trueform_installed_${target} ${source})
  target_compile_features(trueform_installed_${target} PRIVATE cxx_std_17)
  target_link_libraries(trueform_installed_${target} PRIVATE tf::trueform_cpp)
endfunction()

add_exact_operation_consumer(exact_intersection_double_mixed_dynamic
  exact_intersection_double_mixed_dynamic.cpp)
add_exact_operation_consumer(exact_arrangements_double_int64_dynamic
  exact_arrangements_double_int64_dynamic.cpp)
add_exact_operation_consumer(exact_boolean_double_mixed_dynamic
  exact_boolean_double_mixed_dynamic.cpp)
add_exact_operation_consumer(exact_isobands_double_int64_dynamic
  exact_isobands_double_int64_dynamic.cpp)
add_exact_operation_consumer(exact_isocontours_double_int64_dynamic
  exact_isocontours_double_int64_dynamic.cpp)
add_exact_operation_consumer(exact_outer_shell_double_int64_dynamic
  exact_outer_shell_double_int64_dynamic.cpp)

set(legacy_carrier_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_legacy_carriers.map")
if(APPLE)
  target_link_options(trueform_installed_legacy_carriers PRIVATE
    "-Wl,-map,${legacy_carrier_link_map}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(trueform_installed_legacy_carriers PRIVATE
    "-Wl,-Map,${legacy_carrier_link_map}")
endif()

set(registration_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_registration_float_2d.map")
if(APPLE)
  target_link_options(trueform_installed_registration_float_2d PRIVATE
    "-Wl,-map,${registration_link_map}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(trueform_installed_registration_float_2d PRIVATE
    "-Wl,-Map,${registration_link_map}")
endif()

set(primitive_geometry_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_primitive_geometry_float_2d.map")
if(APPLE)
  target_link_options(trueform_installed_primitive_geometry_float_2d PRIVATE
    "-Wl,-map,${primitive_geometry_link_map}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(trueform_installed_primitive_geometry_float_2d PRIVATE
    "-Wl,-Map,${primitive_geometry_link_map}")
endif()

set(mesh_primitives_double_int64_3d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_mesh_primitives_double_int64_3d.map")
set(mesh_measurements_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_mesh_measurements_float_int64_2d.map")
set(mesh_measurements_double_int64_3d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_mesh_measurements_double_int64_3d.map")
set(connect_edges_to_paths_int32_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_connect_edges_to_paths_int32.map")
set(connect_edges_to_paths_int64_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_connect_edges_to_paths_int64.map")
if(APPLE)
  target_link_options(trueform_installed_mesh_primitives_double_int64_3d PRIVATE
    "-Wl,-map,${mesh_primitives_double_int64_3d_link_map}")
  target_link_options(trueform_installed_mesh_measurements_float_int64_2d PRIVATE
    "-Wl,-map,${mesh_measurements_float_int64_2d_link_map}")
  target_link_options(trueform_installed_mesh_measurements_double_int64_3d PRIVATE
    "-Wl,-map,${mesh_measurements_double_int64_3d_link_map}")
  target_link_options(trueform_installed_connect_edges_to_paths_int32 PRIVATE
    "-Wl,-map,${connect_edges_to_paths_int32_link_map}")
  target_link_options(trueform_installed_connect_edges_to_paths_int64 PRIVATE
    "-Wl,-map,${connect_edges_to_paths_int64_link_map}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(trueform_installed_mesh_primitives_double_int64_3d PRIVATE
    "-Wl,-Map,${mesh_primitives_double_int64_3d_link_map}")
  target_link_options(trueform_installed_mesh_measurements_float_int64_2d PRIVATE
    "-Wl,-Map,${mesh_measurements_float_int64_2d_link_map}")
  target_link_options(trueform_installed_mesh_measurements_double_int64_3d PRIVATE
    "-Wl,-Map,${mesh_measurements_double_int64_3d_link_map}")
  target_link_options(trueform_installed_connect_edges_to_paths_int32 PRIVATE
    "-Wl,-Map,${connect_edges_to_paths_int32_link_map}")
  target_link_options(trueform_installed_connect_edges_to_paths_int64 PRIVATE
    "-Wl,-Map,${connect_edges_to_paths_int64_link_map}")
endif()

set(boundary_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_boundary_float_int64_2d.map")
set(boundary_double_int64_3d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_boundary_double_int64_3d.map")
set(orientation_double_int64_3d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_orientation_double_int64_3d.map")
set(triangulate_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_triangulate_float_int64_2d.map")
set(triangulate_double_int64_3d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_triangulate_double_int64_3d.map")
if(APPLE)
  target_link_options(trueform_installed_boundary_float_int64_2d PRIVATE
    "-Wl,-map,${boundary_float_int64_2d_link_map}")
  target_link_options(trueform_installed_boundary_double_int64_3d PRIVATE
    "-Wl,-map,${boundary_double_int64_3d_link_map}")
  target_link_options(trueform_installed_orientation_double_int64_3d PRIVATE
    "-Wl,-map,${orientation_double_int64_3d_link_map}")
  target_link_options(trueform_installed_triangulate_float_int64_2d PRIVATE
    "-Wl,-map,${triangulate_float_int64_2d_link_map}")
  target_link_options(trueform_installed_triangulate_double_int64_3d PRIVATE
    "-Wl,-map,${triangulate_double_int64_3d_link_map}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(trueform_installed_boundary_float_int64_2d PRIVATE
    "-Wl,-Map,${boundary_float_int64_2d_link_map}")
  target_link_options(trueform_installed_boundary_double_int64_3d PRIVATE
    "-Wl,-Map,${boundary_double_int64_3d_link_map}")
  target_link_options(trueform_installed_orientation_double_int64_3d PRIVATE
    "-Wl,-Map,${orientation_double_int64_3d_link_map}")
  target_link_options(trueform_installed_triangulate_float_int64_2d PRIVATE
    "-Wl,-Map,${triangulate_float_int64_2d_link_map}")
  target_link_options(trueform_installed_triangulate_double_int64_3d PRIVATE
    "-Wl,-Map,${triangulate_double_int64_3d_link_map}")
endif()

set(intersects_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_intersects_float_int64_2d.map")
set(neighbor_search_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_neighbor_search_float_int64_2d.map")
set(gather_ids_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_gather_ids_float_int64_2d.map")
set(ray_cast_float_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_ray_cast_float_2d.map")
set(closest_metric_point_pair_float_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_closest_metric_point_pair_float_2d.map")
set(distance_primitive_float_3d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_distance_primitive_float_3d.map")
if(APPLE)
  foreach(link_target IN ITEMS
      intersects_float_int64_2d
      neighbor_search_float_int64_2d
      gather_ids_float_int64_2d
      ray_cast_float_2d
      distance_primitive_float_3d)
    target_link_options(trueform_installed_${link_target} PRIVATE
      "-Wl,-map,${${link_target}_link_map}")
  endforeach()
  target_link_options(
    trueform_installed_closest_metric_point_pair_float_2d PRIVATE
    "-Wl,-map,${closest_metric_point_pair_float_2d_link_map}")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(link_target IN ITEMS
      intersects_float_int64_2d
      neighbor_search_float_int64_2d
      gather_ids_float_int64_2d
      ray_cast_float_2d
      distance_primitive_float_3d)
    target_link_options(trueform_installed_${link_target} PRIVATE
      "-Wl,-Map,${${link_target}_link_map}")
  endforeach()
  target_link_options(
    trueform_installed_closest_metric_point_pair_float_2d PRIVATE
    "-Wl,-Map,${closest_metric_point_pair_float_2d_link_map}")
endif()

set(legacy_primitive_normals_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_legacy_primitive_normals_abi.map")
# Match the target suffixes consumed by the link-map option loop while keeping
# the shorter names used by extraction tests below.
set(legacy_primitive_normals_abi_link_map
  "${legacy_primitive_normals_link_map}")
set(obj_float_int64_dynamic_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_obj_float_int64_dynamic.map")
set(stl_int64_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_stl_int64.map")
set(mesh_normals_double_int64_3d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_mesh_normals_double_int64_3d.map")
set(topology_analysis_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_topology_analysis_float_int64_2d.map")
set(topology_analysis_double_int64_3d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_topology_analysis_double_int64_3d.map")
set(topology_links_int64_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_topology_links_int64.map")
set(clean_forms_double_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_clean_forms_double_int64_2d.map")
set(clean_points_float_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_clean_points_float_2d.map")
set(clean_soups_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_clean_soups_float_int64_2d.map")
set(split_components_double_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_split_components_double_int64_2d.map")
set(reindex_double_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_reindex_double_int64_2d.map")
set(csg_graph_double_int64_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_csg_graph_double_int64.map")
set(domain_labels_double_int64_dynamic_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_domain_labels_double_int64_dynamic.map")

if(APPLE)
  foreach(link_target IN ITEMS
      legacy_primitive_normals_abi
      obj_float_int64_dynamic
      stl_int64
      mesh_normals_double_int64_3d
      topology_analysis_float_int64_2d
      topology_analysis_double_int64_3d
      topology_links_int64
      clean_forms_double_int64_2d
      clean_points_float_2d
      clean_soups_float_int64_2d
      split_components_double_int64_2d
      reindex_double_int64_2d
      csg_graph_double_int64
      domain_labels_double_int64_dynamic)
    target_link_options(trueform_installed_${link_target} PRIVATE
      "-Wl,-map,${${link_target}_link_map}")
  endforeach()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(link_target IN ITEMS
      legacy_primitive_normals_abi
      obj_float_int64_dynamic
      stl_int64
      mesh_normals_double_int64_3d
      topology_analysis_float_int64_2d
      topology_analysis_double_int64_3d
      topology_links_int64
      clean_forms_double_int64_2d
      clean_points_float_2d
      clean_soups_float_int64_2d
      split_components_double_int64_2d
      reindex_double_int64_2d
      csg_graph_double_int64
      domain_labels_double_int64_dynamic)
    target_link_options(trueform_installed_${link_target} PRIVATE
      "-Wl,-Map,${${link_target}_link_map}")
  endforeach()
endif()

set(edge_mesh_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_edge_mesh_float_int64_2d.map")
set(curvature_double_int64_3d_dynamic_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_curvature_double_int64_3d_dynamic.map")
set(k_rings_int64_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_k_rings_int64.map")
set(neighborhoods_float_int64_2d_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_neighborhoods_float_int64_2d.map")
set(remesh_float_int32_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_remesh_float_int32.map")
set(remesh_float_int64_link_map
  "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_remesh_float_int64.map")

if(APPLE)
  foreach(link_target IN ITEMS
      edge_mesh_float_int64_2d
      curvature_double_int64_3d_dynamic
      k_rings_int64
      neighborhoods_float_int64_2d
      remesh_float_int32
      remesh_float_int64)
    target_link_options(trueform_installed_${link_target} PRIVATE
      "-Wl,-map,${${link_target}_link_map}")
  endforeach()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(link_target IN ITEMS
      edge_mesh_float_int64_2d
      curvature_double_int64_3d_dynamic
      k_rings_int64
      neighborhoods_float_int64_2d
      remesh_float_int32
      remesh_float_int64)
    target_link_options(trueform_installed_${link_target} PRIVATE
      "-Wl,-Map,${${link_target}_link_map}")
  endforeach()
endif()

set(exact_operation_link_targets
  exact_intersection_double_mixed_dynamic
  exact_arrangements_double_int64_dynamic
  exact_boolean_double_mixed_dynamic
  exact_isobands_double_int64_dynamic
  exact_isocontours_double_int64_dynamic
  exact_outer_shell_double_int64_dynamic)
foreach(link_target IN LISTS exact_operation_link_targets)
  set(${link_target}_link_map
    "${CMAKE_CURRENT_BINARY_DIR}/trueform_installed_${link_target}.map")
  if(APPLE)
    target_link_options(trueform_installed_${link_target} PRIVATE
      "-Wl,-map,${${link_target}_link_map}")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(trueform_installed_${link_target} PRIVATE
      "-Wl,-Map,${${link_target}_link_map}")
  endif()
endforeach()

enable_testing()
add_test(NAME trueform_installed_consumer COMMAND trueform_installed_consumer)
add_test(NAME trueform_installed_umbrellas COMMAND trueform_installed_umbrellas)
add_test(NAME trueform_installed_legacy_carriers
  COMMAND trueform_installed_legacy_carriers)
add_test(NAME trueform_installed_mesh_primitives_double_int64_3d
  COMMAND trueform_installed_mesh_primitives_double_int64_3d)
add_test(NAME trueform_installed_mesh_measurements_float_int64_2d
  COMMAND trueform_installed_mesh_measurements_float_int64_2d)
add_test(NAME trueform_installed_mesh_measurements_double_int64_3d
  COMMAND trueform_installed_mesh_measurements_double_int64_3d)
add_test(NAME trueform_installed_boundary_float_int64_2d
  COMMAND trueform_installed_boundary_float_int64_2d)
add_test(NAME trueform_installed_boundary_double_int64_3d
  COMMAND trueform_installed_boundary_double_int64_3d)
add_test(NAME trueform_installed_orientation_double_int64_3d
  COMMAND trueform_installed_orientation_double_int64_3d)
add_test(NAME trueform_installed_triangulate_float_int64_2d
  COMMAND trueform_installed_triangulate_float_int64_2d)
add_test(NAME trueform_installed_triangulate_double_int64_3d
  COMMAND trueform_installed_triangulate_double_int64_3d)
add_test(NAME trueform_installed_connect_edges_to_paths_int32
  COMMAND trueform_installed_connect_edges_to_paths_int32)
add_test(NAME trueform_installed_connect_edges_to_paths_int64
  COMMAND trueform_installed_connect_edges_to_paths_int64)
add_test(NAME trueform_installed_intersects_float_int64_2d
  COMMAND trueform_installed_intersects_float_int64_2d)
add_test(NAME trueform_installed_neighbor_search_float_int64_2d
  COMMAND trueform_installed_neighbor_search_float_int64_2d)
add_test(NAME trueform_installed_gather_ids_float_int64_2d
  COMMAND trueform_installed_gather_ids_float_int64_2d)
add_test(NAME trueform_installed_ray_cast_float_2d
  COMMAND trueform_installed_ray_cast_float_2d)
add_test(NAME trueform_installed_closest_metric_point_pair_float_2d
  COMMAND trueform_installed_closest_metric_point_pair_float_2d)
add_test(NAME trueform_installed_distance_primitive_float_3d
  COMMAND trueform_installed_distance_primitive_float_3d)
add_test(NAME trueform_installed_registration_float_2d
  COMMAND trueform_installed_registration_float_2d)
add_test(NAME trueform_installed_primitive_geometry_float_2d
  COMMAND trueform_installed_primitive_geometry_float_2d)
add_test(NAME trueform_installed_legacy_primitive_normals_abi
  COMMAND trueform_installed_legacy_primitive_normals_abi)
add_test(NAME trueform_installed_obj_float_int64_dynamic
  COMMAND trueform_installed_obj_float_int64_dynamic)
add_test(NAME trueform_installed_stl_int64 COMMAND trueform_installed_stl_int64)
add_test(NAME trueform_installed_mesh_normals_double_int64_3d
  COMMAND trueform_installed_mesh_normals_double_int64_3d)
add_test(NAME trueform_installed_topology_analysis_float_int64_2d
  COMMAND trueform_installed_topology_analysis_float_int64_2d)
add_test(NAME trueform_installed_topology_analysis_double_int64_3d
  COMMAND trueform_installed_topology_analysis_double_int64_3d)
add_test(NAME trueform_installed_topology_links_int64
  COMMAND trueform_installed_topology_links_int64)
add_test(NAME trueform_installed_clean_forms_double_int64_2d
  COMMAND trueform_installed_clean_forms_double_int64_2d)
add_test(NAME trueform_installed_clean_points_float_2d
  COMMAND trueform_installed_clean_points_float_2d)
add_test(NAME trueform_installed_clean_soups_float_int64_2d
  COMMAND trueform_installed_clean_soups_float_int64_2d)
add_test(NAME trueform_installed_split_components_double_int64_2d
  COMMAND trueform_installed_split_components_double_int64_2d)
add_test(NAME trueform_installed_reindex_double_int64_2d
  COMMAND trueform_installed_reindex_double_int64_2d)
add_test(NAME trueform_installed_csg_graph_double_int64
  COMMAND trueform_installed_csg_graph_double_int64)
add_test(NAME trueform_installed_domain_labels_double_int64_dynamic
  COMMAND trueform_installed_domain_labels_double_int64_dynamic)
add_test(NAME trueform_installed_edge_mesh_float_int64_2d
  COMMAND trueform_installed_edge_mesh_float_int64_2d)
add_test(NAME trueform_installed_curvature_double_int64_3d_dynamic
  COMMAND trueform_installed_curvature_double_int64_3d_dynamic)
add_test(NAME trueform_installed_k_rings_int64
  COMMAND trueform_installed_k_rings_int64)
add_test(NAME trueform_installed_neighborhoods_float_int64_2d
  COMMAND trueform_installed_neighborhoods_float_int64_2d)
add_test(NAME trueform_installed_remesh_float_int32
  COMMAND trueform_installed_remesh_float_int32)
add_test(NAME trueform_installed_remesh_float_int64
  COMMAND trueform_installed_remesh_float_int64)
foreach(exact_operation_target IN LISTS exact_operation_link_targets)
  add_test(NAME trueform_installed_${exact_operation_target}
    COMMAND trueform_installed_${exact_operation_target})
endforeach()
if(APPLE OR CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_test(NAME trueform_installed_legacy_carrier_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${legacy_carrier_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckLegacyCarrierLinkMap.cmake")
  set_tests_properties(trueform_installed_legacy_carrier_extraction PROPERTIES
    DEPENDS trueform_installed_legacy_carriers)
  add_test(NAME trueform_installed_registration_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${registration_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckRegistrationLinkMap.cmake")
  set_tests_properties(trueform_installed_registration_extraction PROPERTIES
    DEPENDS trueform_installed_registration_float_2d)
  add_test(NAME trueform_installed_primitive_geometry_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${primitive_geometry_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckPrimitiveGeometryLinkMap.cmake")
  set_tests_properties(trueform_installed_primitive_geometry_extraction
    PROPERTIES DEPENDS trueform_installed_primitive_geometry_float_2d)
  add_test(NAME trueform_installed_mesh_primitives_double_int64_3d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${mesh_primitives_double_int64_3d_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckMeshPrimitivesDoubleInt64LinkMap.cmake")
  set_tests_properties(
    trueform_installed_mesh_primitives_double_int64_3d_extraction
    PROPERTIES DEPENDS trueform_installed_mesh_primitives_double_int64_3d)
  add_test(NAME trueform_installed_mesh_measurements_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${mesh_measurements_float_int64_2d_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckMeshMeasurementsFloatInt64_2DLinkMap.cmake")
  set_tests_properties(
    trueform_installed_mesh_measurements_float_int64_2d_extraction
    PROPERTIES DEPENDS trueform_installed_mesh_measurements_float_int64_2d)
  add_test(NAME trueform_installed_mesh_measurements_double_int64_3d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${mesh_measurements_double_int64_3d_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckMeshMeasurementsDoubleInt64_3DLinkMap.cmake")
  set_tests_properties(
    trueform_installed_mesh_measurements_double_int64_3d_extraction
    PROPERTIES DEPENDS trueform_installed_mesh_measurements_double_int64_3d)
  add_test(NAME trueform_installed_boundary_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${boundary_float_int64_2d_link_map}"
      "-DSHARD_LABEL=float/int64/2D boundary"
      "-DEXPECTED_OBJECT=boundary_float_int64_2d.cpp.o"
      "-DFORBIDDEN_PATTERN=boundary_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_boundary_float_int64_2d_extraction
    PROPERTIES DEPENDS trueform_installed_boundary_float_int64_2d)
  add_test(NAME trueform_installed_boundary_double_int64_3d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${boundary_double_int64_3d_link_map}"
      "-DSHARD_LABEL=double/int64/3D boundary"
      "-DEXPECTED_OBJECT=boundary_double_int64_3d.cpp.o"
      "-DFORBIDDEN_PATTERN=boundary_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_boundary_double_int64_3d_extraction
    PROPERTIES DEPENDS trueform_installed_boundary_double_int64_3d)
  add_test(NAME trueform_installed_orientation_double_int64_3d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${orientation_double_int64_3d_link_map}"
      "-DSHARD_LABEL=double/int64/3D positive orientation"
      "-DEXPECTED_OBJECT=orientation_double_int64_3d.cpp.o"
      "-DFORBIDDEN_PATTERN=orientation_(float|double)_int(32|64)_3d\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(
    trueform_installed_orientation_double_int64_3d_extraction
    PROPERTIES DEPENDS trueform_installed_orientation_double_int64_3d)
  add_test(NAME trueform_installed_triangulate_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${triangulate_float_int64_2d_link_map}"
      "-DSHARD_LABEL=float/int64/2D triangulation"
      "-DEXPECTED_OBJECT=triangulate_float_int64_2d.cpp.o"
      "-DFORBIDDEN_PATTERN=triangulate\\.cpp\\.o|triangulate_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(
    trueform_installed_triangulate_float_int64_2d_extraction
    PROPERTIES DEPENDS trueform_installed_triangulate_float_int64_2d)
  add_test(NAME trueform_installed_triangulate_double_int64_3d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${triangulate_double_int64_3d_link_map}"
      "-DSHARD_LABEL=double/int64/3D triangulation"
      "-DEXPECTED_OBJECT=triangulate_double_int64_3d.cpp.o"
      "-DFORBIDDEN_PATTERN=triangulate\\.cpp\\.o|triangulate_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(
    trueform_installed_triangulate_double_int64_3d_extraction
    PROPERTIES DEPENDS trueform_installed_triangulate_double_int64_3d)
  add_test(NAME trueform_installed_connect_edges_to_paths_int32_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${connect_edges_to_paths_int32_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckConnectEdgesToPathsInt32LinkMap.cmake")
  set_tests_properties(
    trueform_installed_connect_edges_to_paths_int32_extraction
    PROPERTIES DEPENDS trueform_installed_connect_edges_to_paths_int32)
  add_test(NAME trueform_installed_connect_edges_to_paths_int64_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${connect_edges_to_paths_int64_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckConnectEdgesToPathsInt64LinkMap.cmake")
  set_tests_properties(
    trueform_installed_connect_edges_to_paths_int64_extraction
    PROPERTIES DEPENDS trueform_installed_connect_edges_to_paths_int64)
  add_test(NAME trueform_installed_intersects_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${intersects_float_int64_2d_link_map}"
      "-DSHARD_LABEL=float/int64/2D intersects"
      "-DEXPECTED_OBJECT=intersects_float_2d.cpp.o"
      "-DFORBIDDEN_PATTERN=intersects_(float|double|mixed)(|_[23]d)\\.cpp\\.o|neighbor_search_(float|double)(|_[23]d)\\.cpp\\.o|ray_cast_(float|double|mixed)(|_2d)\\.cpp\\.o|ray_cast_(mesh|edge_mesh|point_cloud)_(float|double).*\\.cpp\\.o|gather_ids_(float|double)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_intersects_float_int64_2d_extraction
    PROPERTIES DEPENDS trueform_installed_intersects_float_int64_2d)
  add_test(NAME trueform_installed_neighbor_search_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${neighbor_search_float_int64_2d_link_map}"
      "-DSHARD_LABEL=float/int64/2D neighbor search"
      "-DEXPECTED_OBJECT=neighbor_search_float_2d.cpp.o"
      "-DFORBIDDEN_PATTERN=intersects_(float|double|mixed)(|_[23]d)\\.cpp\\.o|neighbor_search_(float|double)(|_[23]d)\\.cpp\\.o|ray_cast_(float|double|mixed)(|_2d)\\.cpp\\.o|ray_cast_(mesh|edge_mesh|point_cloud)_(float|double).*\\.cpp\\.o|gather_ids_(float|double)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(
    trueform_installed_neighbor_search_float_int64_2d_extraction
    PROPERTIES DEPENDS trueform_installed_neighbor_search_float_int64_2d)
  add_test(NAME trueform_installed_gather_ids_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${gather_ids_float_int64_2d_link_map}"
      "-DSHARD_LABEL=float/int64/2D gather IDs"
      "-DEXPECTED_OBJECT=gather_ids_float_2d.cpp.o"
      "-DFORBIDDEN_PATTERN=intersects_(float|double|mixed)(|_[23]d)\\.cpp\\.o|neighbor_search_(float|double)(|_[23]d)\\.cpp\\.o|ray_cast_(float|double|mixed)(|_2d)\\.cpp\\.o|ray_cast_(mesh|edge_mesh|point_cloud)_(float|double).*\\.cpp\\.o|gather_ids_(float|double)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_gather_ids_float_int64_2d_extraction
    PROPERTIES DEPENDS trueform_installed_gather_ids_float_int64_2d)
  add_test(NAME trueform_installed_ray_cast_float_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${ray_cast_float_2d_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckRayCast2DLinkMap.cmake")
  set_tests_properties(trueform_installed_ray_cast_float_2d_extraction
    PROPERTIES DEPENDS trueform_installed_ray_cast_float_2d)
  add_test(NAME trueform_installed_closest_metric_point_pair_float_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${closest_metric_point_pair_float_2d_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckClosestMetricPointPair2DLinkMap.cmake")
  set_tests_properties(
    trueform_installed_closest_metric_point_pair_float_2d_extraction
    PROPERTIES DEPENDS
      trueform_installed_closest_metric_point_pair_float_2d)
  add_test(NAME trueform_installed_distance_primitive_float_3d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${distance_primitive_float_3d_link_map}"
      "-DSHARD_LABEL=float 3D primitive distance"
      "-DEXPECTED_OBJECT=distance_float_3d.cpp.o"
      "-DFORBIDDEN_PATTERN=distance_(float|double|mixed)(_int(32|64))?_[23]d\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_distance_primitive_float_3d_extraction
    PROPERTIES DEPENDS trueform_installed_distance_primitive_float_3d)

  add_test(NAME trueform_installed_legacy_primitive_normals_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${legacy_primitive_normals_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckLegacyPrimitiveNormalsLinkMap.cmake")
  set_tests_properties(trueform_installed_legacy_primitive_normals_extraction
    PROPERTIES DEPENDS trueform_installed_legacy_primitive_normals_abi)
  add_test(NAME trueform_installed_obj_float_int64_dynamic_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${obj_float_int64_dynamic_link_map}"
      "-DSHARD_LABEL=float/int64 dynamic OBJ IO"
      "-DEXPECTED_OBJECT=obj_float_int64_3d.cpp.o"
      "-DFORBIDDEN_PATTERN=obj_(float|double)_int(32|64)_3d\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_obj_float_int64_dynamic_extraction
    PROPERTIES DEPENDS trueform_installed_obj_float_int64_dynamic)
  add_test(NAME trueform_installed_stl_int64_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${stl_int64_link_map}"
      "-DSHARD_LABEL=int64 STL IO"
      "-DEXPECTED_OBJECT=stl_int64_3d.cpp.o"
      "-DFORBIDDEN_PATTERN=stl_int(32|64)_3d\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_stl_int64_extraction PROPERTIES
    DEPENDS trueform_installed_stl_int64)
  add_test(NAME trueform_installed_mesh_normals_double_int64_3d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${mesh_normals_double_int64_3d_link_map}"
      "-DSHARD_LABEL=double/int64/3D mesh normals"
      "-DEXPECTED_OBJECT=normals_double_int64_3d.cpp.o"
      "-DFORBIDDEN_PATTERN=normals_(float|double)_int(32|64)_3d\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(
    trueform_installed_mesh_normals_double_int64_3d_extraction PROPERTIES
    DEPENDS trueform_installed_mesh_normals_double_int64_3d)
  add_test(NAME trueform_installed_topology_analysis_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${topology_analysis_float_int64_2d_link_map}"
      "-DSHARD_LABEL=float/int64/2D topology analysis"
      "-DEXPECTED_OBJECT=analysis_float_int64_2d.cpp.o"
      "-DFORBIDDEN_PATTERN=analysis_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(
    trueform_installed_topology_analysis_float_int64_2d_extraction PROPERTIES
    DEPENDS trueform_installed_topology_analysis_float_int64_2d)
  add_test(NAME trueform_installed_topology_analysis_double_int64_3d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${topology_analysis_double_int64_3d_link_map}"
      "-DSHARD_LABEL=double/int64/3D topology analysis"
      "-DEXPECTED_OBJECT=analysis_double_int64_3d.cpp.o"
      "-DFORBIDDEN_PATTERN=analysis_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(
    trueform_installed_topology_analysis_double_int64_3d_extraction PROPERTIES
    DEPENDS trueform_installed_topology_analysis_double_int64_3d)
  add_test(NAME trueform_installed_topology_links_int64_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${topology_links_int64_link_map}"
      "-DSHARD_LABEL=int64 typed topology links"
      "-DEXPECTED_OBJECT=topology_links_int64.cpp.o"
      "-DFORBIDDEN_PATTERN=topology_links_int(32|64)\\.cpp\\.o|analysis_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o|normals(_(float|double)_int(32|64)_3d)?\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_topology_links_int64_extraction
    PROPERTIES DEPENDS trueform_installed_topology_links_int64)
  add_test(NAME trueform_installed_clean_forms_double_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${clean_forms_double_int64_2d_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckCleanFormsDoubleInt64_2DLinkMap.cmake")
  set_tests_properties(
    trueform_installed_clean_forms_double_int64_2d_extraction PROPERTIES
    DEPENDS trueform_installed_clean_forms_double_int64_2d)
  add_test(NAME trueform_installed_clean_points_float_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${clean_points_float_2d_link_map}"
      "-DSHARD_LABEL=float/2D point cleaning"
      "-DEXPECTED_OBJECT=clean_points_float_2d.cpp.o"
      "-DFORBIDDEN_PATTERN=clean\\.cpp\\.o|clean_points_(float|double)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_clean_points_float_2d_extraction
    PROPERTIES DEPENDS trueform_installed_clean_points_float_2d)
  add_test(NAME trueform_installed_clean_soups_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${clean_soups_float_int64_2d_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckCleanSoupsFloatInt64_2DLinkMap.cmake")
  set_tests_properties(trueform_installed_clean_soups_float_int64_2d_extraction
    PROPERTIES DEPENDS trueform_installed_clean_soups_float_int64_2d)
  add_test(NAME trueform_installed_split_components_double_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${split_components_double_int64_2d_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckSplitComponentsDoubleInt64_2DLinkMap.cmake")
  set_tests_properties(
    trueform_installed_split_components_double_int64_2d_extraction PROPERTIES
    DEPENDS trueform_installed_split_components_double_int64_2d)
  add_test(NAME trueform_installed_reindex_double_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${reindex_double_int64_2d_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckReindexDoubleInt64_2DLinkMap.cmake")
  set_tests_properties(trueform_installed_reindex_double_int64_2d_extraction
    PROPERTIES DEPENDS trueform_installed_reindex_double_int64_2d)
  add_test(NAME trueform_installed_csg_graph_double_int64_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${csg_graph_double_int64_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckCsgGraphDoubleInt64LinkMap.cmake")
  set_tests_properties(trueform_installed_csg_graph_double_int64_extraction
    PROPERTIES DEPENDS trueform_installed_csg_graph_double_int64)
  add_test(NAME trueform_installed_domain_labels_double_int64_dynamic_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${domain_labels_double_int64_dynamic_link_map}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckDomainLabelsDoubleInt64DynamicLinkMap.cmake")
  set_tests_properties(
    trueform_installed_domain_labels_double_int64_dynamic_extraction PROPERTIES
    DEPENDS trueform_installed_domain_labels_double_int64_dynamic)

  add_test(NAME trueform_installed_edge_mesh_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${edge_mesh_float_int64_2d_link_map}"
      "-DSHARD_LABEL=float/int64/2D edge mesh carrier"
      "-DEXPECTED_OBJECT=edge_mesh_float_int64_2d.cpp.o"
      "-DFORBIDDEN_PATTERN=edge_mesh_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_edge_mesh_float_int64_2d_extraction
    PROPERTIES DEPENDS trueform_installed_edge_mesh_float_int64_2d)
  add_test(NAME trueform_installed_curvature_double_int64_3d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${curvature_double_int64_3d_dynamic_link_map}"
      "-DSHARD_LABEL=double/int64/3D dynamic curvature"
      "-DEXPECTED_OBJECT=curvature_double_int64_3d.cpp.o"
      "-DFORBIDDEN_PATTERN=curvature\\.cpp\\.o|curvature_(float|double)_int(32|64)_3d\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_curvature_double_int64_3d_extraction
    PROPERTIES DEPENDS trueform_installed_curvature_double_int64_3d_dynamic)
  add_test(NAME trueform_installed_k_rings_int64_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${k_rings_int64_link_map}"
      "-DSHARD_LABEL=int64 raw k-rings"
      "-DEXPECTED_OBJECT=neighborhood_int64.cpp.o"
      "-DFORBIDDEN_PATTERN=neighborhood_(int(32|64)|(float|double)_int(32|64)_(2d|3d))\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_k_rings_int64_extraction
    PROPERTIES DEPENDS trueform_installed_k_rings_int64)
  add_test(NAME trueform_installed_neighborhoods_float_int64_2d_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${neighborhoods_float_int64_2d_link_map}"
      "-DSHARD_LABEL=float/int64/2D raw metric neighborhoods"
      "-DEXPECTED_OBJECT=neighborhood_float_int64_2d.cpp.o"
      "-DFORBIDDEN_PATTERN=neighborhood_(int(32|64)|(float|double)_int(32|64)_(2d|3d))\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(
    trueform_installed_neighborhoods_float_int64_2d_extraction PROPERTIES
    DEPENDS trueform_installed_neighborhoods_float_int64_2d)
  add_test(NAME trueform_installed_remesh_float_int32_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${remesh_float_int32_link_map}"
      "-DSHARD_LABEL=float/int32 typed remesh"
      "-DEXPECTED_OBJECT=remesh_float_int32.cpp.o"
      "-DFORBIDDEN_PATTERN=remesh_(float|double)(|_int(32|64))\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_remesh_float_int32_extraction
    PROPERTIES DEPENDS trueform_installed_remesh_float_int32)
  add_test(NAME trueform_installed_remesh_float_int64_extraction
    COMMAND "${CMAKE_COMMAND}"
      "-DLINK_MAP=${remesh_float_int64_link_map}"
      "-DSHARD_LABEL=float/int64 typed remesh"
      "-DEXPECTED_OBJECT=remesh_float_int64.cpp.o"
      "-DFORBIDDEN_PATTERN=remesh_(float|double)(|_int(32|64))\\.cpp\\.o"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckAxisShardLinkMap.cmake")
  set_tests_properties(trueform_installed_remesh_float_int64_extraction
    PROPERTIES DEPENDS trueform_installed_remesh_float_int64)

  function(add_exact_operation_extraction_test target label expected forbidden)
    add_test(NAME trueform_installed_${target}_extraction
      COMMAND "${CMAKE_COMMAND}"
        "-DLINK_MAP=${${target}_link_map}"
        "-DSHARD_LABEL=${label}"
        "-DEXPECTED_OBJECTS_CSV=${expected}"
        "-DFORBIDDEN_PATTERN=${forbidden}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckExactOperationLinkMap.cmake")
    set_tests_properties(trueform_installed_${target}_extraction PROPERTIES
      DEPENDS trueform_installed_${target})
  endfunction()

  add_exact_operation_extraction_test(
    exact_intersection_double_mixed_dynamic
    "double mixed-index/dynamic exact intersection"
    "intersection_curves_double_int32_int64_3d.cpp.o,intersection_curves_double_int64_int64_3d.cpp.o,graph_builders_double_int32_int64_3d.cpp.o,graph_builders_double_int64_int64_3d.cpp.o"
    "intersection_curves_(float|double)_int(32|64)_int(32|64)_3d\\.cpp\\.o|graph_builders_(float|double)_int(32|64)_int(32|64)_3d\\\\.cpp\\\\.o")
  add_exact_operation_extraction_test(
    exact_arrangements_double_int64_dynamic
    "double/int64 dynamic exact arrangements"
    "arrangements_mesh_double_int64_3d.cpp.o,arrangements_polygon_double_int64_3d.cpp.o,graph_builders_double_int64_int64_3d.cpp.o"
    "arrangements_(mesh|polygon)_(float|double)_int(32|64)_3d\\.cpp\\.o|graph_builders_(float|double)_int(32|64)_int(32|64)_3d\\\\.cpp\\\\.o")
  add_exact_operation_extraction_test(
    exact_boolean_double_mixed_dynamic
    "double mixed-index/dynamic exact boolean"
    "boolean_double_int32_int64_3d.cpp.o,graph_builders_double_int32_int64_3d.cpp.o"
    "boolean_(float|double)_int(32|64)_int(32|64)_3d\\.cpp\\.o|graph_builders_(float|double)_int(32|64)_int(32|64)_3d\\\\.cpp\\\\.o")
  add_exact_operation_extraction_test(
    exact_isobands_double_int64_dynamic
    "double/int64 dynamic exact isobands"
    "isobands_double_int64_3d.cpp.o"
    "isobands_(float|double)_int(32|64)_3d\\.cpp\\.o|isobands\\.cpp\\.o|graph_builders_(float|double)_int(32|64)_int(32|64)_3d\\\\.cpp\\\\.o")
  add_exact_operation_extraction_test(
    exact_isocontours_double_int64_dynamic
    "double/int64 dynamic exact isocontours"
    "isocontours_double_int64_3d.cpp.o"
    "isocontours_(float|double)_int(32|64)_3d\\.cpp\\.o|isocontours\\.cpp\\.o|graph_builders_(float|double)_int(32|64)_int(32|64)_3d\\\\.cpp\\\\.o")
  add_exact_operation_extraction_test(
    exact_outer_shell_double_int64_dynamic
    "double/int64 dynamic outer shell"
    "outer_shell_double_int64_3d.cpp.o,graph_builders_double_int64_int64_3d.cpp.o"
    "outer_shell_(float|double)_int(32|64)_3d\\.cpp\\.o|graph_builders_(float|double)_int(32|64)_int(32|64)_3d\\\\.cpp\\\\.o")
endif()
