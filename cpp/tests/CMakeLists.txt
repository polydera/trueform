include(CTest)
include(Catch)

add_library(trueform_cpp_test_common INTERFACE)
target_include_directories(trueform_cpp_test_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(trueform_cpp_test_common INTERFACE
  tf::trueform_cpp
  Catch2::Catch2WithMain
)
target_compile_options(trueform_cpp_test_common INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj /W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

add_subdirectory(arrangement)
add_subdirectory(core)
add_subdirectory(clean)
add_subdirectory(csg)
add_subdirectory(geometry)
add_subdirectory(intersect)
add_subdirectory(io)
add_subdirectory(iso)
add_subdirectory(reindex)
add_subdirectory(remesh)
add_subdirectory(spatial)
add_subdirectory(topology)

get_filename_component(TRUEFORM_CPP_TEST_SOURCE_ROOT
  "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE
)

add_custom_target(trueform_cpp_test_boundary
  COMMAND ${CMAKE_COMMAND}
    "-DTRUEFORM_SOURCE_DIR=${TRUEFORM_CPP_TEST_SOURCE_ROOT}"
    -P "${TRUEFORM_CPP_TEST_SOURCE_ROOT}/cmake/CheckCppFacadeTestBoundary.cmake"
  COMMAND ${CMAKE_COMMAND}
    "-DTRUEFORM_SOURCE_DIR=${TRUEFORM_CPP_TEST_SOURCE_ROOT}"
    "-DTRUEFORM_BOUNDARY_CHECKER=${TRUEFORM_CPP_TEST_SOURCE_ROOT}/cmake/CheckCppFacadeTestBoundary.cmake"
    "-DTRUEFORM_BOUNDARY_TEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/boundary-self-test"
    -P "${TRUEFORM_CPP_TEST_SOURCE_ROOT}/cmake/TestCppFacadeTestBoundary.cmake"
  VERBATIM
)

add_custom_target(trueform_cpp_tests DEPENDS
  trueform_cpp_test_boundary
  trueform_cpp_arrangement_tests
  trueform_cpp_clean_tests
  trueform_cpp_core_tests
  trueform_cpp_csg_tests
  trueform_cpp_geometry_tests
  trueform_cpp_intersect_tests
  trueform_cpp_io_tests
  trueform_cpp_iso_tests
  trueform_cpp_reindex_tests
  trueform_cpp_remesh_tests
  trueform_cpp_spatial_tests
  trueform_cpp_topology_tests
)
