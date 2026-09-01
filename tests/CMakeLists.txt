# ==============================================================================
# Catch2-based Test Suite
# ==============================================================================

include(CTest)
include(Catch)

# Common test utilities (INTERFACE library)
add_library(trueform_test_common INTERFACE)
target_include_directories(trueform_test_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(trueform_test_common INTERFACE tf::trueform Catch2::Catch2WithMain)
target_compile_definitions(trueform_test_common INTERFACE TF_TEST_DATA_DIR="${TF_TEST_DATA_DIR}")
target_compile_options(trueform_test_common INTERFACE
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj /wd4244 /wd4267 /wd4458 /wd4701>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O1>)

# The compiled build tier every suite reads. It links tf::trueform
# directly, so it does not depend on trueform_test_common and there is no
# cycle.
add_subdirectory(builders)
target_link_libraries(trueform_test_common INTERFACE trueform_test_builders)

# Add module subdirectories
add_subdirectory(core)
add_subdirectory(topology)
add_subdirectory(spatial)
add_subdirectory(geometry)
add_subdirectory(intersect)
add_subdirectory(arrangement)
add_subdirectory(iso)
add_subdirectory(csg)
add_subdirectory(exact)
add_subdirectory(clean)
add_subdirectory(reindex)
add_subdirectory(remesh)
add_subdirectory(io)

set(TF_TEST_TARGETS
  trueform_core_tests
  trueform_topology_tests
  trueform_spatial_tests
  trueform_geometry_tests
  trueform_intersect_tests
  trueform_arrangement_tests
  trueform_arrangement_planes_tests
  trueform_iso_tests
  trueform_csg_tests
  trueform_exact_tests
  trueform_clean_tests
  trueform_reindex_tests
  trueform_remesh_tests
  trueform_io_tests
)

# A merged translation unit must not outgrow the suite's longest single
# one, or the batch buys CPU at the cost of the build's critical path.
# The sources of one batch share a single anonymous namespace, so their
# file-scope helper names must not collide.
set_target_properties(${TF_TEST_TARGETS} PROPERTIES
  UNITY_BUILD ON
  UNITY_BUILD_BATCH_SIZE 4)

# Aggregate target for all Catch2 tests
add_custom_target(trueform_tests DEPENDS ${TF_TEST_TARGETS})
