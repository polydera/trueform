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
target_compile_options(trueform_test_common INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/bigobj /wd4244 /wd4267 /wd4458 /wd4701>)

# Add module subdirectories
add_subdirectory(core)
add_subdirectory(topology)
add_subdirectory(spatial)
add_subdirectory(geometry)
add_subdirectory(intersect)
add_subdirectory(cut)
add_subdirectory(clean)
add_subdirectory(reindex)
add_subdirectory(io)

# Aggregate target for all Catch2 tests
add_custom_target(trueform_tests DEPENDS
  trueform_core_tests
  trueform_topology_tests
  trueform_spatial_tests
  trueform_geometry_tests
  trueform_intersect_tests
  trueform_cut_tests
  trueform_clean_tests
  trueform_reindex_tests
  trueform_io_tests
)
