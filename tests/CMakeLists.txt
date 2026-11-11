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

# The house int128 class is the only 128-bit path under MSVC; a compiler with
# a native __int128 also builds the tet suites over the class, so every run
# executes the arithmetic MSVC runs.
if(MSVC)
  set(tf_test_int128_class_default OFF)
else()
  set(tf_test_int128_class_default ON)
endif()
option(TF_TEST_INT128_CLASS
  "Also build the tet suites over the house int128 class (TF_FORCE_INT128_FALLBACK)."
  ${tf_test_int128_class_default})

# The class build's own common tier. Every translation unit of a class
# target carries the define, its builders included, so it links neither
# trueform_test_common nor the shared builder library.
if(TF_TEST_INT128_CLASS AND NOT MSVC)
  add_library(trueform_test_int128_class INTERFACE)
  target_include_directories(trueform_test_int128_class INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/builders)
  target_link_libraries(trueform_test_int128_class INTERFACE
    tf::trueform Catch2::Catch2WithMain)
  target_compile_definitions(trueform_test_int128_class INTERFACE
    TF_TEST_DATA_DIR="${TF_TEST_DATA_DIR}" TF_FORCE_INT128_FALLBACK)
  target_compile_options(trueform_test_int128_class INTERFACE
    -O1 $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)
endif()

# A case that runs a parallel primitive at the machine's full concurrency
# carries [parallel] and runs alone under ctest; the rest of its suite
# shares the machine. A [parallel] case that cannot state its claim on the
# machine skips.
#
# A random sweep, a worker-count ladder or a seeded corpus loop carries the
# hidden tag [.fuzz]: ctest never discovers it, and a suite binary runs it
# only when asked, `<suite> "[fuzz]"` for all of them. A spec naming a tag
# would match a hidden case, so the [parallel] spec excludes [fuzz] itself.
#
# A skip is read off the summary Catch2 prints when every case of the run
# skipped — the test-cases row carrying the skipped column alone, which no
# run holding a failure or a pass prints. The exit code says nothing: 4 is
# both "every case skipped" and four failed assertions.
function(tf_discover_tests target prefix)
  catch_discover_tests(${target} TEST_PREFIX "${prefix}"
    TEST_SPEC "~[parallel]")
  catch_discover_tests(${target} TEST_PREFIX "${prefix}"
    TEST_SPEC "[parallel]~[fuzz]"
    PROPERTIES RUN_SERIAL TRUE
               SKIP_REGULAR_EXPRESSION "test cases: [0-9]+ [|] [0-9]+ skipped")
endfunction()

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
add_subdirectory(linalg)
add_subdirectory(csg)
add_subdirectory(fill)
add_subdirectory(exact)
add_subdirectory(clean)
add_subdirectory(reindex)
add_subdirectory(remesh)
add_subdirectory(io)
add_subdirectory(volume)
add_subdirectory(tetmesh)

set(TF_TEST_TARGETS
  trueform_core_tests
  trueform_topology_tests
  trueform_spatial_tests
  trueform_geometry_tests
  trueform_intersect_tests
  trueform_arrangement_tests
  trueform_arrangement_planes_tests
  trueform_iso_tests
  trueform_linalg_tests
  trueform_csg_tests
  trueform_fill_tests
  trueform_exact_tests
  trueform_clean_tests
  trueform_reindex_tests
  trueform_remesh_tests
  trueform_io_tests
  trueform_volume_tests
  trueform_tetmesh_tests
)

# A merged translation unit must not outgrow the suite's longest single
# one, or the batch buys CPU at the cost of the build's critical path.
# The sources of one batch share a single anonymous namespace, so their
# file-scope helper names must not collide.
set_target_properties(${TF_TEST_TARGETS} PROPERTIES
  UNITY_BUILD ON
  UNITY_BUILD_BATCH_SIZE 4)

# The class targets stay out of the unity batches: their builder units are
# independent by design, as in trueform_test_builders.
set(TF_TEST_INT128_CLASS_TARGETS)
foreach(target trueform_exact_int128_class_tests
               trueform_tetmesh_int128_class_tests
               trueform_topology_int128_class_tests)
  if(TARGET ${target})
    list(APPEND TF_TEST_INT128_CLASS_TARGETS ${target})
  endif()
endforeach()

# Aggregate target for all Catch2 tests
add_custom_target(trueform_tests
  DEPENDS ${TF_TEST_TARGETS} ${TF_TEST_INT128_CLASS_TARGETS})
