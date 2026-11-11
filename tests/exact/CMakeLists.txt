add_executable(trueform_exact_tests
  test_det2_sign.cpp
  test_dot_sign.cpp
  test_edge_parameter.cpp
  test_edge_lattice.cpp
  test_classify_wedge.cpp
  test_dyadic_blend.cpp
  test_dyadic_ratio.cpp
  test_wide_dyadic_ratio.cpp
  test_is_between_on_segment.cpp
  test_rebase_parameter.cpp
  test_canonical_plane.cpp
  test_difference_image.cpp
  test_insphere.cpp
  test_snap_parameter_to_edge.cpp
  test_int128_msvc.cpp
  test_int256.cpp
  test_int512.cpp
  test_segment_hits_aabb.cpp
  test_door_placement.cpp
  test_pt_converter_quantum.cpp
)

target_link_libraries(trueform_exact_tests PRIVATE trueform_test_common)

target_compile_options(trueform_exact_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_exact_tests TEST_PREFIX "exact::")

# TF_FORCE_INT128_FALLBACK is a whole-program switch: int256 and int512 are
# non-template classes whose limb is tf::exact::uint128, so a program whose
# units disagree about the macro holds two layouts behind one set of
# symbols. The sources that test the classes take a program of their own,
# every unit of it carrying the macro, as the tet class targets do. It is
# the regression guard for the arithmetic MSVC runs, so it is built
# wherever the suite is.
# The difference image is read by every filter over the 128-bit type, so it
# runs on whichever one its program holds.
add_executable(trueform_exact_int128_class_tests
  test_difference_image.cpp
  test_int128_fallback.cpp
  test_wide_int_differential.cpp
)

target_include_directories(trueform_exact_int128_class_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(trueform_exact_int128_class_tests PRIVATE
  tf::trueform Catch2::Catch2WithMain)

target_compile_definitions(trueform_exact_int128_class_tests PRIVATE
  TF_FORCE_INT128_FALLBACK)

target_compile_options(trueform_exact_int128_class_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O1 -Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj /W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_exact_int128_class_tests
  TEST_PREFIX "exact_int128_class::")
