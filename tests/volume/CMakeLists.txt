add_executable(trueform_volume_tests
  test_volume.cpp
  test_volume_carrier.cpp
  test_resampled_volume.cpp
  test_isocontours.cpp
  test_isosurface.cpp
  test_mesh_sdf_banded.cpp
  test_mesh_sdf_sign.cpp
  test_sharpness.cpp
)

target_link_libraries(trueform_volume_tests PRIVATE trueform_test_common)

target_compile_options(trueform_volume_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_volume_tests TEST_PREFIX "volume::")
