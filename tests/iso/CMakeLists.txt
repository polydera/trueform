# The iso tier's one test home. Two kinds live here: test_iso_face_cuts
# names the per-face cut directly, while test_isocontours covers the
# PUBLIC iso features.
add_executable(trueform_iso_tests
  # machinery — the per-face cut
  test_iso_face_cuts.cpp
  # public features
  test_isocontours.cpp
)

target_link_libraries(trueform_iso_tests PRIVATE trueform_test_common)

target_compile_options(trueform_iso_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_iso_tests TEST_PREFIX "iso::")
