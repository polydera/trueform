add_executable(trueform_reindex_tests
    test_reindex_by_mask.cpp
    test_reindex_by_ids.cpp
    test_reindex_on_points.cpp
    test_reindex_concatenated.cpp
    test_reindex_split_components.cpp
)

target_link_libraries(trueform_reindex_tests PRIVATE trueform_test_common)

target_compile_options(trueform_reindex_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4244 /wd4267 /wd4458 /wd4701>
)

catch_discover_tests(trueform_reindex_tests TEST_PREFIX "reindex::")
