add_executable(boolean_topology_test boolean_topology_test.cpp)
target_link_libraries(boolean_topology_test PRIVATE tf::trueform)
target_compile_options(boolean_topology_test PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_executable(mod_tree_test mod_tree_test.cpp)
target_link_libraries(mod_tree_test PRIVATE tf::trueform)
target_compile_definitions(mod_tree_test PRIVATE
  TRUEFORM_DATA_DIR="${trueform_SOURCE_DIR}"
)
target_compile_options(mod_tree_test PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

add_custom_target(trueform_tests
  DEPENDS
    boolean_topology_test
    mod_tree_test
)
