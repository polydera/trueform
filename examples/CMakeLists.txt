cmake_minimum_required(VERSION 3.15)

project(trueform_examples LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find trueform if not already available
if(NOT TARGET tf::trueform)
  find_package(trueform REQUIRED)
endif()

# Object library with reusable example utilities (cylinder, etc.)
add_library(example_utils OBJECT
  src/cylinder.cpp
)
target_link_libraries(example_utils PUBLIC tf::trueform)
target_include_directories(example_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(example_utils PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /wd4458 /wd4701>
)

# Mesh assembly example
add_executable(mesh_assembly src/mesh_assembly.cpp)
target_link_libraries(mesh_assembly PRIVATE tf::trueform example_utils)
target_include_directories(mesh_assembly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(mesh_assembly PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /wd4458 /wd4701>
)

# Alignment example
add_executable(alignment src/alignment.cpp)
target_link_libraries(alignment PRIVATE tf::trueform)
target_compile_definitions(alignment PRIVATE
  TRUEFORM_DATA_DIR="${trueform_SOURCE_DIR}"
)
target_compile_options(alignment PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /wd4458 /wd4701>
)

# Collect all example targets
add_custom_target(examples DEPENDS mesh_assembly alignment)
