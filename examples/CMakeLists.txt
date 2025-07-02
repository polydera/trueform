file(GLOB example_sources CONFIGURE_DEPENDS *.cpp)

set(EXAMPLE_EXECUTABLES "")
set(EXAMPLE_TARGETS "")

foreach(src ${example_sources})
  get_filename_component(name ${src} NAME_WE)
  set(alias_target "example-${name}")

  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE tf::trueform)
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)

  add_custom_target(${alias_target} DEPENDS ${name})
  list(APPEND EXAMPLE_TARGETS ${alias_target})
  list(APPEND EXAMPLE_EXECUTABLES ${name})
endforeach()

# Optionally try to find VTK and store result for subdir
find_package(VTK QUIET)
set(TRUEFORM_HAS_VTK ${VTK_FOUND})
if(VTK_FOUND)
  message(STATUS "VTK found (version ${VTK_VERSION}), adding VTK examples.")
  add_subdirectory(vtk)
else()
  message(STATUS "VTK not found — skipping VTK examples.")
endif()

# Combine subdirectory results (populated via PARENT_SCOPE)
list(APPEND EXAMPLE_EXECUTABLES ${VTK_EXAMPLE_EXECUTABLES})
list(APPEND EXAMPLE_TARGETS ${VTK_EXAMPLE_TARGETS})

add_subdirectory(nanoflann)
list(APPEND EXAMPLE_EXECUTABLES ${NANOFLANN_EXAMPLE_EXECUTABLES})
list(APPEND EXAMPLE_TARGETS ${NANOFLANN_EXAMPLE_TARGETS})

if(ENABLE_ASAN)
  foreach(target ${EXAMPLE_EXECUTABLES})
    target_compile_options(${target} PRIVATE -fsanitize=address
                                             -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address)
  endforeach()
endif()

# Group target to build all
add_custom_target(examples DEPENDS ${EXAMPLE_EXECUTABLES})

# Copy compile_commands.json only after 'make examples'
add_custom_command(
  TARGET examples
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/compile_commands.json
          ${CMAKE_SOURCE_DIR}/compile_commands.json
  COMMENT
    "Copying compile_commands.json to project root after building examples")
