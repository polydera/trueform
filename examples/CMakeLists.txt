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

  list(APPEND EXAMPLE_EXECUTABLES ${VTK_EXAMPLE_EXECUTABLES})
  list(APPEND EXAMPLE_TARGETS ${VTK_EXAMPLE_TARGETS})
else()
  message(STATUS "VTK not found — skipping VTK examples.")
endif()

add_subdirectory(nanoflann)
list(APPEND EXAMPLE_EXECUTABLES ${NANOFLANN_EXAMPLE_EXECUTABLES})
list(APPEND EXAMPLE_TARGETS ${NANOFLANN_EXAMPLE_TARGETS})

find_package(CGAL QUIET)
set(TRUEFORM_HAS_CGAL ${CGAL_FOUND})
if(CGAL_FOUND)
  message(STATUS "GCAL found (version ${CGAL_VERSION}), adding CGAL examples.")
  add_subdirectory(cgal)

  list(APPEND EXAMPLE_EXECUTABLES ${CGAL_EXAMPLE_EXECUTABLES})
  list(APPEND EXAMPLE_TARGETS ${CGAL_EXAMPLE_TARGETS})
else()
  message(STATUS "CGAL not found — skipping CGAL examples.")
endif()

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
