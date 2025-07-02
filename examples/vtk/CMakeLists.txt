if(NOT TRUEFORM_HAS_VTK)
  return()  # No VTK, no build
endif()

file(GLOB vtk_example_sources CONFIGURE_DEPENDS *.cpp)

set(VTK_EXAMPLE_EXECUTABLES "")
set(VTK_EXAMPLE_TARGETS "")

foreach(src ${vtk_example_sources})
  get_filename_component(name "${src}" NAME_WE)
  set(alias_target "example-vtk-${name}")

  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE tf::trueform ${VTK_LIBRARIES})
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)

  add_custom_target(${alias_target} DEPENDS ${name})
  list(APPEND VTK_EXAMPLE_TARGETS ${alias_target})
  list(APPEND VTK_EXAMPLE_EXECUTABLES ${name})
endforeach()

# Export back to parent scope
set(VTK_EXAMPLE_TARGETS "${VTK_EXAMPLE_TARGETS}" PARENT_SCOPE)
set(VTK_EXAMPLE_EXECUTABLES "${VTK_EXAMPLE_EXECUTABLES}" PARENT_SCOPE)

