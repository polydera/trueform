file(GLOB cgal_example_sources CONFIGURE_DEPENDS *.cpp)

set(CGAL_EXAMPLE_EXECUTABLES "")
set(CGAL_EXAMPLE_TARGETS "")

foreach(src ${cgal_example_sources})
  get_filename_component(name "${src}" NAME_WE)
  set(alias_target "example-cgal-${name}")

  add_executable(${name} ${src})

  target_link_libraries(${name} PRIVATE tf::trueform CGAL::CGAL)

  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)

  add_custom_target(${alias_target} DEPENDS ${name})
  list(APPEND CGAL_EXAMPLE_TARGETS ${alias_target})
  list(APPEND CGAL_EXAMPLE_EXECUTABLES ${name})
endforeach()

# Export back to parent scope
set(CGAL_EXAMPLE_TARGETS "${CGAL_EXAMPLE_TARGETS}" PARENT_SCOPE)
set(CGAL_EXAMPLE_EXECUTABLES "${CGAL_EXAMPLE_EXECUTABLES}" PARENT_SCOPE)

