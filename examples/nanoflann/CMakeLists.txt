# Include the FetchContent module to manage dependencies
include(FetchContent)

# Declare the nanoflann dependency from its official Git repository.
# Using a specific GIT_TAG ensures that the build is reproducible.
FetchContent_Declare(
  nanoflann
  GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
  GIT_TAG        v1.5.5 # Using a recent, stable version of nanoflann
)

# This command will download the source code (if not already fetched)
# and make the targets defined in its CMakeLists.txt (i.e., 'nanoflann')
# available for linking in our project.
FetchContent_MakeAvailable(nanoflann)

# --- Original loop modified to link against nanoflann ---

file(GLOB vtk_example_sources CONFIGURE_DEPENDS *.cpp)

set(NANO_FLANN_EXAMPLE_EXECUTABLES "")
set(NANO_FLANN_EXAMPLE_TARGETS "")

foreach(src ${vtk_example_sources})
  get_filename_component(name "${src}" NAME_WE)
  set(alias_target "example-nanoflann-${name}")

  add_executable(${name} ${src})

  target_link_libraries(${name} PRIVATE tf::trueform nanoflann)

  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)

  add_custom_target(${alias_target} DEPENDS ${name})
  list(APPEND NANO_FLANN_EXAMPLE_TARGETS ${alias_target})
  list(APPEND NANO_FLANN_EXAMPLE_EXECUTABLES ${name})
endforeach()

# Export back to parent scope (this part is unchanged)
set(NANO_FLANN_EXAMPLE_TARGETS "${NANO_FLANN_EXAMPLE_TARGETS}" PARENT_SCOPE)
set(NANO_FLANN_EXAMPLE_EXECUTABLES "${NANO_FLANN_EXAMPLE_EXECUTABLES}" PARENT_SCOPE)
