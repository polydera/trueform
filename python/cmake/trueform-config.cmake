# trueform-config.cmake
# Self-contained CMake configuration for pip-installed trueform
#
# Usage:
#   execute_process(
#     COMMAND "${Python_EXECUTABLE}" -c "import trueform; print(trueform.get_cmake_dir())"
#     OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE trueform_ROOT)
#   find_package(trueform CONFIG REQUIRED)
#   target_link_libraries(mytarget PRIVATE tf::trueform)

include_guard(GLOBAL)

# Compute package root (one level up from cmake/)
get_filename_component(TF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(TF_DIR "${TF_DIR}" PATH)
set(TF_DIR ${TF_DIR} CACHE INTERNAL "")

# TBB is required - must be installed on the system
include(CMakeFindDependencyMacro)
find_dependency(TBB CONFIG REQUIRED)

# Create the interface target
if (NOT TARGET tf::trueform)
  add_library(tf::trueform INTERFACE IMPORTED)
  set_target_properties(tf::trueform PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TF_DIR}/include"
    INTERFACE_LINK_LIBRARIES TBB::tbb
  )
endif()
