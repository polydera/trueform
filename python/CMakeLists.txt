# ==============================================================================
# Python Bindings Configuration
# ==============================================================================
message(STATUS "trueform: Configuring Python bindings")

include(CheckCXXCompilerFlag)

# 1. Compiler Tweaks
# ------------------------------------------------------------------------------
# Suppress warnings from nanobind on newer compilers regarding literal operators
check_cxx_compiler_flag("-Wno-deprecated-literal-operator" SUPPORTS_DEPRECATED_LITERAL_OPERATOR)
if(SUPPORTS_DEPRECATED_LITERAL_OPERATOR)
  add_compile_options(-Wno-deprecated-literal-operator)
endif()

# 2. Dependencies
# ------------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Found Python ${Python_VERSION}")

find_package(nanobind CONFIG REQUIRED)

# 3. Source Collection
# ------------------------------------------------------------------------------
# Explicit file lists — no GLOB. Add new files to the respective
# sources.cmake or headers.cmake when creating new bindings.

# Top-level binding sources (module entry points)
set(BINDING_TOP_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/clean.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cut.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/intersect.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/reindex.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/remesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/spatial.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/topology.cpp
)

# Per-module binding sources
include(${CMAKE_CURRENT_SOURCE_DIR}/src/clean/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/core/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/cut/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/intersect/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/io/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/reindex/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/remesh/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/spatial/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/topology/sources.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/src/util/sources.cmake)

set(BINDING_SOURCES
  ${BINDING_TOP_SOURCES}
  ${MODULE_CLEAN_SOURCES}
  ${MODULE_CORE_SOURCES}
  ${MODULE_CUT_SOURCES}
  ${MODULE_GEOMETRY_SOURCES}
  ${MODULE_INTERSECT_SOURCES}
  ${MODULE_IO_SOURCES}
  ${MODULE_REINDEX_SOURCES}
  ${MODULE_REMESH_SOURCES}
  ${MODULE_SPATIAL_SOURCES}
  ${MODULE_TOPOLOGY_SOURCES}
  ${MODULE_UTIL_SOURCES}
)

# Binding headers
include(${CMAKE_CURRENT_SOURCE_DIR}/include/headers.cmake)

# 4. Target Definition (_trueform)
# ------------------------------------------------------------------------------
message(STATUS "Creating _trueform Python extension module")

nanobind_add_module(
        _trueform
        NOMINSIZE
        NB_STATIC
        ${BINDING_SOURCES}
        ${BINDING_HEADERS}
)

# 5. Linkage & Includes
# ------------------------------------------------------------------------------
# Link against the main C++ library
target_link_libraries(_trueform PRIVATE tf::trueform)

# Includes: Nanobind headers + Local binding headers
target_include_directories(_trueform SYSTEM PRIVATE "${NANOBIND_INCLUDE_DIR}")
target_include_directories(_trueform PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# 6. Compiler Flags & Optimization
# ------------------------------------------------------------------------------
if(MSVC)
  target_compile_options(_trueform PRIVATE /W3 /wd4244 /wd4267 /wd4458 /wd4701)
else()
  # -Wno-nested-anon-types and -Wno-zero-length-array suppress nanobind warnings
  target_compile_options(_trueform PRIVATE -Wall -Wextra -Wpedantic -Wno-nested-anon-types -Wno-zero-length-array)
endif()

# x86 wheels target the AVX2 + FMA (Haswell, 2013) baseline - the lib's SIMD
# paths (core/config/math.hpp) light up there. This makes the wheel REQUIRE an
# AVX2 CPU (it will fault on older ones; there is no runtime fallback). The
# flag-support probe self-skips where the flag is rejected - e.g. ARM
# (macOS arm64), which uses NEON.
if(MSVC)
  target_compile_options(_trueform PRIVATE /arch:AVX2)
else()
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2" TF_HAS_MAVX2)
  if(TF_HAS_MAVX2)
    target_compile_options(_trueform PRIVATE -mavx2 -mfma)
  endif()
endif()

# 7. Output Directory Layout
# ------------------------------------------------------------------------------
# We want the compiled module (.so/.pyd) to sit inside the 'trueform' package folder
# in the build directory, allowing for in-place testing.
set_target_properties(_trueform PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/trueform"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/trueform"
        EXCLUDE_FROM_ALL TRUE
)

# 8. Version Injection
# ------------------------------------------------------------------------------
set(TRUEFORM_VERSION ${PROJECT_VERSION})

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/src/trueform/_version.py"
        "__version__ = \"${TRUEFORM_VERSION}\"\n"
)

# 9. Python Source Management (Copy to Build Dir)
# ------------------------------------------------------------------------------
# Glob pure Python files (only trueform package - tests/examples run from source)
file(GLOB_RECURSE PYTHON_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/trueform/*.py")

# Create a custom target to sync Python files to the build directory
add_custom_target(trueform_copy_python_files ALL
        DEPENDS
        ${PYTHON_SOURCE_FILES}
)

add_custom_command(
        TARGET trueform_copy_python_files POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/src/trueform"
        "${CMAKE_CURRENT_BINARY_DIR}/src/trueform"
        COMMENT "Syncing Python sources to build directory"
        VERBATIM
)

# Ensure the binary extension waits for the python file copy
add_dependencies(_trueform trueform_copy_python_files)

# 10. Final Target
# ------------------------------------------------------------------------------
# Helper target for users to run explicitly
add_custom_target(trueform_python DEPENDS _trueform)
message(STATUS "Python bindings configured. Build with 'cmake --build . --target trueform_python'")

# 11. Installation
# ------------------------------------------------------------------------------
# Install the compiled extension
install(TARGETS _trueform LIBRARY DESTINATION trueform)

# Install the generated version file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/src/trueform/_version.py" DESTINATION trueform)

# Install the pure Python package
install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/trueform"
        DESTINATION .
        FILES_MATCHING PATTERN "*.py"
)

# 12. C++ Headers & CMake Config for Wheel Distribution
# ------------------------------------------------------------------------------
# Install the header-only C++ library inside the Python package.
# Users can find headers via: python -c "import trueform; print(trueform.get_include())"
install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/"
        DESTINATION trueform/include
)

# Generate CMake config and install for find_package(trueform) against the
# copied header-only library (operator-new backend, independent of _trueform.so).
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/trueform-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/trueform-config.cmake"
        @ONLY
)
install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/trueform-config.cmake"
        DESTINATION trueform/cmake_config
)

if(TF_USE_MIMALLOC AND NOT TF_USE_SYSTEM_MIMALLOC)
  install(
          FILES "${CMAKE_CURRENT_SOURCE_DIR}/../licenses/LICENSE.mimalloc"
          DESTINATION trueform/licenses
  )
endif()

# Install Conan test_package for conan create verification
install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../conan/trueform/test_package/"
        DESTINATION trueform/conan/test_package
        PATTERN "build" EXCLUDE
        PATTERN "CMakeUserPresets.json" EXCLUDE
)

