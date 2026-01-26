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
# Note: GLOB_RECURSE is convenient but requires re-running CMake if files are added.
file(GLOB_RECURSE BINDING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE BINDING_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

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
  target_compile_options(_trueform PRIVATE -Wall -Wextra)
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

# Install CMake config for find_package(trueform) support
install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/trueform-config.cmake"
        DESTINATION trueform/cmake_config
)
