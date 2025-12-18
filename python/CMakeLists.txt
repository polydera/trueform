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
  target_compile_options(_trueform PRIVATE /W4)
else()
  target_compile_options(_trueform PRIVATE -Wall -Wextra)
endif()

# Enable aggressive optimizations in Release mode
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  check_cxx_compiler_flag("-march=native" SUPPORTS_MARCH_NATIVE)
  if(SUPPORTS_MARCH_NATIVE)
    message(STATUS "Compiler supports -march=native for Python bindings")
    target_compile_options(_trueform PRIVATE -march=native)
  else()
    message(WARNING "Compiler does NOT support -march=native")
  endif()
endif()

# 7. Output Directory Layout
# ------------------------------------------------------------------------------
# We want the compiled module (.so/.pyd) to sit inside the 'trueform' package folder
# in the build directory, allowing for in-place testing.
set_target_properties(_trueform PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/trueform"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/trueform"
        EXCLUDE_FROM_ALL TRUE
)

# 8. Runtime Dependencies (DLLs/Shared Libs)
# ------------------------------------------------------------------------------
# Copy TBB shared library next to the extension module so it can be found at runtime.
if(NOT ${TF_USE_STATIC_TBB})
  message(STATUS "Configuring TBB share library copy for Python bindings ${TF_USE_STATIC_TBB}")
  install(FILES $<TARGET_FILE:TBB::tbb> DESTINATION _trueform)

  add_custom_command(
          TARGET _trueform POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:TBB::tbb>
          $<TARGET_FILE_DIR:_trueform>
          COMMENT "Copying TBB shared library to executable directory"
  )

  if(APPLE)
    set_target_properties(_trueform PROPERTIES
            MACOSX_RPATH ON
            BUILD_RPATH     "@loader_path"
            INSTALL_RPATH   "@loader_path"
    )
  endif()

  if(UNIX AND NOT APPLE)
    set_target_properties(_trueform PROPERTIES
            BUILD_RPATH   "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
    )
  endif()
endif()

# 9. Version Injection
# ------------------------------------------------------------------------------
# If built via scikit-build-core, use that version. Otherwise, use dev placeholder.
if(DEFINED SKBUILD_PROJECT_VERSION)
  set(TRUEFORM_VERSION ${SKBUILD_PROJECT_VERSION})
else()
  set(TRUEFORM_VERSION "0.0.0-dev")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/trueform/_version.py"
        "__version__ = \"${TRUEFORM_VERSION}\"\n"
)

# 10. Python Source Management (Copy to Build Dir)
# ------------------------------------------------------------------------------
# Glob pure Python files
file(GLOB_RECURSE PYTHON_SOURCE_FILES    "${CMAKE_CURRENT_SOURCE_DIR}/trueform/*.py")
file(GLOB_RECURSE PYTHON_TEST_FILES      "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.py")
file(GLOB_RECURSE PYTHON_EXAMPLE_FILES   "${CMAKE_CURRENT_SOURCE_DIR}/examples/**/*.py")
# file(GLOB_RECURSE PYTHON_BENCHMARK_FILES "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.py")

# Create a custom target to sync Python files to the build directory
add_custom_target(copy_python_files ALL
        DEPENDS
        ${PYTHON_SOURCE_FILES}
        ${PYTHON_TEST_FILES}
        ${PYTHON_EXAMPLE_FILES}
)

add_custom_command(
        TARGET copy_python_files POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/trueform"
        "${CMAKE_CURRENT_BINARY_DIR}/trueform"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        "${CMAKE_CURRENT_BINARY_DIR}/tests"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/examples"
        "${CMAKE_CURRENT_BINARY_DIR}/examples"
        COMMENT "Syncing Python sources, tests, and examples to build directory"
        VERBATIM
)

# Ensure the binary extension waits for the python file copy
add_dependencies(_trueform copy_python_files)

# 11. Final Target
# ------------------------------------------------------------------------------
# Helper target for users to run explicitly
add_custom_target(python DEPENDS _trueform)
message(STATUS "Python bindings configured. Build with 'cmake --build . --target python'")

# 12. Installation
# ------------------------------------------------------------------------------
# Install the compiled extension
install(TARGETS _trueform LIBRARY DESTINATION trueform)

# Install the generated version file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/trueform/_version.py" DESTINATION trueform)

# Install the pure Python package
install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/trueform"
        DESTINATION .
        FILES_MATCHING PATTERN "*.py"
)