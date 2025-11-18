# Python bindings for trueform

message(STATUS "Configuring Python bindings for trueform")

# Find Python
find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)
message(STATUS "Found Python ${Python_VERSION}")

# Fetch nanobind
message(STATUS "Fetching nanobind...")
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.0.0)
FetchContent_MakeAvailable(nanobind)
message(STATUS "nanobind fetched successfully")

# Glob all source files
file(GLOB_RECURSE BINDING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR} binding sources")

# Glob all headers
file(GLOB_RECURSE BINDING_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR} binding headers")

# Create single Python extension module
message(STATUS "Creating _trueform Python extension module")

nanobind_add_module(
  _trueform
  NOMINSIZE
  NB_STATIC
  ${BINDING_SOURCES}
  ${BINDING_HEADERS}
)

target_include_directories(_trueform SYSTEM PRIVATE
  ${nanobind_SOURCE_DIR}/include
)

# Link against trueform (available from parent)
target_link_libraries(_trueform PRIVATE tf::trueform)

# Include directories
target_include_directories(
  _trueform
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Platform-specific compiler flags
if(MSVC)
  target_compile_options(_trueform PRIVATE /W4)
else()
  target_compile_options(_trueform PRIVATE -Wall -Wextra)
endif()

# Release build optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckCXXCompilerFlag)

  check_cxx_compiler_flag("-march=native" SUPPORTS_MARCH_NATIVE)
  if(SUPPORTS_MARCH_NATIVE)
    message(STATUS "Compiler supports -march=native for Python bindings")
    target_compile_options(_trueform PRIVATE -march=native)
  else()
    message(WARNING "Compiler does NOT support -march=native")
  endif()

endif()


target_compile_options(_trueform PRIVATE -ftime-trace)

# Exclude from ALL target
set_target_properties(_trueform PROPERTIES EXCLUDE_FROM_ALL TRUE)

# Set output directory for _trueform to be inside trueform package
set_target_properties(
  _trueform
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/trueform
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/trueform
)

# Glob all Python files to track them as dependencies
file(GLOB_RECURSE PYTHON_SOURCE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/trueform/*.py")
file(GLOB_RECURSE PYTHON_TEST_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.py")
file(GLOB_RECURSE PYTHON_EXAMPLE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/**/*.py")
# file(GLOB_RECURSE PYTHON_BENCHMARK_FILES
#   "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.py")

message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR} Python source files")
message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR} Python test files")
message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR} Python example files")
# message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR} Python benchmark files")

# Create a custom target that depends on all Python files
add_custom_target(
  copy_python_files ALL
  DEPENDS
    ${PYTHON_SOURCE_FILES}
    ${PYTHON_TEST_FILES}
    ${PYTHON_EXAMPLE_FILES}
    # ${PYTHON_BENCHMARK_FILES}
)

# Copy Python source files to build directory whenever they change
add_custom_command(
  TARGET copy_python_files
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/trueform
      ${CMAKE_CURRENT_BINARY_DIR}/trueform
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
      ${CMAKE_CURRENT_BINARY_DIR}/tests
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/examples
      ${CMAKE_CURRENT_BINARY_DIR}/examples
  # COMMAND
  #   ${CMAKE_COMMAND} -E copy_directory
  #     ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
  #     ${CMAKE_CURRENT_BINARY_DIR}/benchmarks
  COMMENT "Copying Python source, test, example, and benchmark files to build directory"
  VERBATIM
)

# Make the python target depend on copying Python files
add_dependencies(_trueform copy_python_files)

# Create 'python' target to build bindings
add_custom_target(python DEPENDS _trueform)
message(
  STATUS
    "Python bindings configured. Build with 'cmake --build . --target python'"
)

# Install the extension module and Python files
install(TARGETS _trueform LIBRARY DESTINATION trueform)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/trueform
  DESTINATION .
  FILES_MATCHING
  PATTERN "*.py"
)

