cmake_minimum_required(VERSION 3.16)

# CMP0148: Use the new behavior for finding Python interpreters and libraries
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# ==============================================================================
# Platform configuration
# ==============================================================================
if(APPLE AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
  message(STATUS
          "MACOSX_DEPLOYMENT_TARGET not set in environment. "
          "Defaulting CMAKE_OSX_DEPLOYMENT_TARGET to ${CMAKE_OSX_DEPLOYMENT_TARGET}"
  )
endif()

# ==============================================================================
# Project Configuration
# ==============================================================================
project(trueform VERSION 0.9.13 LANGUAGES CXX)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/trueform/version.hpp
    @ONLY
)

message(STATUS "CMAKE_OSX_DEPLOYMENT_TARGET = ${CMAKE_OSX_DEPLOYMENT_TARGET}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Options & Variables
# ==============================================================================
include(CMakeDependentOption)

# 1. Build features
option(TF_BUILD_PYTHON            "Build Python bindings." OFF)
option(TF_BUILD_TYPESCRIPT        "Build TypeScript/WASM bindings." OFF)
option(TF_BUILD_BENCHMARKS        "Build benchmarks." OFF)
option(TF_BUILD_EXAMPLES          "Build examples." OFF)
option(TF_BUILD_VTK_INTEGRATION   "Build trueform VTK integration." OFF)
option(TF_ENABLE_ASAN             "Enable Address Sanitizer" OFF)
option(TF_USE_SYSTEM_LIBS         "Use system installed libraries where possible." ON)
option(TF_ENABLE_IPO              "Enable Interprocedural Optimization (LTO)." OFF)
# mimalloc backend. One master switch + one acquisition sub-option:
#   TF_USE_MIMALLOC        master on/off. OFF => operator new/delete backend,
#                          no dependency, no TF_WITH_MIMALLOC define, no link.
#                          (auto-forced OFF under ASan and TypeScript, below.)
#   TF_USE_SYSTEM_MIMALLOC find the user's mimalloc (must be MI_OVERRIDE=OFF)
#                          instead of fetching+vendoring our own. Only meaningful
#                          when TF_USE_MIMALLOC is ON.
# Everything downstream (define, link, fetch/find, license, installed-config
# dependency) is gated on TF_USE_MIMALLOC, so -DTF_USE_MIMALLOC=OFF disables it
# end to end (e.g. the NuGet build, which ships the operator-new backend).
option(TF_USE_MIMALLOC            "Use mimalloc as the internal allocation backend (explicit, no global override)." ON)
option(TF_USE_SYSTEM_MIMALLOC     "find_package the user's own mimalloc (must be MI_OVERRIDE=OFF) instead of fetching+vendoring our own." OFF)

# 2. Dependent options
## TF_BUILD_VTK_EXAMPLES: Build VTK examples only if VTK integration is enabled.
cmake_dependent_option(
        TF_BUILD_VTK_EXAMPLES
        "Build VTK examples."
        OFF
        "TF_BUILD_VTK_INTEGRATION"
        OFF
)

# TypeScript/WASM builds are cross-compiled with Emscripten.
# Force incompatible options OFF and use static TBB via CPM.
if(TF_BUILD_TYPESCRIPT)
  if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "TF_BUILD_TYPESCRIPT requires Emscripten. Configure with: emcmake cmake ...")
  endif()
  set(TF_BUILD_PYTHON          OFF CACHE BOOL "" FORCE)
  set(TF_BUILD_BENCHMARKS      OFF CACHE BOOL "" FORCE)
  set(TF_BUILD_EXAMPLES        OFF CACHE BOOL "" FORCE)
  set(TF_BUILD_VTK_INTEGRATION OFF CACHE BOOL "" FORCE)
  set(TF_BUILD_TESTS           OFF CACHE BOOL "" FORCE)
  set(TF_ENABLE_ASAN           OFF CACHE BOOL "" FORCE)
  set(TF_USE_SYSTEM_LIBS       OFF CACHE BOOL "" FORCE)
endif()


# Dependency Versions (Cache variables allow user overrides)
set(TF_TBB_VERSION "2022.3.0" CACHE STRING
        "Version of TBB to fetch for Python builds.")

set(TF_NANOBIND_VERSION "2.0.0" CACHE STRING
        "Version of nanobind to fetch for Python builds.")

set(TF_MIMALLOC_VERSION "3.3.2" CACHE STRING
        "Version (git tag without 'v') of mimalloc to fetch when not found on the system.")

# ==============================================================================
# Compiler Flags
# ==============================================================================
if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(/EHsc) # Enable standard C++ exceptions for clang-cl
endif()

if(TF_ENABLE_ASAN)
  message(STATUS "trueform: Address Sanitizer enabled.")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

if(TF_ENABLE_ASAN AND TF_USE_MIMALLOC)
  message(STATUS "TF_ENABLE_ASAN ON: forcing TF_USE_MIMALLOC OFF.")
  set(TF_USE_MIMALLOC OFF CACHE BOOL "" FORCE)
endif()

if(TF_BUILD_TYPESCRIPT AND TF_USE_MIMALLOC)
  # wasm gets mimalloc via Emscripten -sMALLOC=mimalloc; a second copy collides on mi_*.
  message(STATUS "TF_BUILD_TYPESCRIPT ON: forcing TF_USE_MIMALLOC OFF (Emscripten provides mimalloc).")
  set(TF_USE_MIMALLOC OFF CACHE BOOL "" FORCE)
endif()

if(TF_ENABLE_IPO)
  message(STATUS "trueform: Interprocedural Optimization (LTO) enabled.")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "IPO/LTO is not supported by the current compiler. Disbling TF_ENABLE_IPO.")
    set(TF_ENABLE_IPO OFF)
  endif()
endif()

# ==============================================================================
# Target Definition (Interface Library)
# ==============================================================================
add_library(trueform INTERFACE)
add_library(tf::trueform ALIAS trueform)

# Fetch/Locate dependencies (See cmake/ThirdParty.cmake)
include(cmake/ThirdParty.cmake)

# Linkage & Includes
target_link_libraries(trueform INTERFACE TBB::tbb)
target_compile_features(trueform INTERFACE cxx_std_17)

if(TF_USE_MIMALLOC)
  target_compile_definitions(trueform INTERFACE TF_WITH_MIMALLOC)
  target_link_libraries(trueform INTERFACE mimalloc-static)
endif()

target_include_directories(trueform INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ==============================================================================
# Subdirectories
# ==============================================================================
if(TF_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(TF_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(TF_BUILD_TYPESCRIPT)
  add_subdirectory(typescript)
endif()

if(TF_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(TF_BUILD_VTK_INTEGRATION)
  add_subdirectory(vtk)
endif()

# ==============================================================================
# Installation & Packaging
# ==============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# 1. Install the library target and export the set
install(
        TARGETS trueform
        EXPORT trueformTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 2. Install Headers
install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 3. Register the export set (creates trueformTargets.cmake)
install(
        EXPORT trueformTargets
        FILE trueformTargets.cmake
        NAMESPACE tf::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trueform
)

# 4. Generate and Install Package Config Files
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/trueformConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/trueformConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/trueformConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trueform
)

install(
        FILES
        ${CMAKE_CURRENT_BINARY_DIR}/trueformConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/trueformConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trueform
)

# 5. Install Licenses
install(
        FILES
        ${CMAKE_CURRENT_LIST_DIR}/LICENSE
        ${CMAKE_CURRENT_LIST_DIR}/LICENSE.noncommercial
        ${CMAKE_CURRENT_LIST_DIR}/licenses/LICENSE.TBB
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses
)

if(TF_USE_MIMALLOC AND NOT TF_USE_SYSTEM_MIMALLOC)
  install(
          FILES ${CMAKE_CURRENT_LIST_DIR}/licenses/LICENSE.mimalloc
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses
  )
endif()

if(TF_BUILD_PYTHON)
  install(
          FILES ${CMAKE_CURRENT_LIST_DIR}/licenses/LICENSE.nanobind
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses
  )
endif()

# Tests
option(TF_BUILD_TESTS "Build test targets." OFF)
if(TF_BUILD_TESTS)
  add_subdirectory(tests)
endif()
