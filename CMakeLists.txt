cmake_minimum_required(VERSION 3.15)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type (default Release)" FORCE)
endif()

option(ENABLE_ASAN "Enable Address Sanitizer" OFF)

# Build options (all OFF by default)
option(BUILD_EXAMPLES "Build examples." OFF)
option(BUILD_PYTHON "Build Python bindings." OFF)
option(BUILD_BENCHMARKS "Build benchmarks." OFF)
option(BUILD_VTK "Build VTK integration library." OFF)
option(BUILD_VTK_EXAMPLES "Build VTK examples." OFF)
option(USE_SYSTEM_TBB "Use system-installed TBB." ON)
mark_as_advanced(USE_SYSTEM_TBB)

# Python and Examples/VTK are mutually exclusive (Python needs static TBB, Examples/VTK need system TBB)
if((BUILD_EXAMPLES OR BUILD_VTK OR BUILD_VTK_EXAMPLES) AND BUILD_PYTHON)
  message(FATAL_ERROR "BUILD_EXAMPLES/BUILD_VTK/BUILD_VTK_EXAMPLES and BUILD_PYTHON are mutually exclusive. "
    "Python bindings require static TBB, while VTK requires system TBB.")
endif()

# VTK examples require VTK library
if(BUILD_VTK_EXAMPLES AND NOT BUILD_VTK)
  set(BUILD_VTK ON CACHE BOOL "Build VTK integration library." FORCE)
endif()

# Python requires static TBB
if(BUILD_PYTHON AND USE_SYSTEM_TBB)
  set(USE_SYSTEM_TBB OFF CACHE BOOL "Use system-installed TBB." FORCE)
endif()

# VTK requires system TBB
if(BUILD_VTK AND NOT USE_SYSTEM_TBB)
  set(USE_SYSTEM_TBB ON CACHE BOOL "Use system-installed TBB." FORCE)
endif()

if(ENABLE_ASAN)
  message("Enabling ASAN and DEBUG")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

project(trueform LANGUAGES CXX)

if(MSVC)
  # Enable exception support for MSVC
    add_compile_options(/EHsc)
endif()

add_library(trueform INTERFACE)
add_library(tf::trueform ALIAS trueform)

target_compile_features(trueform INTERFACE cxx_std_17)
include(cmake/GetTBB.cmake)

# Link TBB to trueform target
target_link_libraries(trueform INTERFACE TBB::tbb)
target_include_directories(
  trueform INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:include>)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(BUILD_EXAMPLES)
  add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()
if(BUILD_PYTHON)
  add_subdirectory(python)
endif()
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(BUILD_VTK)
  add_subdirectory(vtk)
endif()

include(GNUInstallDirs)

install(
  TARGETS trueform
  EXPORT trueformTargets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT trueformTargets
  FILE trueformTargets.cmake
  NAMESPACE tf::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trueform)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/trueformConfigVersion.cmake
  VERSION 0.0.0
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/trueformConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/trueformConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trueform)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/trueformConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/trueformConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trueform)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/LICENSE
              ${CMAKE_CURRENT_LIST_DIR}/LICENSE.noncommercial
              ${CMAKE_CURRENT_LIST_DIR}/licenses/LICENSE.TBB
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses)
if(BUILD_PYTHON)
  install(FILES ${CMAKE_CURRENT_LIST_DIR}/licenses/LICENSE.nanobind
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses)
endif()
