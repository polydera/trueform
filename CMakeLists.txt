cmake_minimum_required(VERSION 3.15)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type (default Release)" FORCE)
endif()

option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(BUILD_BENCHMARKS "Build benchmarks." OFF)

# Configure build options based on benchmark mode
if(BUILD_BENCHMARKS)
  set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples." FORCE)
  set(BUILD_PYTHON OFF CACHE BOOL "Build Python bindings." FORCE)
  set(USE_SYSTEM_TBB ON CACHE BOOL "Use system-installed TBB." FORCE)
else()
  option(BUILD_EXAMPLES "Build examples." OFF)
  option(BUILD_PYTHON "Build Python bindings." ON)
  option(USE_SYSTEM_TBB "Use system-installed TBB if found." OFF)
  mark_as_advanced(USE_SYSTEM_TBB)
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
