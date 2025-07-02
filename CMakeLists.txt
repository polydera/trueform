cmake_minimum_required(VERSION 3.14)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type (default Release)" FORCE)
endif()


option(ENABLE_ASAN "Enable Address Sanitizer" OFF)

if(ENABLE_ASAN)
  message("Enabling ASAN and DEBUG")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

project(trueform LANGUAGES CXX)

add_library(trueform INTERFACE)
add_library(tf::trueform ALIAS trueform)

target_compile_features(trueform INTERFACE cxx_std_17)

include(FetchContent)

find_package(TBB QUIET)

if(NOT TBB_FOUND)
  message(STATUS "TBB not found, using FetchContent to download and build it.")

  FetchContent_Declare(
    tbb
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG v2021.11.0)
  set(TBB_TEST
      OFF
      CACHE BOOL "Disable TBB tests" FORCE)
  FetchContent_MakeAvailable(tbb)
endif()

# Link TBB to trueform target
target_link_libraries(trueform INTERFACE TBB::tbb)
target_include_directories(
  trueform INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:include>)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(examples EXCLUDE_FROM_ALL)

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
