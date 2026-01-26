# ==============================================================================
# Dependency Management (CPM)
# ==============================================================================

# Add the current directory to the module path to locate CPM.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}")
include(CPM)

# Dependency Versions
set(TF_CATCH2_VERSION "3.5.2" CACHE STRING "Catch2 version for testing")

# Allow the project to use system libraries if the global flag is set
set(CPM_USE_LOCAL_PACKAGES ${TF_USE_SYSTEM_LIBS})

# ------------------------------------------------------------------------------
# Dependency: oneTBB (Threading Building Blocks)
# ------------------------------------------------------------------------------
if(TF_BUILD_PYTHON)
  # Python wheels bundle static TBB - force fetch, ignore system TBB
  set(CPM_USE_LOCAL_PACKAGES OFF)
  CPMAddPackage(
    NAME                TBB
    GITHUB_REPOSITORY   oneapi-src/oneTBB
    GIT_TAG             v${TF_TBB_VERSION}
    OPTIONS
    "BUILD_SHARED_LIBS OFF"
    "TBB_BUILD ON"
    "TBB_INSTALL ON"
    "TBB_TEST OFF"
    "TBB_STRICT OFF"
    "TBBMALLOC_BUILD OFF"
    "TBBMALLOC_PROXY_BUILD OFF"
    "TBBBIND_BUILD OFF"
    "TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON"
    "CMAKE_POLICY_VERSION_MINIMUM 3.5"
  )
  # Reset for other dependencies
  set(CPM_USE_LOCAL_PACKAGES ${TF_USE_SYSTEM_LIBS})
else()
  # C++ builds require system TBB
  find_package(TBB CONFIG REQUIRED)
endif()

# ------------------------------------------------------------------------------
# Dependency: Nanobind (Python Bindings)
# ------------------------------------------------------------------------------
if(TF_BUILD_PYTHON)

  # Ensure a Python environment is available before fetching bindings
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  CPMAddPackage(
          NAME                nanobind
          GITHUB_REPOSITORY   wjakob/nanobind
          GIT_TAG             v${TF_NANOBIND_VERSION}
          OPTIONS
          "NB_USE_SUBMODULE_DEPS ON"
          "NB_TEST OFF"
          "NB_TEST_SHARED_BUILD OFF"
  )

  # Manual include workarounds:
  # Sometimes nanobind config is not immediately picked up by CMake scope
  # if fetched via CPM, so we manually check and include it.
  if(EXISTS "${CPM_PACKAGE_nanobind_SOURCE_DIR}/cmake/nanobind-config.cmake")
    include("${CPM_PACKAGE_nanobind_SOURCE_DIR}/cmake/nanobind-config.cmake")
  endif()

endif()

# ------------------------------------------------------------------------------
# Dependency: Catch2 (Testing Framework)
# ------------------------------------------------------------------------------
if(TF_BUILD_TESTS)
  CPMAddPackage(
    NAME Catch2
    GITHUB_REPOSITORY catchorg/Catch2
    VERSION ${TF_CATCH2_VERSION}
    OPTIONS
      "CATCH_INSTALL_DOCS OFF"
      "CATCH_INSTALL_EXTRAS OFF"
  )
  list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
endif()